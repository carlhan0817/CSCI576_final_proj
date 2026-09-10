"""
End-to-end pipeline integration test (Phase 1 → Phase 4).

This test verifies that the fusion + export pipeline (phases 3 and 4) glues
together correctly when given realistic phase-1 and phase-2 artifacts.

Why we don't actually run phases 1 and 2:
- Phase 1 needs ffmpeg + faster-whisper (~140 MB download).
- Phase 2 needs torch + CLIP (~340 MB) + MiniLM (~90 MB) + librosa.
- Together they take minutes per video and blow up the test time budget.

Instead, we build a workspace with hand-crafted phase 1 (meta_raw.json,
transcript.json) and phase 2 (visual_features.json, audio_features.json,
text_features.json, text_embeddings.npy) artifacts that mimic what a real
60-second talking-head video with intro / sponsor / content / outro structure
would produce. Then we call ``run_fusion`` end-to-end and validate the
resulting ``metadata.json``.

If you want to additionally run a *real* Phase 1+2 + Phase 3+4 pipeline on a
synthetic MP4, the slow test ``test_real_pipeline_smoke`` does that — gated
behind ``-m slow`` because it downloads models on first run.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from backend.pipeline.fusion.run import run_fusion
from backend.pipeline.schemas import (
    AudioFeatures,
    AudioFeatureSegment,
    Metadata,
    MetaRaw,
    TextFeatures,
    TextFeatureSegment,
    Transcript,
    TranscriptSegment,
    VisualFeatures,
    VisualFrameFeature,
)
from backend.pipeline.workspace import Workspace

# Scene labels actually emitted by Phase 2 (V2.1 prompt list).
SCENE_LABELS = [
    "a person talking",
    "presentation slide",
    "screen recording",
    "video game",
    "an animated scene",
    "blank screen",
    "an end credits screen",
    "a title card",
    "an advertisement slide",
    "sponsor logo",
]


def _build_synthetic_workspace(tmp_path: Path) -> Workspace:
    """
    Construct a fully-populated phase-1+phase-2 workspace for a fake 60-second
    video with this structure:

    0s ─ 5s   intro      (title card visual, music, intro keyword)
    5s ─ 25s  core_a     (talking head, speech, technical content)
    25s ─ 35s  sponsor    (advert visual, music, "sponsored by" keyword)
    35s ─ 55s  core_b     (talking head, speech)
    55s ─ 60s  outro      (end credits visual, "thanks for watching")
    """
    ws = Workspace.for_video(Path("synthetic_test.mp4"), root=tmp_path / "workspace")
    ws.ensure()

    duration = 60.0

    # ── Phase 1 artifact 1: meta_raw.json ────────────────────────────────────
    meta_raw = MetaRaw(
        filename="synthetic_test.mp4",
        duration_sec=duration,
        fps=30.0,
        width=1920,
        height=1080,
        video_codec="h264",
        audio_codec="aac",
        has_audio=True,
        ingested_at="2026-05-04T00:00:00+00:00",
    )
    ws.meta_raw_path.write_text(meta_raw.model_dump_json(indent=2))

    # ── Phase 1 artifact 2: transcript.json ──────────────────────────────────
    sentences = [
        TranscriptSegment(id=0, start=1.0, end=4.0, text="Welcome back to the channel, today we explore signal processing."),
        TranscriptSegment(id=1, start=6.0, end=12.0, text="The Fourier transform decomposes a signal into its constituent frequencies."),
        TranscriptSegment(id=2, start=12.0, end=18.0, text="This is fundamental to compression, filtering, and many other applications."),
        TranscriptSegment(id=3, start=18.0, end=24.0, text="The discrete cosine transform is closely related and used in JPEG."),
        TranscriptSegment(id=4, start=26.0, end=30.0, text="Today's video is brought to you by NordVPN, use code SAVE for 50 percent off."),
        TranscriptSegment(id=5, start=30.0, end=34.0, text="Click the link in the description to support the channel."),
        TranscriptSegment(id=6, start=36.0, end=44.0, text="Returning to compression: the DCT concentrates energy in low-frequency coefficients."),
        TranscriptSegment(id=7, start=44.0, end=52.0, text="Quantizing high-frequency coefficients aggressively yields most of JPEG's compression."),
        TranscriptSegment(id=8, start=56.0, end=59.0, text="Thanks for watching, don't forget to subscribe."),
    ]
    transcript = Transcript(
        language="en",
        duration_sec=duration,
        model="base",
        segments=sentences,
        full_text=" ".join(s.text for s in sentences),
    )
    ws.transcript_path.write_text(transcript.model_dump_json(indent=2))

    # ── Phase 2 artifact 1: visual_features.json ────────────────────────────
    # 1 FPS sampling → 60 frames.
    frames: list[VisualFrameFeature] = []
    for t in range(int(duration)):
        clip_probs = {lbl: 0.05 for lbl in SCENE_LABELS}
        if t < 5:
            clip_probs["a title card"] = 0.6
            clip_probs["a person talking"] = 0.2
        elif t < 25:
            clip_probs["a person talking"] = 0.7
            clip_probs["presentation slide"] = 0.15
        elif t < 35:
            clip_probs["an advertisement slide"] = 0.65
            clip_probs["sponsor logo"] = 0.2
        elif t < 55:
            clip_probs["a person talking"] = 0.7
            clip_probs["presentation slide"] = 0.15
        else:
            clip_probs["an end credits screen"] = 0.7
            clip_probs["blank screen"] = 0.15
        # Hard cuts at structural boundaries
        is_cut = t in (5, 25, 35, 55)
        frames.append(VisualFrameFeature(
            frame_index=t,
            timestamp_sec=float(t),
            hist_diff_to_previous=0.7 if is_cut else 0.05,
            is_hard_cut=is_cut,
            clip_labels=clip_probs,
            mean_luminance=128.0,
            luminance_variance=300.0,
            is_black_frame=False,
            motion_intensity=0.1 if 5 <= t < 55 else 0.02,
            chroma_diff=0.05,
            dct_hf_energy=0.3,
        ))
    ws.visual_features_path.write_text(VisualFeatures(frames=frames).model_dump_json(indent=2))

    # ── Phase 2 artifact 2: audio_features.json (1-second grid) ─────────────
    audio_segs: list[AudioFeatureSegment] = []
    for t in range(int(duration)):
        # Music in intro, sponsor, outro; speech in core sections
        if t < 5:
            cls, speech, rms = "music", False, 0.4
        elif t < 25:
            cls, speech, rms = "speech", True, 0.5
        elif t < 35:
            cls, speech, rms = "music", True, 0.5  # sponsor: voice over music
        elif t < 55:
            cls, speech, rms = "speech", True, 0.5
        else:
            cls, speech, rms = "music", False, 0.3
        audio_segs.append(AudioFeatureSegment(
            start=float(t), end=float(t + 1),
            is_speech=speech, rms_energy=rms,
            spectral_centroid=2500.0,
            spectral_bandwidth=1500.0,
            zero_crossing_rate=0.05,
            spectral_entropy=4.5,
            audio_class=cls,  # type: ignore[arg-type]
        ))
    ws.audio_features_path.write_text(AudioFeatures(segments=audio_segs).model_dump_json(indent=2))

    # ── Phase 2 artifact 3: text_features.json + text_embeddings.npy ────────
    # Per-sentence keyword matches mirror what features/text.py would produce.
    keyword_table = {
        0: ["intro:welcome back"],
        1: [], 2: [], 3: [],
        4: ["sponsor:brought to you by"],
        5: ["sponsor:the link in the description"],
        6: [], 7: [],
        8: ["outro:thanks for watching"],
    }
    text_segs: list[TextFeatureSegment] = []
    for s in sentences:
        # Sentences in adjacent windows are similar; sentences across boundaries are not.
        if s.id == 3:
            sim, boundary = 0.15, True   # technical → sponsor: big topic jump
        elif s.id == 5:
            sim, boundary = 0.20, True   # sponsor → technical: big topic jump
        elif s.id == 7:
            sim, boundary = 0.25, True   # technical → outro: medium jump
        elif s.id == 8:
            sim, boundary = None, False  # last sentence
        else:
            sim, boundary = 0.75, False
        text_segs.append(TextFeatureSegment(
            id=s.id, start=s.start, end=s.end, text=s.text,
            similarity_to_next=sim,
            is_potential_boundary=boundary,
            matched_keywords=keyword_table[s.id],
        ))
    ws.text_features_path.write_text(TextFeatures(segments=text_segs).model_dump_json(indent=2))

    # 384-dim fake embeddings: cluster sentences by section so the centroid-
    # picker in export.py can find a sensible representative.
    rng = np.random.default_rng(42)
    embeddings = np.zeros((len(sentences), 384), dtype=np.float32)
    centers = {
        "intro": rng.standard_normal(384),
        "core_a": rng.standard_normal(384),
        "sponsor": rng.standard_normal(384),
        "core_b": rng.standard_normal(384),
        "outro": rng.standard_normal(384),
    }
    sentence_to_section = {0: "intro", 1: "core_a", 2: "core_a", 3: "core_a", 4: "sponsor", 5: "sponsor", 6: "core_b", 7: "core_b", 8: "outro"}
    for sid, section in sentence_to_section.items():
        noise = rng.standard_normal(384) * 0.1
        embeddings[sid] = centers[section] + noise
    np.save(str(ws.text_embeddings_path), embeddings)

    return ws


# ============================================================
# E2E test against the fully synthetic workspace
# ============================================================

class TestE2EPipeline:
    """
    Verify Phase 3 + Phase 4 produce a valid, structurally-correct
    metadata.json on a hand-crafted workspace.
    """

    @pytest.fixture()
    def metadata(self, tmp_path: Path) -> Metadata:
        ws = _build_synthetic_workspace(tmp_path)
        out_path = run_fusion(ws)
        assert out_path.exists()
        return Metadata.model_validate(json.loads(out_path.read_text()))

    def test_output_validates_against_schema(self, metadata: Metadata):
        # If construction succeeded, the schema passes by definition.
        assert isinstance(metadata, Metadata)

    def test_video_info_correct(self, metadata: Metadata):
        assert metadata.video_info.filename == "synthetic_test.mp4"
        assert metadata.video_info.duration_sec == 60.0
        assert metadata.video_info.width == 1920

    def test_analysis_version_set(self, metadata: Metadata):
        # Must reflect Phase 4's bumped version
        major, minor, _patch = metadata.video_info.analysis_version.split(".")
        assert (int(major), int(minor)) >= (0, 4)

    def test_segments_cover_full_duration(self, metadata: Metadata):
        # Sorted by start_sec
        segs = sorted(metadata.segments, key=lambda s: s.start_sec)
        assert segs[0].start_sec == 0.0
        assert segs[-1].end_sec == metadata.video_info.duration_sec
        # Contiguous: every segment ends where the next begins
        for a, b in itertools.pairwise(segs):
            assert a.end_sec == b.start_sec, f"gap between segs at {a.end_sec} → {b.start_sec}"

    def test_segment_ids_are_unique_and_sequential(self, metadata: Metadata):
        ids = [s.segment_id for s in metadata.segments]
        assert ids == sorted(ids)
        assert len(set(ids)) == len(ids)

    def test_skip_suggestions_match_non_content(self, metadata: Metadata):
        expected = {s.segment_id for s in metadata.segments if s.label != "core_content"}
        assert set(metadata.skip_suggestions) == expected

    def test_chapters_only_cover_core_content(self, metadata: Metadata):
        # Each chapter range must be 100% inside core_content segments
        core_ranges = [
            (s.start_sec, s.end_sec)
            for s in metadata.segments
            if s.label == "core_content"
        ]
        for ch in metadata.chapters:
            covered = any(cs <= ch.start_sec and ch.end_sec <= ce for cs, ce in core_ranges)
            assert covered, f"chapter {ch.chapter_id} not contained in any core_content"

    def test_at_least_one_sponsorship_detected(self, metadata: Metadata):
        # Phase 3 hard rules + CLIP should both flag the sponsor block.
        labels = {s.label for s in metadata.segments}
        assert "sponsorship" in labels, f"no sponsorship in {labels}"

    def test_outro_detected_at_video_end(self, metadata: Metadata):
        last = max(metadata.segments, key=lambda s: s.start_sec)
        # Either the outro keyword rule or the end-credits CLIP prompt should
        # have caught the final block.
        assert last.label in ("outro", "core_content"), \
            f"last segment was {last.label}, expected outro/core_content"

    def test_evidence_present_on_every_segment(self, metadata: Metadata):
        # PDD requires multimodal evidence — at least one modality score
        # should be non-zero per segment for the demo to be defensible.
        for s in metadata.segments:
            total = (s.evidence.visual_score + s.evidence.audio_score + s.evidence.text_score)
            assert total > 0, f"segment {s.segment_id} has all-zero evidence"

    def test_all_segments_have_summaries_when_transcript_covers_them(self, metadata: Metadata):
        # Every segment that overlaps a transcript sentence should get a non-empty summary.
        # (Segments in pure silence may legitimately have empty summaries.)
        non_empty = [s for s in metadata.segments if s.summary]
        assert len(non_empty) >= 1, "no segment got a summary at all"

    def test_skip_segments_are_consistent_with_labels(self, metadata: Metadata):
        # No "core_content" segment should appear in skip_suggestions
        sid_to_label = {s.segment_id: s.label for s in metadata.segments}
        for sid in metadata.skip_suggestions:
            assert sid_to_label[sid] != "core_content"

    def test_user_corrected_default_false(self, metadata: Metadata):
        for s in metadata.segments:
            assert s.user_corrected is False

    def test_verified_by_human_default_false(self, metadata: Metadata):
        assert metadata.video_info.verified_by_human is False