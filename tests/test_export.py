"""
Tests for backend/pipeline/fusion/export.py (Phase 4: Metadata Export).

These tests cover all 5 sub-tasks defined in PDD V2.1 §4.4:
    1. Schema validation (pydantic)
    2. Chapter generation (with representative-sentence titles)
    3. Skip suggestions
    4. Per-segment summaries (representative sentence by embedding similarity)
    5. Version stamping (analysis_version)

We avoid the heavy upstream stack — there is no Whisper, CLIP, or MiniLM here.
Inputs are constructed directly from the schemas, embeddings are tiny fake
arrays, and we verify the JSON output is valid against ``Metadata``.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List

import numpy as np
import pytest
from pydantic import ValidationError

from backend.pipeline.workspace import Workspace
from backend.pipeline.schemas import (
    Metadata,
    MetaRaw,
    Segment,
    SegmentEvidence,
    Transcript,
    TranscriptSegment,
)
from backend.pipeline.fusion.export import (
    ANALYSIS_VERSION,
    _pick_representative_sentence,
    _sentences_in_window,
    attach_summaries,
    build_chapters,
    build_metadata,
    build_skip_suggestions,
    export_metadata,
)


# ============================================================
# Fixtures
# ============================================================

def _make_segment(
    sid: int,
    start: float,
    end: float,
    label: str = "core_content",
    confidence: float = 0.7,
    summary: str = "",
    visual: float = 0.5,
    audio: float = 0.5,
    text: float = 0.5,
    triggered: list[str] | None = None,
) -> Segment:
    """Construct a Segment with sane defaults for testing."""
    return Segment(
        segment_id=sid,
        start_sec=start,
        end_sec=end,
        label=label,  # type: ignore[arg-type]
        confidence=confidence,
        evidence=SegmentEvidence(
            visual_score=visual,
            audio_score=audio,
            text_score=text,
            triggered_rules=triggered or [],
        ),
        summary=summary,
        user_corrected=False,
    )


def _make_transcript_seg(sid: int, start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(id=sid, start=start, end=end, text=text)


def _meta_raw(duration: float = 60.0) -> MetaRaw:
    return MetaRaw(
        filename="dummy.mp4",
        duration_sec=duration,
        fps=30.0,
        width=1920,
        height=1080,
        video_codec="h264",
        audio_codec="aac",
        has_audio=True,
        ingested_at="2026-05-04T00:00:00+00:00",
    )


@pytest.fixture()
def workspace(tmp_path: Path) -> Workspace:
    ws = Workspace.for_video(Path("dummy.mp4"), root=tmp_path / "workspace")
    ws.ensure()
    return ws


# ============================================================
# Sub-task 4: representative-sentence picker
# ============================================================

class TestSentencesInWindow:
    def test_full_containment_returned(self):
        sentences = [_make_transcript_seg(0, 1.0, 2.0, "inside")]
        out = _sentences_in_window(sentences, 0.0, 5.0)
        assert len(out) == 1

    def test_no_overlap_excluded(self):
        sentences = [_make_transcript_seg(0, 10.0, 12.0, "after window")]
        out = _sentences_in_window(sentences, 0.0, 5.0)
        assert out == []

    def test_partial_overlap_at_start_included(self):
        # Sentence spans the window's left edge (window starts at 5).
        sentences = [_make_transcript_seg(0, 3.0, 6.0, "spans start")]
        out = _sentences_in_window(sentences, 5.0, 10.0)
        assert len(out) == 1

    def test_partial_overlap_at_end_included(self):
        sentences = [_make_transcript_seg(0, 4.0, 7.0, "spans end")]
        out = _sentences_in_window(sentences, 0.0, 5.0)
        assert len(out) == 1

    def test_touching_boundary_excluded(self):
        # A sentence ending exactly at the window's start is NOT included.
        sentences = [_make_transcript_seg(0, 3.0, 5.0, "ends at start")]
        out = _sentences_in_window(sentences, 5.0, 10.0)
        assert out == []


class TestPickRepresentativeSentence:
    def test_empty_returns_empty_string(self):
        assert _pick_representative_sentence([], None) == ""

    def test_no_embeddings_falls_back_to_longest(self):
        sents = [
            _make_transcript_seg(0, 0, 1, "short"),
            _make_transcript_seg(1, 1, 2, "this is a much longer sentence that wins"),
            _make_transcript_seg(2, 2, 3, "mid length sentence"),
        ]
        assert _pick_representative_sentence(sents, embeddings=None).startswith("this is a much")

    def test_with_embeddings_picks_centroid_neighbor(self):
        # Three "topical" embeddings around (1, 0) and one outlier at (0, 1).
        # The representative should be one of the three on the cluster, not the outlier.
        emb = np.array([
            [1.0, 0.0],   # id 0 — on cluster
            [0.95, 0.05], # id 1 — closest to centroid
            [0.9, 0.1],   # id 2 — on cluster
            [0.0, 1.0],   # id 3 — outlier
        ], dtype=np.float32)
        sents = [
            _make_transcript_seg(0, 0, 1, "topical A"),
            _make_transcript_seg(1, 1, 2, "topical B"),
            _make_transcript_seg(2, 2, 3, "topical C"),
            _make_transcript_seg(3, 3, 4, "outlier"),
        ]
        rep = _pick_representative_sentence(sents, embeddings=emb)
        assert rep != "outlier"
        assert rep.startswith("topical")

    def test_truncation_to_max_chars(self):
        long_text = "x" * 500
        sent = _make_transcript_seg(0, 0, 1, long_text)
        rep = _pick_representative_sentence([sent], embeddings=None, max_chars=50)
        assert len(rep) == 50

    def test_id_out_of_embedding_range_falls_back(self):
        # Sentence id 99 has no embedding row — should still produce a sensible
        # answer via fallback rather than crash.
        emb = np.zeros((2, 4), dtype=np.float32)
        sents = [_make_transcript_seg(99, 0, 1, "the only sentence")]
        rep = _pick_representative_sentence(sents, embeddings=emb)
        assert rep == "the only sentence"


# ============================================================
# Sub-task 4 wrapper: attach_summaries
# ============================================================

class TestAttachSummaries:
    def test_summary_replaced_with_representative(self):
        seg = _make_segment(0, 0.0, 5.0, summary="placeholder")
        sents = [_make_transcript_seg(0, 1.0, 2.0, "real sentence")]
        out = attach_summaries([seg], sents, embeddings=None)
        assert out[0].summary == "real sentence"

    def test_empty_window_yields_empty_summary(self):
        seg = _make_segment(0, 100.0, 105.0, summary="had something")
        # transcript only covers 0–10 → no overlap with [100, 105)
        sents = [_make_transcript_seg(0, 0.0, 10.0, "early")]
        out = attach_summaries([seg], sents, embeddings=None)
        assert out[0].summary == ""

    def test_other_fields_preserved(self):
        seg = _make_segment(
            7, 3.0, 9.0,
            label="sponsorship", confidence=0.92,
            visual=0.4, audio=0.6, text=0.8,
            triggered=["sponsor:sponsored by"],
        )
        sents = [_make_transcript_seg(0, 4.0, 5.0, "ad copy")]
        out = attach_summaries([seg], sents, embeddings=None)
        assert out[0].segment_id == 7
        assert out[0].label == "sponsorship"
        assert out[0].confidence == 0.92
        assert out[0].evidence.text_score == 0.8
        assert out[0].evidence.triggered_rules == ["sponsor:sponsored by"]

    def test_input_not_mutated(self):
        seg = _make_segment(0, 0.0, 5.0, summary="ORIGINAL")
        attach_summaries([seg], [_make_transcript_seg(0, 1, 2, "new")], embeddings=None)
        assert seg.summary == "ORIGINAL"  # caller's object unchanged


# ============================================================
# Sub-task 2: chapters
# ============================================================

class TestBuildChapters:
    def test_no_segments_no_chapters(self):
        assert build_chapters([], [], None) == []

    def test_single_core_content_yields_one_chapter(self):
        segs = [_make_segment(0, 0.0, 10.0, label="core_content")]
        sents = [_make_transcript_seg(0, 0.0, 5.0, "topic introduction sentence")]
        chapters = build_chapters(segs, sents, None)
        assert len(chapters) == 1
        assert chapters[0].start_sec == 0.0
        assert chapters[0].end_sec == 10.0

    def test_adjacent_core_content_merged(self):
        segs = [
            _make_segment(0, 0.0, 5.0, label="core_content"),
            _make_segment(1, 5.0, 10.0, label="core_content"),
            _make_segment(2, 10.0, 15.0, label="core_content"),
        ]
        chapters = build_chapters(segs, [], None)
        assert len(chapters) == 1
        assert chapters[0].start_sec == 0.0
        assert chapters[0].end_sec == 15.0

    def test_non_content_breaks_chapter(self):
        segs = [
            _make_segment(0, 0.0, 5.0, label="core_content"),
            _make_segment(1, 5.0, 10.0, label="sponsorship"),
            _make_segment(2, 10.0, 20.0, label="core_content"),
        ]
        chapters = build_chapters(segs, [], None)
        assert len(chapters) == 2
        assert chapters[0].end_sec == 5.0
        assert chapters[1].start_sec == 10.0
        # chapter_ids are sequential
        assert chapters[0].chapter_id == 0
        assert chapters[1].chapter_id == 1

    def test_only_non_content_yields_no_chapters(self):
        segs = [
            _make_segment(0, 0.0, 5.0, label="intro"),
            _make_segment(1, 5.0, 10.0, label="sponsorship"),
            _make_segment(2, 10.0, 15.0, label="outro"),
        ]
        assert build_chapters(segs, [], None) == []

    def test_title_from_transcript_when_available(self):
        segs = [_make_segment(0, 0.0, 10.0, label="core_content")]
        sents = [_make_transcript_seg(0, 0.0, 5.0, "Introducing the topic.")]
        chapters = build_chapters(segs, sents, None)
        assert "topic" in chapters[0].title.lower()

    def test_title_fallback_when_no_transcript(self):
        segs = [_make_segment(0, 0.0, 10.0, label="core_content")]
        chapters = build_chapters(segs, [], None)
        assert chapters[0].title == "Chapter 1"

    def test_title_truncated_at_80_chars(self):
        # Single very long sentence — chapter title must respect the cap.
        segs = [_make_segment(0, 0.0, 10.0, label="core_content")]
        long = "this is a very long sentence " * 20  # ~600 chars
        sents = [_make_transcript_seg(0, 0.0, 5.0, long)]
        chapters = build_chapters(segs, sents, None)
        # ≤ 80 because we trim, replace tail with "..." if overflow
        assert len(chapters[0].title) <= 80


# ============================================================
# Sub-task 3: skip suggestions
# ============================================================

class TestBuildSkipSuggestions:
    def test_all_core_content_no_skips(self):
        segs = [_make_segment(i, i * 5.0, (i + 1) * 5.0, label="core_content") for i in range(3)]
        assert build_skip_suggestions(segs) == []

    def test_only_non_content_ids_returned(self):
        segs = [
            _make_segment(0, 0.0, 5.0, label="intro"),
            _make_segment(1, 5.0, 15.0, label="core_content"),
            _make_segment(2, 15.0, 18.0, label="sponsorship"),
            _make_segment(3, 18.0, 25.0, label="core_content"),
            _make_segment(4, 25.0, 30.0, label="outro"),
        ]
        assert build_skip_suggestions(segs) == [0, 2, 4]

    def test_preserves_input_order(self):
        # IDs that aren't 0..N-1 are still reported in their list order.
        segs = [
            _make_segment(42, 0.0, 5.0, label="filler"),
            _make_segment(7, 5.0, 10.0, label="dead_air"),
        ]
        assert build_skip_suggestions(segs) == [42, 7]


# ============================================================
# Sub-task 1: schema validation
# ============================================================

class TestSchemaValidation:
    def test_valid_inputs_produce_valid_metadata(self):
        meta = build_metadata(
            _meta_raw(),
            [_make_segment(0, 0.0, 10.0, label="core_content")],
            [],
            None,
        )
        # If any field were wrong this would have raised.
        assert isinstance(meta, Metadata)

    def test_invalid_label_rejected(self):
        # Bypass _make_segment's Literal type by going through dict construction
        with pytest.raises(ValidationError):
            Segment(
                segment_id=0,
                start_sec=0.0,
                end_sec=5.0,
                label="not_a_real_label",  # type: ignore[arg-type]
                confidence=0.5,
                evidence=SegmentEvidence(),
            )

    def test_negative_start_sec_rejected(self):
        with pytest.raises(ValidationError):
            Segment(
                segment_id=0,
                start_sec=-1.0,
                end_sec=5.0,
                label="core_content",
                confidence=0.5,
                evidence=SegmentEvidence(),
            )

    def test_confidence_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            Segment(
                segment_id=0,
                start_sec=0.0,
                end_sec=5.0,
                label="core_content",
                confidence=1.5,  # > 1.0
                evidence=SegmentEvidence(),
            )

    def test_extra_fields_in_metadata_rejected(self):
        # Metadata has extra="forbid"
        with pytest.raises(ValidationError):
            Metadata(
                video_info=build_metadata(_meta_raw(), [], [], None).video_info,
                segments=[],
                chapters=[],
                skip_suggestions=[],
                bogus_field="nope",  # type: ignore[call-arg]
            )


# ============================================================
# Sub-task 5: version stamping
# ============================================================

class TestVersionStamping:
    def test_default_version_applied(self):
        meta = build_metadata(_meta_raw(), [], [], None)
        assert meta.video_info.analysis_version == ANALYSIS_VERSION

    def test_version_at_least_0_4_0(self):
        # Phase 4 contract: version must reflect the export upgrade.
        major, minor, _patch = ANALYSIS_VERSION.split(".")
        assert (int(major), int(minor)) >= (0, 4)

    def test_version_overridable(self):
        meta = build_metadata(_meta_raw(), [], [], None, analysis_version="9.9.9-test")
        assert meta.video_info.analysis_version == "9.9.9-test"

    def test_verified_by_human_default_false(self):
        meta = build_metadata(_meta_raw(), [], [], None)
        assert meta.video_info.verified_by_human is False


# ============================================================
# End-to-end export_metadata: the public interface
# ============================================================

class TestExportMetadata:
    def test_writes_metadata_json(self, workspace: Workspace):
        segs = [_make_segment(0, 0.0, 10.0, label="core_content")]
        out = export_metadata(workspace, _meta_raw(duration=10.0), segs)
        assert out == workspace.metadata_path
        assert out.exists()

    def test_output_validates_against_full_schema(self, workspace: Workspace):
        segs = [
            _make_segment(0, 0.0, 5.0, label="intro"),
            _make_segment(1, 5.0, 20.0, label="core_content"),
            _make_segment(2, 20.0, 25.0, label="sponsorship"),
            _make_segment(3, 25.0, 60.0, label="core_content"),
            _make_segment(4, 60.0, 65.0, label="outro"),
        ]
        export_metadata(workspace, _meta_raw(duration=65.0), segs)
        data = json.loads(workspace.metadata_path.read_text())
        meta = Metadata.model_validate(data)
        assert len(meta.segments) == 5
        assert len(meta.chapters) == 2  # two core_content blocks
        assert meta.skip_suggestions == [0, 2, 4]

    def test_round_trip_with_transcript_and_embeddings(self, workspace: Workspace):
        # Write phase 1 transcript and phase 2 embeddings into the workspace.
        sentences = [
            _make_transcript_seg(0, 0.0, 2.0, "Welcome to the show."),
            _make_transcript_seg(1, 2.0, 4.0, "Today we discuss compression."),
            _make_transcript_seg(2, 4.0, 6.0, "JPEG uses the discrete cosine transform."),
            _make_transcript_seg(3, 6.0, 8.0, "Thanks for watching, please subscribe."),
        ]
        transcript = Transcript(
            language="en",
            duration_sec=8.0,
            model="base",
            segments=sentences,
            full_text=" ".join(s.text for s in sentences),
        )
        workspace.transcript_path.write_text(transcript.model_dump_json())

        # Fake but plausible 384-dim embeddings.
        rng = np.random.default_rng(42)
        emb = rng.standard_normal((4, 384)).astype(np.float32)
        np.save(str(workspace.text_embeddings_path), emb)

        segs = [
            _make_segment(0, 0.0, 2.0, label="intro", summary="will be replaced"),
            _make_segment(1, 2.0, 6.0, label="core_content", summary="will be replaced"),
            _make_segment(2, 6.0, 8.0, label="outro", summary="will be replaced"),
        ]
        export_metadata(workspace, _meta_raw(duration=8.0), segs)

        meta = Metadata.model_validate(json.loads(workspace.metadata_path.read_text()))
        # Each segment got a real summary from its window, not "will be replaced".
        for s in meta.segments:
            assert s.summary != "will be replaced"
            assert s.summary != ""
        # Chapters cover the core_content range.
        assert len(meta.chapters) == 1
        assert meta.chapters[0].start_sec == 2.0
        assert meta.chapters[0].end_sec == 6.0
        # Skip suggestions are intro + outro.
        assert meta.skip_suggestions == [0, 2]

    def test_missing_embeddings_does_not_crash(self, workspace: Workspace):
        # No embeddings file written — should still produce valid output.
        segs = [_make_segment(0, 0.0, 5.0, label="core_content", summary="x")]
        export_metadata(workspace, _meta_raw(duration=5.0), segs)
        meta = Metadata.model_validate(json.loads(workspace.metadata_path.read_text()))
        assert len(meta.segments) == 1

    def test_missing_transcript_does_not_crash(self, workspace: Workspace):
        segs = [_make_segment(0, 0.0, 5.0, label="core_content")]
        # Workspace has no transcript.json; export must degrade.
        export_metadata(workspace, _meta_raw(duration=5.0), segs)
        meta = Metadata.model_validate(json.loads(workspace.metadata_path.read_text()))
        # Segment summary will be empty since no transcript existed.
        assert meta.segments[0].summary == ""

    def test_overwrites_existing_metadata(self, workspace: Workspace):
        workspace.metadata_path.write_text("not real json")
        segs = [_make_segment(0, 0.0, 5.0, label="core_content")]
        export_metadata(workspace, _meta_raw(duration=5.0), segs)
        # Now valid JSON.
        Metadata.model_validate(json.loads(workspace.metadata_path.read_text()))

    def test_duration_zero_yields_empty_output(self, workspace: Workspace):
        # Edge case: duration_sec=0 (allowed by MetaRaw which uses ge=0.0)
        # No segments → empty chapters and skip_suggestions, no crash.
        out = export_metadata(workspace, _meta_raw(duration=0.0), [])
        meta = Metadata.model_validate(json.loads(out.read_text()))
        assert meta.segments == []
        assert meta.chapters == []
        assert meta.skip_suggestions == []