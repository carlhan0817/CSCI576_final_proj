"""
Tests for backend/pipeline/features/text.py

Happy cases: keyword detection, boundary detection, embedding persistence.
Sad cases: empty transcript, single segment, missing transcript file.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from backend.pipeline.features.text import (
    DEFAULT_SIMILARITY_THRESHOLD,
    _match_keywords,
    extract_text_features,
)
from backend.pipeline.schemas import TextFeatures, Transcript, TranscriptSegment
from backend.pipeline.workspace import Workspace


def _make_transcript(segments: list[dict], tmp_path: Path) -> Workspace:
    """Write a fake transcript.json and return the workspace."""
    ws = Workspace.for_video(Path("test.mp4"), root=tmp_path / "workspace")
    ws.ensure()
    transcript = Transcript(
        language="en",
        duration_sec=float(max(s["end"] for s in segments)) if segments else 0.0,
        model="base",
        segments=[TranscriptSegment(**s) for s in segments],
        full_text=" ".join(s["text"] for s in segments),
    )
    ws.transcript_path.write_text(transcript.model_dump_json(indent=2))
    return ws


# ── Unit tests for _match_keywords ───────────────────────────────────────────

class TestMatchKeywords:
    def test_sponsor_keyword_detected(self):
        matches = _match_keywords("Today's video is brought to you by NordVPN.")
        assert any("sponsor" in m for m in matches)

    def test_intro_keyword_detected(self):
        matches = _match_keywords("Welcome back to the channel, in today's video...")
        assert any("intro" in m for m in matches)

    def test_outro_keyword_detected(self):
        matches = _match_keywords("Thanks for watching, don't forget to subscribe!")
        assert any("outro" in m for m in matches)

    def test_self_promo_keyword_detected(self):
        matches = _match_keywords("Check out my Patreon for more content.")
        assert any("self_promo" in m for m in matches)

    def test_recap_keyword_detected(self):
        matches = _match_keywords("Last time we discussed Fourier transforms.")
        assert any("recap" in m for m in matches)

    def test_no_match_returns_empty_list(self):
        matches = _match_keywords("Today we will study signal processing.")
        assert matches == []

    def test_case_insensitive(self):
        matches = _match_keywords("SPONSORED BY MegaCorp Inc.")
        assert any("sponsor" in m for m in matches)

    def test_at_most_one_match_per_group(self):
        # Even if multiple sponsor keywords appear, only one sponsor entry is emitted.
        text = "This video is sponsored. Brought to you by. Use code SAVE."
        matches = _match_keywords(text)
        sponsor_matches = [m for m in matches if m.startswith("sponsor:")]
        assert len(sponsor_matches) == 1

    def test_multiple_groups_can_match(self):
        text = "Welcome back! Thanks for watching. Last time we covered..."
        matches = _match_keywords(text)
        groups = {m.split(":")[0] for m in matches}
        assert len(groups) >= 2


# ── Integration tests (all call MiniLM — mark slow so they're skippable) ───────

class TestExtractTextFeatures:
    def _default_segments(self):
        return [
            {"id": 0, "start": 0.0, "end": 3.0, "text": "Welcome back to the channel."},
            {"id": 1, "start": 3.0, "end": 6.0, "text": "Today we discuss signal processing."},
            {"id": 2, "start": 6.0, "end": 9.0, "text": "Thanks for watching, like and subscribe!"},
        ]

    @pytest.mark.slow
    def test_output_json_created(self, tmp_path):
        ws = _make_transcript(self._default_segments(), tmp_path)
        out = extract_text_features(ws, device="cpu")
        assert out.exists()

    @pytest.mark.slow
    def test_output_validates_against_schema(self, tmp_path):
        ws = _make_transcript(self._default_segments(), tmp_path)
        extract_text_features(ws, device="cpu")
        data = json.loads(ws.text_features_path.read_text())
        features = TextFeatures.model_validate(data)
        assert len(features.segments) == 3

    @pytest.mark.slow
    def test_embeddings_npy_created(self, tmp_path):
        ws = _make_transcript(self._default_segments(), tmp_path)
        extract_text_features(ws, device="cpu")
        assert ws.text_embeddings_path.exists()

    @pytest.mark.slow
    def test_embeddings_shape_matches_segment_count(self, tmp_path):
        ws = _make_transcript(self._default_segments(), tmp_path)
        extract_text_features(ws, device="cpu")
        emb = np.load(str(ws.text_embeddings_path))
        assert emb.shape == (3, 384)

    @pytest.mark.slow
    def test_last_segment_similarity_to_next_is_none(self, tmp_path):
        ws = _make_transcript(self._default_segments(), tmp_path)
        extract_text_features(ws, device="cpu")
        data = json.loads(ws.text_features_path.read_text())
        features = TextFeatures.model_validate(data)
        assert features.segments[-1].similarity_to_next is None

    @pytest.mark.slow
    def test_intro_keyword_matched_in_first_segment(self, tmp_path):
        ws = _make_transcript(self._default_segments(), tmp_path)
        extract_text_features(ws, device="cpu")
        data = json.loads(ws.text_features_path.read_text())
        features = TextFeatures.model_validate(data)
        first = features.segments[0]
        assert any("intro" in kw for kw in first.matched_keywords)

    @pytest.mark.slow
    def test_outro_keyword_matched_in_last_segment(self, tmp_path):
        ws = _make_transcript(self._default_segments(), tmp_path)
        extract_text_features(ws, device="cpu")
        data = json.loads(ws.text_features_path.read_text())
        features = TextFeatures.model_validate(data)
        last = features.segments[-1]
        assert any("outro" in kw for kw in last.matched_keywords)

    @pytest.mark.slow
    def test_no_keywords_in_neutral_segment(self, tmp_path):
        neutral = [
            {"id": 0, "start": 0.0, "end": 3.0, "text": "The Fourier transform decomposes signals."},
            {"id": 1, "start": 3.0, "end": 6.0, "text": "Convolution in time equals multiplication in frequency."},
        ]
        ws = _make_transcript(neutral, tmp_path)
        extract_text_features(ws, device="cpu")
        data = json.loads(ws.text_features_path.read_text())
        features = TextFeatures.model_validate(data)
        for seg in features.segments:
            assert seg.matched_keywords == []

    @pytest.mark.slow
    def test_semantically_distant_pair_flagged_as_boundary(self, tmp_path):
        """A sponsor sentence after a technical sentence should trigger a boundary."""
        segments = [
            {"id": 0, "start": 0.0, "end": 3.0,
             "text": "The discrete cosine transform is used in JPEG compression."},
            {"id": 1, "start": 3.0, "end": 6.0,
             "text": "This video is brought to you by NordVPN, click the link below."},
        ]
        ws = _make_transcript(segments, tmp_path)
        extract_text_features(ws, device="cpu", threshold=DEFAULT_SIMILARITY_THRESHOLD)
        data = json.loads(ws.text_features_path.read_text())
        features = TextFeatures.model_validate(data)
        sim = features.segments[0].similarity_to_next
        assert sim is not None and sim < DEFAULT_SIMILARITY_THRESHOLD

    @pytest.mark.slow
    def test_semantically_similar_pair_not_flagged(self, tmp_path):
        segments = [
            {"id": 0, "start": 0.0, "end": 3.0,
             "text": "Sampling rate determines the highest frequency we can capture."},
            {"id": 1, "start": 3.0, "end": 6.0,
             "text": "The Nyquist theorem states we need twice the maximum frequency."},
        ]
        ws = _make_transcript(segments, tmp_path)
        extract_text_features(ws, device="cpu", threshold=DEFAULT_SIMILARITY_THRESHOLD)
        data = json.loads(ws.text_features_path.read_text())
        features = TextFeatures.model_validate(data)
        sim = features.segments[0].similarity_to_next
        assert sim is not None and sim > DEFAULT_SIMILARITY_THRESHOLD

    @pytest.mark.slow
    def test_cache_skips_on_second_call(self, tmp_path):
        ws = _make_transcript(self._default_segments(), tmp_path)
        extract_text_features(ws, device="cpu")
        mtime1 = ws.text_features_path.stat().st_mtime
        extract_text_features(ws, device="cpu")
        mtime2 = ws.text_features_path.stat().st_mtime
        assert mtime1 == mtime2

    # ── Sad cases ─────────────────────────────────────────────────────────────

    def test_empty_transcript_produces_empty_features(self, tmp_path):
        ws = _make_transcript([], tmp_path)
        out = extract_text_features(ws, device="cpu")
        data = json.loads(out.read_text())
        features = TextFeatures.model_validate(data)
        assert features.segments == []

    def test_empty_transcript_produces_empty_embeddings(self, tmp_path):
        ws = _make_transcript([], tmp_path)
        extract_text_features(ws, device="cpu")
        emb = np.load(str(ws.text_embeddings_path))
        assert emb.shape == (0, 384)

    def test_single_segment_similarity_is_none(self, tmp_path):
        ws = _make_transcript(
            [{"id": 0, "start": 0.0, "end": 5.0, "text": "Only one sentence here."}],
            tmp_path,
        )
        extract_text_features(ws, device="cpu")
        data = json.loads(ws.text_features_path.read_text())
        features = TextFeatures.model_validate(data)
        assert features.segments[0].similarity_to_next is None
        assert features.segments[0].is_potential_boundary is False

    def test_missing_transcript_raises_file_not_found(self, tmp_path):
        ws = Workspace.for_video(Path("no_transcript.mp4"), root=tmp_path / "ws")
        ws.ensure()
        # transcript_path does not exist
        with pytest.raises((FileNotFoundError, OSError)):
            extract_text_features(ws, device="cpu")
