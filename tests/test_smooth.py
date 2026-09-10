"""
Tests for backend/pipeline/fusion/smooth.py (Phase 3: temporal smoothing).
"""
from __future__ import annotations

import pytest

from backend.pipeline.fusion.smooth import (
    MIN_SEGMENT_DURATION,
    absorb_short_segments,
    merge_adjacent_same_label,
    renumber,
    smooth_pipeline,
)
from backend.pipeline.schemas import Segment, SegmentEvidence


def _seg(sid: int, start: float, end: float, label: str = "core_content", confidence: float = 0.7, summary: str = "") -> Segment:
    return Segment(
        segment_id=sid,
        start_sec=start,
        end_sec=end,
        label=label,  # type: ignore[arg-type]
        confidence=confidence,
        evidence=SegmentEvidence(
            visual_score=0.5, audio_score=0.5, text_score=0.5,
            triggered_rules=[],
        ),
        summary=summary,
        user_corrected=False,
    )


# ---------------------------------------------------------------------------
# merge_adjacent_same_label
# ---------------------------------------------------------------------------

class TestMergeAdjacent:
    def test_merges_two_adjacent_same_label(self):
        segs = [
            _seg(0, 0.0, 5.0, "core_content"),
            _seg(1, 5.0, 10.0, "core_content"),
        ]
        out = merge_adjacent_same_label(segs)
        assert len(out) == 1
        assert out[0].start_sec == 0.0
        assert out[0].end_sec == 10.0

    def test_does_not_merge_different_labels(self):
        segs = [
            _seg(0, 0.0, 5.0, "core_content"),
            _seg(1, 5.0, 10.0, "sponsorship"),
        ]
        out = merge_adjacent_same_label(segs)
        assert len(out) == 2

    def test_merges_chain_of_three(self):
        segs = [
            _seg(0, 0.0, 5.0, "core_content"),
            _seg(1, 5.0, 10.0, "core_content"),
            _seg(2, 10.0, 15.0, "core_content"),
        ]
        out = merge_adjacent_same_label(segs)
        assert len(out) == 1
        assert out[0].end_sec == 15.0

    def test_empty_list_returns_empty(self):
        assert merge_adjacent_same_label([]) == []

    def test_single_segment_unchanged(self):
        segs = [_seg(0, 0.0, 5.0, "intro")]
        out = merge_adjacent_same_label(segs)
        assert len(out) == 1
        assert out[0].label == "intro"

    def test_evidence_combined_in_merge(self):
        segs = [
            _seg(0, 0.0, 5.0, "core_content", confidence=0.5),
            _seg(1, 5.0, 10.0, "core_content", confidence=0.9),
        ]
        out = merge_adjacent_same_label(segs)
        # Confidence is the max of merged
        assert out[0].confidence == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# absorb_short_segments
# ---------------------------------------------------------------------------

class TestAbsorbShort:
    def test_short_segment_absorbed(self):
        # 1-second segment between two longer ones (below MIN_SEGMENT_DURATION)
        segs = [
            _seg(0, 0.0, 10.0, "core_content", confidence=0.8),
            _seg(1, 10.0, 11.0, "filler", confidence=0.3),
            _seg(2, 11.0, 20.0, "core_content", confidence=0.8),
        ]
        out = absorb_short_segments(segs)
        # The middle filler should disappear
        assert all(s.label != "filler" or (s.end_sec - s.start_sec) >= MIN_SEGMENT_DURATION for s in out)

    def test_short_at_start_merged_into_right(self):
        segs = [
            _seg(0, 0.0, 1.0, "filler"),
            _seg(1, 1.0, 10.0, "core_content"),
        ]
        out = absorb_short_segments(segs)
        # Output preserves the [0, 10) coverage but the short segment is gone
        assert len(out) == 1
        assert out[0].start_sec == 0.0
        assert out[0].end_sec == 10.0
        assert out[0].label == "core_content"

    def test_short_at_end_merged_into_left(self):
        segs = [
            _seg(0, 0.0, 10.0, "core_content"),
            _seg(1, 10.0, 11.0, "filler"),
        ]
        out = absorb_short_segments(segs)
        assert len(out) == 1
        assert out[0].end_sec == 11.0
        assert out[0].label == "core_content"

    def test_long_enough_segment_kept(self):
        # Exactly MIN_SEGMENT_DURATION should be kept
        segs = [
            _seg(0, 0.0, 10.0, "core_content"),
            _seg(1, 10.0, 10.0 + MIN_SEGMENT_DURATION, "filler"),
            _seg(2, 10.0 + MIN_SEGMENT_DURATION, 20.0, "core_content"),
        ]
        out = absorb_short_segments(segs)
        # All three should remain — none are shorter than MIN_SEGMENT_DURATION
        assert len(out) == 3

    def test_single_segment_left_alone_even_if_short(self):
        # Only one segment in the entire video — can't be absorbed
        segs = [_seg(0, 0.0, 0.5, "intro")]
        out = absorb_short_segments(segs)
        assert len(out) == 1

    def test_higher_confidence_neighbor_wins(self):
        segs = [
            _seg(0, 0.0, 10.0, "intro", confidence=0.4),
            _seg(1, 10.0, 11.0, "filler"),
            _seg(2, 11.0, 20.0, "core_content", confidence=0.95),
        ]
        out = absorb_short_segments(segs)
        # The short filler should merge into the higher-confidence right neighbor
        assert len(out) == 2
        # The right segment now spans [10, 20)
        right = next(s for s in out if s.label == "core_content")
        assert right.start_sec == 10.0
        assert right.end_sec == 20.0


# ---------------------------------------------------------------------------
# renumber
# ---------------------------------------------------------------------------

class TestRenumber:
    def test_renumber_starts_from_zero(self):
        segs = [
            _seg(7, 0.0, 5.0),
            _seg(42, 5.0, 10.0, "intro"),
        ]
        out = renumber(segs)
        assert [s.segment_id for s in out] == [0, 1]

    def test_renumber_preserves_order(self):
        segs = [
            _seg(99, 0.0, 5.0, "intro"),
            _seg(0, 5.0, 10.0, "core_content"),
            _seg(50, 10.0, 15.0, "outro"),
        ]
        out = renumber(segs)
        assert [s.label for s in out] == ["intro", "core_content", "outro"]
        assert [s.segment_id for s in out] == [0, 1, 2]


# ---------------------------------------------------------------------------
# smooth_pipeline (full chain)
# ---------------------------------------------------------------------------

class TestSmoothPipeline:
    def test_merges_then_absorbs_then_renumbers(self):
        segs = [
            _seg(0, 0.0, 5.0, "core_content"),
            _seg(1, 5.0, 10.0, "core_content"),  # adjacent same → merge
            _seg(2, 10.0, 11.0, "filler"),       # short → absorb
            _seg(3, 11.0, 20.0, "core_content"),
        ]
        out = smooth_pipeline(segs)
        # Expected: one big core_content from 0 to 20, ID 0
        assert len(out) == 1
        assert out[0].segment_id == 0
        assert out[0].start_sec == 0.0
        assert out[0].end_sec == 20.0
        assert out[0].label == "core_content"

    def test_pipeline_idempotent_on_clean_input(self):
        # Already-smoothed input should pass through unchanged
        segs = [
            _seg(0, 0.0, 10.0, "intro"),
            _seg(1, 10.0, 100.0, "core_content"),
            _seg(2, 100.0, 110.0, "outro"),
        ]
        out = smooth_pipeline(segs)
        assert len(out) == 3
        assert [s.label for s in out] == ["intro", "core_content", "outro"]

    def test_pipeline_handles_empty(self):
        assert smooth_pipeline([]) == []

    def test_no_overlap_or_gap_after_smoothing(self):
        # No two segments should overlap, and (since smooth doesn't insert
        # gaps) every segment should start exactly where the previous ended.
        segs = [
            _seg(0, 0.0, 5.0, "intro"),
            _seg(1, 5.0, 10.0, "core_content"),
            _seg(2, 10.0, 11.0, "filler"),
            _seg(3, 11.0, 20.0, "core_content"),
            _seg(4, 20.0, 25.0, "outro"),
        ]
        out = smooth_pipeline(segs)
        for i in range(len(out) - 1):
            assert out[i].end_sec == out[i + 1].start_sec
        assert out[0].start_sec == 0.0
        assert out[-1].end_sec == 25.0