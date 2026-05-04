"""
Tests for backend/pipeline/fusion/classify.py (Phase 3: per-segment
classification).
"""
from __future__ import annotations
import numpy as np
import pytest

from backend.pipeline.schemas import TranscriptSegment
from backend.pipeline.fusion.rules import RuleHit
from backend.pipeline.fusion.classify import (
    CLIP_LABEL_TO_SEGMENT,
    classify_segments,
    _aggregate_segment_features,
    _resolve_rule_label_for_segment,
    _classify_by_clip_and_audio,
    _extract_segment_summary,
    _interval_overlap_seconds,
)


def _basic_grid(T: int, **arrays) -> dict:
    """Per-second grid with all the columns classify.py expects."""
    g = {
        "is_speech": np.zeros(T, dtype=np.int8),
        "rms_energy": np.zeros(T, dtype=np.float32),
        "hist_diff": np.zeros(T, dtype=np.float32),
        "is_hard_cut": np.zeros(T, dtype=np.int8),
        "is_music": np.zeros(T, dtype=np.int8),
        "has_text": np.zeros(T, dtype=np.int8),
        "text_sim_to_next": np.full(T, np.nan, dtype=np.float32),
        "text_sentence_id": np.full(T, -1, dtype=np.int32),
    }
    g.update(arrays)
    return g


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestIntervalOverlap:
    def test_no_overlap(self):
        assert _interval_overlap_seconds(0, 5, 10, 20) == 0

    def test_full_overlap(self):
        assert _interval_overlap_seconds(0, 10, 0, 10) == 10

    def test_partial_overlap(self):
        assert _interval_overlap_seconds(0, 10, 5, 15) == 5

    def test_one_inside_other(self):
        assert _interval_overlap_seconds(0, 20, 5, 10) == 5

    def test_touching_returns_zero(self):
        assert _interval_overlap_seconds(0, 5, 5, 10) == 0


class TestAggregateSegmentFeatures:
    def test_means_computed_over_window(self):
        T = 10
        grid = _basic_grid(T, is_speech=np.array([1] * 5 + [0] * 5, dtype=np.int8))
        agg = _aggregate_segment_features(grid, 0, 10)
        assert agg["is_speech"] == pytest.approx(0.5)

    def test_text_sentence_id_skipped(self):
        grid = _basic_grid(10)
        agg = _aggregate_segment_features(grid, 0, 10)
        assert "text_sentence_id" not in agg

    def test_nan_aware_text_sim(self):
        T = 10
        sim = np.full(T, np.nan, dtype=np.float32)
        sim[2] = 0.4
        sim[7] = 0.6
        grid = _basic_grid(T, text_sim_to_next=sim)
        agg = _aggregate_segment_features(grid, 0, 10)
        assert agg["text_sim_to_next"] == pytest.approx(0.5)

    def test_empty_window_yields_zeros(self):
        grid = _basic_grid(10)
        agg = _aggregate_segment_features(grid, 5, 5)  # zero-length window
        assert agg["is_speech"] == 0.0


class TestResolveRuleLabel:
    def test_high_coverage_rule_picked(self):
        hits = [RuleHit(start_sec=0, end_sec=10, label="sponsorship", rule_name="sponsor_keyword", confidence=0.85)]
        label, conf, triggered = _resolve_rule_label_for_segment(hits, 2, 8)
        # 100% covered
        assert label == "sponsorship"
        assert conf == 0.85
        assert "sponsor_keyword" in triggered

    def test_low_coverage_rule_ignored(self):
        # Rule covers only 1 of 10 segment seconds → 10% coverage
        hits = [RuleHit(start_sec=0, end_sec=1, label="sponsorship", rule_name="sponsor_keyword", confidence=0.85)]
        label, conf, triggered = _resolve_rule_label_for_segment(hits, 0, 10)
        assert label == ""
        assert conf == 0.0

    def test_highest_confidence_wins_when_overlapping(self):
        hits = [
            RuleHit(0, 10, "filler", "low_conf_rule", confidence=0.3),
            RuleHit(0, 10, "sponsorship", "high_conf_rule", confidence=0.85),
        ]
        label, conf, _ = _resolve_rule_label_for_segment(hits, 0, 10)
        assert label == "sponsorship"
        assert conf == 0.85

    def test_no_rules_returns_empty(self):
        label, conf, triggered = _resolve_rule_label_for_segment([], 0, 10)
        assert label == ""
        assert conf == 0.0
        assert triggered == []


class TestClipLabelMapping:
    """Phase 3 fix #1 ensured all 10 V2.1 CLIP prompts map somewhere.
    These tests pin that contract so a future prompt-list change can't
    silently regress."""

    def test_all_v21_prompts_covered(self):
        # If new prompts are added, this test should fail until the mapping
        # is updated.
        v21_prompts = [
            "presentation slide", "person talking", "screen recording",
            "video game", "animated scene", "blank screen",
            "end credits", "title card", "advertisement", "sponsor logo",
        ]
        for prompt in v21_prompts:
            matched = False
            for substring in CLIP_LABEL_TO_SEGMENT:
                if substring in prompt:
                    matched = True
                    break
            assert matched, f"CLIP prompt '{prompt}' has no SegmentLabel mapping"

    def test_sponsor_logo_maps_to_sponsorship(self):
        assert CLIP_LABEL_TO_SEGMENT["sponsor logo"] == "sponsorship"

    def test_end_credits_maps_to_outro(self):
        assert CLIP_LABEL_TO_SEGMENT["end credits"] == "outro"

    def test_title_card_maps_to_intro(self):
        assert CLIP_LABEL_TO_SEGMENT["title card"] == "intro"


class TestClassifyByClipAndAudio:
    def test_no_clip_columns_defaults_to_core_content(self):
        agg = {"is_speech": 0.5, "has_text": 1.0}
        label, conf, scores = _classify_by_clip_and_audio(agg)
        assert label == "core_content"

    def test_advertisement_clip_gives_sponsorship(self):
        agg = {
            "clip_an advertisement slide": 0.9,
            "clip_a person talking": 0.1,
            "is_speech": 0.5,
            "has_text": 1.0,
        }
        label, conf, scores = _classify_by_clip_and_audio(agg)
        assert label == "sponsorship"
        assert scores["visual"] == pytest.approx(0.9)

    def test_no_speech_demotes_to_filler(self):
        # Strong "person talking" CLIP but no speech → likely filler/B-roll
        agg = {
            "clip_a person talking": 0.9,
            "is_speech": 0.05,
            "has_text": 0.0,
        }
        label, _, _ = _classify_by_clip_and_audio(agg)
        assert label == "filler"


class TestExtractSegmentSummary:
    def test_empty_transcript_returns_empty(self):
        assert _extract_segment_summary([], 0.0, 10.0) == ""

    def test_picks_longest_contained_sentence(self):
        sents = [
            TranscriptSegment(id=0, start=1.0, end=2.0, text="short"),
            TranscriptSegment(id=1, start=2.0, end=3.0, text="this is a much longer one"),
        ]
        summary = _extract_segment_summary(sents, 0.0, 5.0)
        assert "longer" in summary

    def test_truncated_to_200_chars(self):
        long_text = "x" * 500
        sents = [TranscriptSegment(id=0, start=1.0, end=2.0, text=long_text)]
        assert len(_extract_segment_summary(sents, 0.0, 5.0)) == 200

    def test_falls_back_to_overlapping_when_none_contained(self):
        # No sentence is fully inside [5,10) but one overlaps
        sents = [TranscriptSegment(id=0, start=4.0, end=7.0, text="overlap-only")]
        assert "overlap" in _extract_segment_summary(sents, 5.0, 10.0)


# ---------------------------------------------------------------------------
# classify_segments — the public function
# ---------------------------------------------------------------------------

class TestClassifySegments:
    def test_single_segment_uses_rule_when_covered(self):
        T = 10
        grid = _basic_grid(T)
        boundaries = [0, T]
        rule_hits = [RuleHit(0, T, "intro", "intro_window", confidence=0.7)]
        segs = classify_segments(grid, boundaries, rule_hits, [])
        assert len(segs) == 1
        assert segs[0].label == "intro"
        assert "intro_window" in segs[0].evidence.triggered_rules

    def test_segment_falls_through_to_clip_when_no_rule(self):
        T = 10
        grid = _basic_grid(
            T,
            **{"clip_an end credits screen": np.full(T, 0.9, dtype=np.float32)},
        )
        boundaries = [0, T]
        segs = classify_segments(grid, boundaries, [], [])
        assert segs[0].label == "outro"

    def test_segment_id_is_index(self):
        T = 30
        grid = _basic_grid(T)
        boundaries = [0, 10, 20, 30]
        segs = classify_segments(grid, boundaries, [], [])
        assert [s.segment_id for s in segs] == [0, 1, 2]

    def test_zero_length_segment_skipped(self):
        T = 10
        grid = _basic_grid(T)
        # Two boundaries at the same position should produce no segment between
        # them.
        boundaries = [0, 5, 5, 10]
        segs = classify_segments(grid, boundaries, [], [])
        # 3 valid intervals: [0,5), [5,10) — the middle [5,5) is skipped
        assert all(s.end_sec > s.start_sec for s in segs)

    def test_evidence_scores_populated(self):
        T = 10
        grid = _basic_grid(
            T,
            is_speech=np.ones(T, dtype=np.int8),
            **{"clip_a person talking": np.full(T, 0.7, dtype=np.float32)},
        )
        segs = classify_segments(grid, [0, T], [], [])
        evidence = segs[0].evidence
        # All three modality scores should be set (not all zero).
        assert evidence.visual_score + evidence.audio_score > 0

    def test_summary_attached_when_transcript_provided(self):
        T = 10
        grid = _basic_grid(T)
        transcript = [TranscriptSegment(id=0, start=2.0, end=5.0, text="hello world")]
        segs = classify_segments(grid, [0, T], [], transcript)
        assert "hello" in segs[0].summary

    def test_confidence_in_unit_interval(self):
        # confidence must be valid for downstream Pydantic validation
        T = 10
        grid = _basic_grid(
            T,
            is_speech=np.ones(T, dtype=np.int8),
            **{"clip_a person talking": np.full(T, 0.9, dtype=np.float32)},
        )
        segs = classify_segments(grid, [0, T], [], [])
        for s in segs:
            assert 0.0 <= s.confidence <= 1.0