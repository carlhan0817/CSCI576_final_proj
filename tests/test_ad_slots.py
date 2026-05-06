"""Tests for backend.pipeline.fusion.ad_slots.score_candidates."""
from __future__ import annotations
import numpy as np
import pytest

from backend.pipeline.schemas import Segment, SegmentEvidence
from backend.pipeline.fusion.ad_slots import (
    score_candidates,
    MIN_BLOCK_SEC,
    MIN_AD_SPACING_SEC,
    SPONSOR_EXCLUSION_SEC,
    W_SIGNAL,
    W_TRANSITION,
    BURST_DENSITY_THRESHOLD,
    LUM_JUMP_THRESHOLD,
    _NON_CHAPTER_LABELS,
    _signal_strength,
    _transition_bonus,
    _sponsor_penalty,
    _find_burst_starts,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _seg(sid, start, end, label="core_content", confidence=0.8):
    return Segment(
        segment_id=sid,
        start_sec=float(start),
        end_sec=float(end),
        label=label,
        confidence=confidence,
        evidence=SegmentEvidence(
            visual_score=0.5, audio_score=0.5, text_score=0.5, triggered_rules=[]
        ),
        summary="",
        user_corrected=False,
    )


def _grid(T, hard_cut_at=None, speech_on=None, hist_diff_at=None):
    g = {
        "is_speech":   np.zeros(T, dtype=np.int8),
        "is_hard_cut": np.zeros(T, dtype=np.uint8),
        "hist_diff":   np.zeros(T, dtype=np.float32),
        "rms_energy":  np.zeros(T, dtype=np.float32),
    }
    if hard_cut_at is not None:
        g["is_hard_cut"][hard_cut_at] = 1
    if speech_on is not None:
        for t in speech_on:
            g["is_speech"][t] = 1
    if hist_diff_at is not None:
        for t, v in hist_diff_at.items():
            g["hist_diff"][t] = v
    return g


# ── Candidate pool ─────────────────────────────────────────────────────────────

class TestCandidatePool:
    def test_single_segment_falls_back_to_natural_break_candidates(self):
        # Single segment has no internal boundaries → fall back to raw candidates.
        segs = [_seg(0, 0, 1000)]
        result = score_candidates(segs, [500.0], None, 1000.0, n_ads=1)
        assert len(result) == 1
        assert result[0].timestamp_sec == 500.0

    def test_multi_segment_uses_segment_starts(self):
        # Two segments: only t=500 is a segment start (excluding 0).
        segs = [_seg(0, 0, 500, "core_content"), _seg(1, 500, 1000, "filler")]
        # Pass a raw candidate far from the segment boundary — it should NOT be used.
        result = score_candidates(segs, [300.0], None, 1000.0, n_ads=1)
        assert len(result) == 1
        assert result[0].timestamp_sec == 500.0  # segment boundary wins

    def test_empty_candidates_and_no_segments_returns_empty(self):
        result = score_candidates([], [], None, 1000.0, n_ads=1)
        assert result == []

    def test_n_ads_zero_returns_empty(self):
        segs = [_seg(0, 0, 500, "core_content"), _seg(1, 500, 1000, "filler")]
        result = score_candidates(segs, [], None, 1000.0, n_ads=0)
        assert result == []


# ── Min-block filter ───────────────────────────────────────────────────────────

class TestMinBlockFilter:
    def test_boundary_too_close_to_start_excluded(self):
        # Segment boundary at t=50: left_block=50 < MIN_BLOCK_SEC=90.
        segs = [_seg(0, 0, 50, "core_content"), _seg(1, 50, 1000, "filler")]
        result = score_candidates(segs, [], None, 1000.0, n_ads=1)
        assert result == []

    def test_boundary_too_close_to_end_excluded(self):
        # Segment boundary at t=950: right_block=50 < MIN_BLOCK_SEC=90.
        segs = [_seg(0, 0, 950, "core_content"), _seg(1, 950, 1000, "filler")]
        result = score_candidates(segs, [], None, 1000.0, n_ads=1)
        assert result == []

    def test_boundary_exactly_at_min_block_included(self):
        segs = [_seg(0, 0, MIN_BLOCK_SEC, "core_content"),
                _seg(1, MIN_BLOCK_SEC, 1000.0, "filler")]
        result = score_candidates(segs, [], None, 1000.0, n_ads=1)
        assert len(result) == 1


# ── Spacing constraint ─────────────────────────────────────────────────────────

class TestSpacingConstraint:
    def test_two_close_boundaries_only_one_selected(self):
        # Segment boundaries at 400 and 500 are 100s apart, under MIN_AD_SPACING_SEC=300.
        segs = [
            _seg(0, 0,   400, "core_content"),
            _seg(1, 400, 500, "filler"),
            _seg(2, 500, 1000, "core_content"),
        ]
        result = score_candidates(segs, [], None, 1000.0, n_ads=2)
        assert len(result) == 1

    def test_well_spaced_boundaries_both_selected(self):
        # Boundaries at 300 and 700: 400s apart > MIN_AD_SPACING_SEC=300.
        segs = [
            _seg(0, 0,   300, "core_content"),
            _seg(1, 300, 700, "filler"),
            _seg(2, 700, 1000, "core_content"),
        ]
        result = score_candidates(segs, [], None, 1000.0, n_ads=2)
        assert len(result) == 2
        timestamps = sorted(c.timestamp_sec for c in result)
        assert timestamps == [300.0, 700.0]


# ── Transition bonus ──────────────────────────────────────────────────────────

class TestTransitionBonus:
    def test_label_change_gives_bonus_1(self):
        segs = [_seg(0, 0, 500, "core_content"), _seg(1, 500, 1000, "recap")]
        bonus = _transition_bonus(500.0, segs)
        assert bonus == 1.0

    def test_same_label_gives_bonus_0(self):
        # Fix 1 preserves hard-cut boundaries between same-label segments.
        segs = [_seg(0, 0, 500, "core_content"), _seg(1, 500, 1000, "core_content")]
        bonus = _transition_bonus(500.0, segs)
        assert bonus == 0.0

    def test_non_chapter_label_on_right_gives_bonus_0(self):
        # Transitions INTO non-chapter labels (ad, filler, dead_air, transition) return 0.
        for label in _NON_CHAPTER_LABELS:
            segs = [_seg(0, 0, 500, "core_content"), _seg(1, 500, 1000, label)]
            assert _transition_bonus(500.0, segs) == 0.0, f"expected 0 for core_content→{label}"

    def test_non_chapter_label_on_left_gives_bonus_0(self):
        # Transitions OUT OF non-chapter labels also return 0.
        for label in _NON_CHAPTER_LABELS:
            segs = [_seg(0, 0, 500, label), _seg(1, 500, 1000, "core_content")]
            assert _transition_bonus(500.0, segs) == 0.0, f"expected 0 for {label}→core_content"

    def test_label_change_beats_same_label_with_hard_cut(self):
        # t=500: core_content → core_content (same label, hard cut) → bonus=0, signal=1.0
        # t=700: core_content → recap (label change) → bonus=1.0, signal=0.2
        # With W_TRANSITION=0.5, t=700 scores 0.4*0.2+0.5*1.0=0.58 vs t=500 scores 0.4*1.0+0=0.4
        T = 1000
        g = _grid(T, hard_cut_at=500)
        segs = [
            _seg(0, 0,   500, "core_content"),
            _seg(1, 500, 700, "core_content"),
            _seg(2, 700, 1000, "recap"),
        ]
        result = score_candidates(segs, [], g, 1000.0, n_ads=1)
        assert result[0].timestamp_sec == 700.0

    def test_transition_score_field_populated(self):
        segs = [_seg(0, 0, 500, "core_content"), _seg(1, 500, 1000, "recap")]
        result = score_candidates(segs, [], None, 1000.0, n_ads=1)
        assert len(result) == 1
        assert result[0].transition_score == 1.0


# ── Signal scoring ─────────────────────────────────────────────────────────────

class TestSignalScoring:
    def test_hard_cut_scores_1(self):
        T = 1000
        g = _grid(T, hard_cut_at=500)
        assert _signal_strength(500.0, g) == 1.0

    def test_speech_to_silence_scores_0_7(self):
        T = 1000
        g = _grid(T, speech_on=list(range(499)))
        assert _signal_strength(499.0, g) == 0.7

    def test_silence_to_speech_scores_0_5(self):
        T = 1000
        g = _grid(T, speech_on=list(range(500, 600)))
        assert _signal_strength(500.0, g) == 0.5

    def test_scene_change_scores_0_4(self):
        T = 1000
        g = _grid(T, hist_diff_at={400: 0.5})
        assert _signal_strength(400.0, g) == 0.4

    def test_hard_cut_boundary_beats_speech_flip_boundary(self):
        # t=400: hard cut + label change (core_content→recap) → sig=1.0, trans=1.0 → score=0.90
        # t=699: silence candidate (speech→silence)           → sig=0.7, trans=0.5  → score=0.53
        # t=700: segment boundary (recap→core_content)        → sig=0.2, trans=1.0  → score=0.58
        # Use "recap" (not "filler") so the t=400 transition bonus is 1.0, not zeroed by non-chapter penalty.
        T = 1000
        g = _grid(T, hard_cut_at=400, speech_on=list(range(699)))
        segs = [
            _seg(0, 0,   400, "core_content"),
            _seg(1, 400, 700, "recap"),
            _seg(2, 700, 1000, "core_content"),
        ]
        result = score_candidates(segs, [], g, 1000.0, n_ads=1)
        # t=400 scores 0.4*1.0+0.5*1.0=0.90, beats silence candidate at 699 (0.53)
        assert result[0].timestamp_sec == 400.0


# ── Sponsor penalty ────────────────────────────────────────────────────────────

class TestSponsorPenalty:
    def test_no_ad_segments_means_zero_penalty(self):
        segs = [_seg(0, 0, 1000, "core_content")]
        assert _sponsor_penalty(500.0, segs) == 0.0

    def test_candidate_close_to_ad_penalized(self):
        segs = [
            _seg(0, 0, 490, "core_content"),
            _seg(1, 490, 510, "ad"),
            _seg(2, 510, 1000, "core_content"),
        ]
        penalty = _sponsor_penalty(520.0, segs)  # 10s from ad end
        assert penalty > 0.0

    def test_candidate_far_from_ad_not_penalized(self):
        segs = [
            _seg(0, 0, 490, "core_content"),
            _seg(1, 490, 510, "ad"),
            _seg(2, 510, 1000, "core_content"),
        ]
        penalty = _sponsor_penalty(200.0, segs)  # 290s from nearest ad boundary
        assert penalty == 0.0


# ── Result properties ──────────────────────────────────────────────────────────

class TestResultProperties:
    def test_n_ads_respected(self):
        # Many well-spaced segment boundaries; request exactly 2.
        segs = [_seg(i, i * 200, (i + 1) * 200, "core_content" if i % 2 == 0 else "filler")
                for i in range(10)]
        result = score_candidates(segs, [], None, 2000.0, n_ads=2)
        assert len(result) == 2

    def test_result_sorted_by_timestamp(self):
        segs = [
            _seg(0, 0,    300, "core_content"),
            _seg(1, 300,  700, "filler"),
            _seg(2, 700, 1000, "core_content"),
        ]
        result = score_candidates(segs, [], None, 1000.0, n_ads=2)
        timestamps = [c.timestamp_sec for c in result]
        assert timestamps == sorted(timestamps)

    def test_output_fields_in_range(self):
        segs = [_seg(0, 0, 500, "core_content"), _seg(1, 500, 1000, "filler")]
        result = score_candidates(segs, [], None, 1000.0, n_ads=1)
        c = result[0]
        assert 0.0 <= c.score <= 1.0
        assert 0.0 <= c.signal_strength <= 1.0
        assert 0.0 <= c.transition_score <= 1.0
        assert c.left_block_sec > 0.0
        assert c.right_block_sec > 0.0


# ── Burst detection ────────────────────────────────────────────────────────────

def _burst_grid(T, burst_cuts=None, lum_at=None):
    """Build a grid with hc_burst_density and lum_context_delta columns."""
    g = _grid(T)
    if burst_cuts:
        for t in burst_cuts:
            g["is_hard_cut"][t] = 1
    # Compute hc_burst_density from is_hard_cut
    hc = g["is_hard_cut"].astype(np.int32)
    cs = np.concatenate([[0], np.cumsum(hc)])
    bd = np.zeros(T, dtype=np.float32)
    for t in range(T):
        bd[t] = (cs[min(t + 60, T)] - cs[t]) / 60
    g["hc_burst_density"] = bd
    g["lum_context_delta"] = np.zeros(T, dtype=np.float32)
    if lum_at:
        for t, v in lum_at.items():
            g["lum_context_delta"][t] = v
    return g


class TestBurstDetection:
    def test_no_burst_returns_empty(self):
        # Single hard cut with low density → no burst
        g = _burst_grid(200, burst_cuts=[100])
        assert _find_burst_starts(g) == []

    def test_burst_start_found_at_first_hard_cut(self):
        # 10 hard cuts in [100, 110) → density = 10/60 ≈ 0.167 ≥ threshold
        g = _burst_grid(300, burst_cuts=list(range(100, 110)))
        bursts = _find_burst_starts(g)
        assert bursts == [100.0]

    def test_two_separate_bursts_both_found(self):
        # Two burst clusters separated by a quiet gap
        g = _burst_grid(500, burst_cuts=list(range(100, 110)) + list(range(400, 410)))
        bursts = _find_burst_starts(g)
        assert 100.0 in bursts
        assert 400.0 in bursts
        assert len(bursts) == 2

    def test_none_grid_returns_empty(self):
        assert _find_burst_starts(None) == []

    def test_grid_without_burst_density_returns_empty(self):
        g = _grid(200, hard_cut_at=100)
        assert _find_burst_starts(g) == []

    def test_burst_candidate_selected_over_segment_boundary(self):
        # Burst of 10 cuts at t=700 (not a segment boundary) vs segment at t=500.
        # Burst should win: sig=1.0, trans=1.0, pen=0 → composite=0.9
        # Segment at 500 (no burst): sig=0.5 (fallback, no burst data), trans=0 (filler), composite=0.2
        T = 1200
        g = _burst_grid(T, burst_cuts=list(range(700, 710)),
                        lum_at={700: 50.0})  # large lum jump at burst start
        segs = [_seg(0, 0, 500, "core_content"), _seg(1, 500, 1200, "core_content")]
        result = score_candidates(segs, [], g, 1200.0, n_ads=1)
        assert result[0].timestamp_sec == 700.0

    def test_hard_cut_with_burst_and_lum_scores_1(self):
        g = _burst_grid(200, burst_cuts=list(range(100, 110)),
                        lum_at={100: LUM_JUMP_THRESHOLD + 5})
        assert _signal_strength(100.0, g) == 1.0

    def test_hard_cut_with_burst_no_lum_scores_0_95(self):
        g = _burst_grid(200, burst_cuts=list(range(100, 110)))
        # burst_density ≥ threshold, lum_delta = 0 → 0.95
        assert _signal_strength(100.0, g) == 0.95

    def test_plain_hard_cut_with_burst_data_scores_0_8(self):
        # Single cut, low density → plain hard cut, score demoted to 0.80
        g = _burst_grid(200, burst_cuts=[100])
        assert _signal_strength(100.0, g) == 0.80

    def test_hard_cut_without_burst_data_scores_1_legacy(self):
        # Grid has no hc_burst_density → legacy behaviour: any hard cut = 1.0
        g = _grid(200, hard_cut_at=100)
        assert _signal_strength(100.0, g) == 1.0
