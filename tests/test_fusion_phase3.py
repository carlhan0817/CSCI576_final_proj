"""Phase 3 fusion pipeline tests — covers all 4 fixes from phase3-fixes.md."""
from __future__ import annotations

import numpy as np
import pytest

from backend.pipeline.fusion.boundaries import find_boundaries
from backend.pipeline.fusion.classify import (
    CLIP_LABEL_TO_SEGMENT,
    _classify_by_clip_and_audio,
)
from backend.pipeline.fusion.rules import (
    DEAD_AIR_RMS_THRESHOLD,
    rule_dead_air,
    rule_intro_window,
    rule_outro_window,
    rule_recap_keyword,
    rule_self_promo_keyword,
    rule_sponsor_keyword,
)
from backend.pipeline.schemas import TextFeatures, TextFeatureSegment

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_grid(T: int, **overrides) -> dict[str, np.ndarray]:
    """Minimal valid grid of length T; override any column via kwargs."""
    grid: dict[str, np.ndarray] = {
        "is_speech":    np.zeros(T, dtype=np.int8),
        "is_hard_cut":  np.zeros(T, dtype=np.int8),
        "hist_diff":    np.zeros(T, dtype=np.float32),
        "text_sim_to_next": np.full(T, np.nan, dtype=np.float64),
        "has_text":     np.zeros(T, dtype=np.float32),
        "rms_energy":   np.zeros(T, dtype=np.float32),
    }
    grid.update(overrides)
    return grid


def _make_text_features(segments=None) -> TextFeatures:
    return TextFeatures(segments=segments or [])


def _make_text_seg(start: float, end: float, text: str, keywords=None) -> TextFeatureSegment:
    return TextFeatureSegment(
        id=0,
        start=start,
        end=end,
        text=text,
        similarity_to_next=None,
        is_potential_boundary=False,
        matched_keywords=keywords or [],
    )


# ── Fix 1: CLIP label map covers all 10 V2.1 prompts ─────────────────────────

class TestClipLabelMap:
    def test_map_has_10_entries(self):
        assert len(CLIP_LABEL_TO_SEGMENT) == 10

    @pytest.mark.parametrize("substring,expected", [
        ("presentation slide", "core_content"),
        ("person talking",     "core_content"),
        ("screen recording",   "core_content"),
        ("video game",         "core_content"),
        ("animated scene",     "core_content"),
        ("blank screen",       "transition"),
        ("end credits",        "outro"),
        ("title card",         "intro"),
        ("advertisement",      "sponsorship"),
        ("sponsor logo",       "sponsorship"),
    ])
    def test_each_prompt_maps_correctly(self, substring, expected):
        assert CLIP_LABEL_TO_SEGMENT[substring] == expected

    def test_end_credits_not_core_content(self):
        # Previously fell through to core_content default — regression guard.
        T = 10
        _make_grid(T, **{"clip_end credits": np.ones(T, dtype=np.float32)})
        # Force all other clip keys to zero
        label, _, _ = _classify_by_clip_and_audio(
            {"clip_end credits": 0.95, "is_speech": 0.0, "has_text": 0.0}
        )
        assert label == "outro"

    def test_title_card_maps_to_intro(self):
        label, _, _ = _classify_by_clip_and_audio(
            {"clip_title card": 0.9, "is_speech": 0.0, "has_text": 0.0}
        )
        assert label == "intro"

    def test_advertisement_maps_to_sponsorship(self):
        label, _, _ = _classify_by_clip_and_audio(
            {"clip_advertisement": 0.9, "is_speech": 0.0, "has_text": 0.0}
        )
        assert label == "sponsorship"


# ── Fix 2: rule_dead_air requires RMS < threshold AND no speech ───────────────

class TestRuleDeadAir:
    def test_silence_and_low_rms_triggers_dead_air(self):
        T = 20
        grid = _make_grid(T,
            is_speech=np.zeros(T, dtype=np.int8),
            rms_energy=np.full(T, 0.001, dtype=np.float32),
        )
        hits = rule_dead_air(grid)
        assert len(hits) == 1
        assert hits[0].label == "dead_air"

    def test_silence_but_high_rms_does_not_trigger(self):
        # Music: no speech but loud — must NOT become dead_air.
        T = 20
        grid = _make_grid(T,
            is_speech=np.zeros(T, dtype=np.int8),
            rms_energy=np.full(T, 0.3, dtype=np.float32),  # loud music
        )
        hits = rule_dead_air(grid)
        assert hits == []

    def test_speech_present_does_not_trigger(self):
        T = 20
        grid = _make_grid(T,
            is_speech=np.ones(T, dtype=np.int8),
            rms_energy=np.full(T, 0.001, dtype=np.float32),
        )
        hits = rule_dead_air(grid)
        assert hits == []

    def test_missing_rms_column_falls_back_gracefully(self):
        # Old workspace without rms_energy — silence alone should still fire.
        T = 20
        grid = _make_grid(T, is_speech=np.zeros(T, dtype=np.int8))
        del grid["rms_energy"]
        hits = rule_dead_air(grid)
        assert len(hits) == 1

    def test_short_silence_below_min_duration_not_flagged(self):
        T = 10
        grid = _make_grid(T,
            is_speech=np.zeros(T, dtype=np.int8),
            rms_energy=np.zeros(T, dtype=np.float32),
        )
        # Only 4 silent seconds — below DEAD_AIR_MIN_DURATION=5
        grid["is_speech"][0:4] = 0
        grid["is_speech"][4:] = 1
        hits = rule_dead_air(grid)
        assert hits == []

    def test_rms_threshold_boundary(self):
        T = 10
        # Exactly at threshold — should NOT fire (rule uses strict <)
        grid = _make_grid(T,
            is_speech=np.zeros(T, dtype=np.int8),
            rms_energy=np.full(T, DEAD_AIR_RMS_THRESHOLD, dtype=np.float32),
        )
        hits = rule_dead_air(grid)
        assert hits == []


# ── Fix 3: boundaries — Signal 5 adds music-transition candidates ─────────────

class TestBoundariesMusicSignal:
    def _minimal_grid(self, T: int) -> dict[str, np.ndarray]:
        return _make_grid(T)

    def test_music_flip_adds_boundary(self):
        T = 20
        grid = _make_grid(T)
        is_music = np.zeros(T, dtype=np.int8)
        is_music[10:] = 1  # music starts at second 10
        grid["is_music"] = is_music
        boundaries = find_boundaries(grid)
        assert 10 in boundaries

    def test_no_music_column_does_not_crash(self):
        T = 20
        grid = _make_grid(T)
        # is_music not in grid at all
        boundaries = find_boundaries(grid)
        assert boundaries[0] == 0
        assert boundaries[-1] == T

    def test_constant_music_adds_no_extra_boundaries(self):
        T = 20
        grid = _make_grid(T)
        grid["is_music"] = np.ones(T, dtype=np.int8)  # always music, no flip
        baseline = find_boundaries(_make_grid(T))
        with_music = find_boundaries(grid)
        # Same result: no transitions to detect
        assert set(baseline) == set(with_music)

    def test_multiple_music_flips_add_multiple_boundaries(self):
        T = 30
        grid = _make_grid(T)
        is_music = np.zeros(T, dtype=np.int8)
        is_music[5:10] = 1   # music 5→10
        is_music[20:25] = 1  # music 20→25
        grid["is_music"] = is_music
        boundaries = find_boundaries(grid)
        # 5 and 10 should be there (or within MIN_BOUNDARY_GAP_SEC dedup)
        assert 5 in boundaries
        assert 10 in boundaries


# ── Fix 4: keyword rules read matched_keywords, not raw text ─────────────────

class TestKeywordRulesUseMatchedKeywords:
    def test_sponsor_keyword_fires_on_matched_keywords(self):
        seg = _make_text_seg(30.0, 35.0, "anything", ["sponsor:sponsored by"])
        tf = _make_text_features([seg])
        hits = rule_sponsor_keyword(tf, T=100)
        assert len(hits) == 1
        assert hits[0].label == "sponsorship"

    def test_sponsor_keyword_ignores_raw_text_if_no_matched_keyword(self):
        # Text contains the phrase but matched_keywords is empty.
        seg = _make_text_seg(30.0, 35.0, "sponsored by acme corp", [])
        tf = _make_text_features([seg])
        hits = rule_sponsor_keyword(tf, T=100)
        assert hits == []

    def test_self_promo_keyword_fires_on_matched_keywords(self):
        seg = _make_text_seg(10.0, 15.0, "anything", ["self_promo:patreon"])
        tf = _make_text_features([seg])
        hits = rule_self_promo_keyword(tf, T=100)
        assert len(hits) == 1
        assert hits[0].label == "self_promotion"

    def test_recap_keyword_fires_on_matched_keywords(self):
        seg = _make_text_seg(5.0, 10.0, "anything", ["recap:last time"])
        tf = _make_text_features([seg])
        hits = rule_recap_keyword(tf, T=100)
        assert len(hits) == 1
        assert hits[0].label == "recap"

    def test_intro_window_fires_within_first_90s(self):
        seg = _make_text_seg(5.0, 8.0, "anything", ["intro:welcome back"])
        tf = _make_text_features([seg])
        grid = _make_grid(200)
        hits = rule_intro_window(tf, grid)
        assert len(hits) == 1
        assert hits[0].label == "intro"
        assert hits[0].start_sec == 0

    def test_intro_window_ignores_keyword_after_90s(self):
        seg = _make_text_seg(100.0, 105.0, "anything", ["intro:welcome back"])
        tf = _make_text_features([seg])
        grid = _make_grid(200)
        hits = rule_intro_window(tf, grid)
        assert hits == []

    def test_outro_window_fires_in_last_90s(self):
        T = 300
        seg = _make_text_seg(250.0, 255.0, "anything", ["outro:thanks for watching"])
        tf = _make_text_features([seg])
        grid = _make_grid(T)
        hits = rule_outro_window(tf, grid)
        assert len(hits) == 1
        assert hits[0].label == "outro"
        assert hits[0].end_sec == T

    def test_outro_window_ignores_keyword_before_cutoff(self):
        T = 300
        seg = _make_text_seg(10.0, 15.0, "anything", ["outro:thanks for watching"])
        tf = _make_text_features([seg])
        grid = _make_grid(T)
        hits = rule_outro_window(tf, grid)
        assert hits == []

    def test_wrong_prefix_does_not_cross_fire(self):
        # sponsor prefix should not fire intro rule
        seg = _make_text_seg(5.0, 8.0, "anything", ["sponsor:sponsored by"])
        tf = _make_text_features([seg])
        grid = _make_grid(200)
        hits = rule_intro_window(tf, grid)
        assert hits == []
