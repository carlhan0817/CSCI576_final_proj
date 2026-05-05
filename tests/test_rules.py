"""
Tests for backend/pipeline/fusion/rules.py (Phase 3: hard rule triggers).
"""
from __future__ import annotations
import numpy as np
import pytest

from backend.pipeline.schemas import TextFeatures, TextFeatureSegment
from backend.pipeline.fusion.rules import (
    RuleHit,
    DEAD_AIR_MIN_DURATION,
    DEAD_AIR_RMS_THRESHOLD,
    HOLDING_SCREEN_MIN_DURATION,
    INTRO_WINDOW_SEC,
    OUTRO_WINDOW_SEC,
    _find_runs,
    rule_dead_air,
    rule_holding_screen,
    rule_sponsor_keyword,
    rule_self_promo_keyword,
    rule_recap_keyword,
    rule_intro_window,
    rule_outro_window,
    run_all_rules,
)


def _grid(T: int, **arrays) -> dict:
    """Build a per-second grid of length T with sensible defaults."""
    g = {
        "is_speech": np.zeros(T, dtype=np.int8),
        "rms_energy": np.zeros(T, dtype=np.float32),
        "hist_diff": np.zeros(T, dtype=np.float32),
        "is_hard_cut": np.zeros(T, dtype=np.int8),
        "is_music": np.zeros(T, dtype=np.int8),
    }
    g.update(arrays)
    return g


def _text_features(segments: list[dict]) -> TextFeatures:
    return TextFeatures(segments=[TextFeatureSegment(**s) for s in segments])


# ---------------------------------------------------------------------------
# _find_runs
# ---------------------------------------------------------------------------

class TestFindRuns:
    def test_finds_simple_run(self):
        arr = np.array([0, 1, 1, 1, 0, 1, 1])
        runs = _find_runs(arr, min_length=2)
        # First run [1..4) length 3, second run [5..7) length 2
        assert (1, 4) in runs
        assert (5, 7) in runs

    def test_min_length_filters_short_runs(self):
        arr = np.array([1, 0, 1, 1, 1, 0])
        # min_length=3 → first run of length 1 dropped, second of length 3 kept
        runs = _find_runs(arr, min_length=3)
        assert runs == [(2, 5)]

    def test_no_runs_returns_empty(self):
        runs = _find_runs(np.zeros(10, dtype=np.int8), min_length=1)
        assert runs == []


# ---------------------------------------------------------------------------
# rule_dead_air
# ---------------------------------------------------------------------------

class TestDeadAir:
    def test_long_silence_with_low_rms_triggers(self):
        T = 20
        is_speech = np.zeros(T, dtype=np.int8)
        rms = np.zeros(T, dtype=np.float32)  # all very low
        # Make some speech in middle so it's not the entire video
        is_speech[10:13] = 1
        rms[10:13] = 0.5
        grid = _grid(T, is_speech=is_speech, rms_energy=rms)
        hits = rule_dead_air(grid)
        assert len(hits) >= 1
        assert hits[0].label == "dead_air"

    def test_loud_music_does_not_trigger_dead_air(self):
        # No speech but high RMS → music, not dead air.
        T = 20
        is_speech = np.zeros(T, dtype=np.int8)
        rms = np.full(T, 0.5, dtype=np.float32)  # well above threshold
        grid = _grid(T, is_speech=is_speech, rms_energy=rms)
        hits = rule_dead_air(grid)
        assert hits == []

    def test_short_silence_below_min_duration_ignored(self):
        T = 20
        is_speech = np.ones(T, dtype=np.int8)
        rms = np.full(T, 0.5, dtype=np.float32)
        # Short pause (< DEAD_AIR_MIN_DURATION seconds) should not fire
        is_speech[5:5 + DEAD_AIR_MIN_DURATION - 1] = 0
        rms[5:5 + DEAD_AIR_MIN_DURATION - 1] = 0.0
        grid = _grid(T, is_speech=is_speech, rms_energy=rms)
        assert rule_dead_air(grid) == []


# ---------------------------------------------------------------------------
# rule_holding_screen
# ---------------------------------------------------------------------------

class TestHoldingScreen:
    def test_long_static_with_no_speech_triggers(self):
        T = 30
        # Static region (low hist_diff) with no speech at the start
        hist = np.full(T, 0.5, dtype=np.float32)
        hist[:HOLDING_SCREEN_MIN_DURATION + 2] = 0.01  # static
        is_speech = np.ones(T, dtype=np.int8)
        is_speech[:HOLDING_SCREEN_MIN_DURATION + 2] = 0
        grid = _grid(T, hist_diff=hist, is_speech=is_speech)
        hits = rule_holding_screen(grid)
        assert any(h.label == "holding_screen" for h in hits)

    def test_static_with_speech_does_not_trigger(self):
        # Person on a static background talking — not a holding screen
        T = 30
        hist = np.full(T, 0.01, dtype=np.float32)
        is_speech = np.ones(T, dtype=np.int8)
        grid = _grid(T, hist_diff=hist, is_speech=is_speech)
        assert rule_holding_screen(grid) == []


# ---------------------------------------------------------------------------
# Keyword-based rules
# ---------------------------------------------------------------------------

class TestSponsorKeyword:
    def test_sponsor_match_creates_window(self):
        text = _text_features([
            {"id": 0, "start": 30.0, "end": 32.0, "text": "Today's video is brought to you by NordVPN.", "matched_keywords": ["sponsor:brought to you by"]},
        ])
        hits = rule_sponsor_keyword(text, T=120)
        assert len(hits) == 1
        assert hits[0].label == "sponsorship"
        # Pads ±15s around the matched sentence
        assert hits[0].start_sec == 15
        assert hits[0].end_sec == 47

    def test_no_sponsor_keyword_no_hit(self):
        text = _text_features([
            {"id": 0, "start": 0.0, "end": 5.0, "text": "Welcome.", "matched_keywords": ["intro:welcome"]},
        ])
        assert rule_sponsor_keyword(text, T=60) == []

    def test_window_clipped_to_grid_bounds(self):
        # Sponsor keyword right at t=0 — start should be clipped to 0.
        text = _text_features([
            {"id": 0, "start": 0.0, "end": 2.0, "text": "Sponsored by Acme.", "matched_keywords": ["sponsor:sponsored by"]},
        ])
        hits = rule_sponsor_keyword(text, T=10)
        assert hits[0].start_sec == 0  # not negative


class TestSelfPromoAndRecap:
    def test_self_promo_keyword(self):
        text = _text_features([
            {"id": 0, "start": 50.0, "end": 52.0, "text": "Subscribe to my channel!", "matched_keywords": ["self_promo:subscribe to my channel"]},
        ])
        hits = rule_self_promo_keyword(text, T=120)
        assert len(hits) == 1
        assert hits[0].label == "self_promotion"

    def test_recap_keyword(self):
        text = _text_features([
            {"id": 0, "start": 5.0, "end": 8.0, "text": "Last time we saw...", "matched_keywords": ["recap:last time"]},
        ])
        hits = rule_recap_keyword(text, T=120)
        assert len(hits) == 1
        assert hits[0].label == "recap"


class TestIntroOutroWindows:
    def test_intro_keyword_within_window_fires(self):
        text = _text_features([
            {"id": 0, "start": 5.0, "end": 8.0, "text": "Welcome back!", "matched_keywords": ["intro:welcome back"]},
        ])
        grid = _grid(T=600)  # 10 minutes
        hits = rule_intro_window(text, grid)
        assert len(hits) == 1
        assert hits[0].label == "intro"
        assert hits[0].start_sec == 0  # intro always starts at 0

    def test_intro_keyword_outside_window_ignored(self):
        text = _text_features([
            {"id": 0, "start": INTRO_WINDOW_SEC + 50.0, "end": INTRO_WINDOW_SEC + 53.0, "text": "Welcome back to our follow-up section.", "matched_keywords": ["intro:welcome back"]},
        ])
        grid = _grid(T=600)
        assert rule_intro_window(text, grid) == []

    def test_outro_keyword_in_last_window_fires(self):
        T = 600
        text = _text_features([
            {"id": 0, "start": T - 30.0, "end": T - 27.0, "text": "Thanks for watching!", "matched_keywords": ["outro:thanks for watching"]},
        ])
        grid = _grid(T=T)
        hits = rule_outro_window(text, grid)
        assert len(hits) == 1
        assert hits[0].label == "outro"
        assert hits[0].end_sec == T

    def test_outro_keyword_too_early_ignored(self):
        T = 600
        text = _text_features([
            {"id": 0, "start": 50.0, "end": 53.0, "text": "Thanks for watching the intro example", "matched_keywords": ["outro:thanks for watching"]},
        ])
        grid = _grid(T=T)
        assert rule_outro_window(text, grid) == []


# ---------------------------------------------------------------------------
# run_all_rules — composition
# ---------------------------------------------------------------------------

class TestRunAllRules:
    def test_fires_multiple_rule_types(self):
        # Build a video with both sponsor + dead air present
        T = 60
        is_speech = np.ones(T, dtype=np.int8)
        rms = np.full(T, 0.5, dtype=np.float32)
        # Long silence at end
        is_speech[40:] = 0
        rms[40:] = 0.0
        grid = _grid(T, is_speech=is_speech, rms_energy=rms)

        text = _text_features([
            {"id": 0, "start": 10.0, "end": 12.0, "text": "Sponsored by Acme.", "matched_keywords": ["sponsor:sponsored by"]},
        ])

        hits = run_all_rules(grid, text)
        labels = {h.label for h in hits}
        assert "sponsorship" in labels
        assert "dead_air" in labels

    def test_no_keywords_no_keyword_hits(self):
        """With zero text features, none of the keyword-based rules fire.
        (dead_air may still fire from the silent grid — that's the rule's job.)"""
        grid = _grid(T=10)
        text = _text_features([])
        hits = run_all_rules(grid, text)
        keyword_labels = {"sponsorship", "self_promotion", "recap", "intro", "outro"}
        for h in hits:
            assert h.label not in keyword_labels


class TestRuleAdBlock:
    """rule_ad_block requires (a) sustained high style_drift, (b) at least one
    OCR commercial pattern hit, (c) bounding hard-cuts."""

    def _build_grid(self, T, drift_high, ocr_hit_t, hard_cuts):
        g = _grid(T)
        g["style_drift"] = np.zeros(T, dtype=np.float32)
        for s, e in drift_high:
            g["style_drift"][s:e] = 0.6
        g["has_url"] = np.zeros(T, dtype=np.int8)
        g["has_price"] = np.zeros(T, dtype=np.int8)
        g["has_phone"] = np.zeros(T, dtype=np.int8)
        g["has_cta"] = np.zeros(T, dtype=np.int8)
        g["has_brand_lockup"] = np.zeros(T, dtype=np.int8)
        if ocr_hit_t is not None:
            g["has_url"][ocr_hit_t] = 1
        for c in hard_cuts:
            g["is_hard_cut"][c] = 1
        return g

    def test_drift_with_ocr_and_cut_fires(self):
        from backend.pipeline.fusion.rules import rule_ad_block
        # 30-second drift block from t=50 to t=80, OCR url at t=60, cuts at 49 and 81
        g = self._build_grid(T=200, drift_high=[(50, 80)], ocr_hit_t=60, hard_cuts=[49, 81])
        hits = rule_ad_block(g)
        assert len(hits) == 1
        h = hits[0]
        assert h.label == "sponsorship"
        assert h.rule_name == "ad_block"
        assert 49 <= h.start_sec <= 51 and 79 <= h.end_sec <= 82
        assert h.confidence >= 0.85

    def test_drift_without_ocr_does_not_fire(self):
        from backend.pipeline.fusion.rules import rule_ad_block
        g = self._build_grid(T=200, drift_high=[(50, 80)], ocr_hit_t=None, hard_cuts=[49, 81])
        hits = rule_ad_block(g)
        assert hits == []

    def test_short_drift_below_min_duration_does_not_fire(self):
        from backend.pipeline.fusion.rules import rule_ad_block
        # 5-second drift is too short
        g = self._build_grid(T=200, drift_high=[(50, 55)], ocr_hit_t=52, hard_cuts=[49, 56])
        hits = rule_ad_block(g)
        assert hits == []

    def test_ocr_outside_drift_block_does_not_count(self):
        from backend.pipeline.fusion.rules import rule_ad_block
        # OCR url 30s away from the drift block — the rule should not pair them.
        g = self._build_grid(T=200, drift_high=[(50, 80)], ocr_hit_t=20, hard_cuts=[49, 81])
        hits = rule_ad_block(g)
        assert hits == []

    def test_no_hard_cut_boundary_still_fires_at_lower_confidence(self):
        from backend.pipeline.fusion.rules import rule_ad_block
        # Drift + OCR but no cut ⇒ still fire (drift+OCR is enough), but confidence lower.
        g = self._build_grid(T=200, drift_high=[(50, 80)], ocr_hit_t=60, hard_cuts=[])
        hits = rule_ad_block(g)
        assert len(hits) == 1
        assert hits[0].confidence < 0.85
        assert hits[0].confidence >= 0.6