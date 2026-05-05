"""
Full Phase 3 fusion pipeline tests — align, boundaries (all signals),
classify, rules (holding_screen, run_all_rules), smooth, and export.
Complements test_fusion_phase3.py which covers only the 4 documented fixes.
"""
from __future__ import annotations
import json
import numpy as np
import pytest
from pathlib import Path
from typing import Dict, List

from backend.pipeline.schemas import (
    AudioFeatures, AudioFeatureSegment,
    VisualFeatures, VisualFrameFeature,
    TextFeatures, TextFeatureSegment,
    Segment, SegmentEvidence, Chapter, MetaRaw,
)
from backend.pipeline.fusion.align import build_per_second_grid
from backend.pipeline.fusion.boundaries import find_boundaries, MIN_BOUNDARY_GAP_SEC
from backend.pipeline.fusion.classify import (
    _classify_by_clip_and_audio,
    _resolve_rule_label_for_segment,
    classify_segments,
)
from backend.pipeline.fusion.rules import (
    RuleHit, rule_holding_screen, run_all_rules,
)
from backend.pipeline.fusion.smooth import (
    merge_adjacent_same_label,
    absorb_short_segments,
    renumber,
    smooth_pipeline,
    split_oversized_segments,
    propagate_context,
    promote_stranded_candidates,
    MIN_SEGMENT_DURATION,
    MAX_SEG_DURATION,
    CONTEXT_FILLER_MAX_SEC,
    PROMOTE_THRESHOLD,
)
from backend.pipeline.fusion.export import (
    build_chapters,
    build_skip_suggestions,
    export_metadata,
)
from backend.pipeline.workspace import Workspace


# ── Shared builders ───────────────────────────────────────────────────────────

def _seg(sid: int, start: float, end: float, label: str,
         confidence: float = 0.8, summary: str = "") -> Segment:
    return Segment(
        segment_id=sid, start_sec=start, end_sec=end,
        label=label, confidence=confidence,
        evidence=SegmentEvidence(), summary=summary,
    )


def _make_audio(segments_kwargs: List[dict]) -> AudioFeatures:
    segs = [AudioFeatureSegment(**kw) for kw in segments_kwargs]
    return AudioFeatures(segments=segs)


def _make_visual(frames_kwargs: List[dict]) -> VisualFeatures:
    frames = [VisualFrameFeature(**kw) for kw in frames_kwargs]
    return VisualFeatures(frames=frames)


def _make_text(segs_kwargs: List[dict]) -> TextFeatures:
    segs = [TextFeatureSegment(**kw) for kw in segs_kwargs]
    return TextFeatures(segments=segs)


def _minimal_audio(T: int) -> AudioFeatures:
    return _make_audio([
        dict(start=float(t), end=float(t + 1),
             is_speech=False, rms_energy=0.0, spectral_centroid=0.0,
             spectral_bandwidth=0.0, zero_crossing_rate=0.0,
             spectral_entropy=0.0, audio_class="silence")
        for t in range(T)
    ])


def _minimal_visual(T: int) -> VisualFeatures:
    return _make_visual([
        dict(frame_index=t, timestamp_sec=float(t),
             hist_diff_to_previous=0.0, is_hard_cut=False, clip_labels={})
        for t in range(T)
    ])


def _minimal_text() -> TextFeatures:
    return TextFeatures(segments=[])


def _grid(T: int, **overrides) -> Dict[str, np.ndarray]:
    g: Dict[str, np.ndarray] = {
        "is_speech":        np.zeros(T, dtype=np.int8),
        "is_hard_cut":      np.zeros(T, dtype=np.int8),
        "hist_diff":        np.zeros(T, dtype=np.float32),
        "text_sim_to_next": np.full(T, np.nan, dtype=np.float64),
        "has_text":         np.zeros(T, dtype=np.float32),
        "rms_energy":       np.zeros(T, dtype=np.float32),
    }
    g.update(overrides)
    return g


# ─────────────────────────────────────────────────────────────────────────────
# align.py — build_per_second_grid
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildPerSecondGrid:
    def test_grid_length_equals_ceil_duration(self):
        T = 5
        grid = build_per_second_grid(
            _minimal_visual(T), _minimal_audio(T), _minimal_text(), float(T)
        )
        assert len(grid["is_speech"]) == T

    def test_fractional_duration_rounds_up(self):
        audio = _minimal_audio(4)
        grid = build_per_second_grid(
            _minimal_visual(4), audio, _minimal_text(), 3.7
        )
        assert len(grid["is_speech"]) == 4  # ceil(3.7) = 4

    def test_speech_segment_sets_is_speech(self):
        T = 5
        audio = _make_audio([
            dict(start=2.0, end=3.0, is_speech=True, rms_energy=0.1,
                 spectral_centroid=0.0, spectral_bandwidth=0.0,
                 zero_crossing_rate=0.0, spectral_entropy=0.0, audio_class="speech"),
            *[dict(start=float(t), end=float(t+1), is_speech=False,
                   rms_energy=0.0, spectral_centroid=0.0, spectral_bandwidth=0.0,
                   zero_crossing_rate=0.0, spectral_entropy=0.0, audio_class="silence")
              for t in range(T) if t != 2]
        ])
        grid = build_per_second_grid(_minimal_visual(T), audio, _minimal_text(), float(T))
        assert grid["is_speech"][2] == 1
        assert grid["is_speech"][0] == 0

    def test_audio_music_sets_is_music(self):
        T = 3
        audio = _make_audio([
            dict(start=1.0, end=2.0, is_speech=False, rms_energy=0.3,
                 spectral_centroid=0.0, spectral_bandwidth=0.0,
                 zero_crossing_rate=0.0, spectral_entropy=0.0, audio_class="music"),
            dict(start=0.0, end=1.0, is_speech=False, rms_energy=0.0,
                 spectral_centroid=0.0, spectral_bandwidth=0.0,
                 zero_crossing_rate=0.0, spectral_entropy=0.0, audio_class="silence"),
            dict(start=2.0, end=3.0, is_speech=False, rms_energy=0.0,
                 spectral_centroid=0.0, spectral_bandwidth=0.0,
                 zero_crossing_rate=0.0, spectral_entropy=0.0, audio_class="silence"),
        ])
        grid = build_per_second_grid(_minimal_visual(T), audio, _minimal_text(), float(T))
        assert grid["is_music"][1] == 1
        assert grid["is_music"][0] == 0

    def test_visual_hard_cut_propagated(self):
        T = 5
        visual = _make_visual([
            dict(frame_index=t, timestamp_sec=float(t),
                 hist_diff_to_previous=0.8 if t == 3 else 0.0,
                 is_hard_cut=(t == 3), clip_labels={})
            for t in range(T)
        ])
        grid = build_per_second_grid(visual, _minimal_audio(T), _minimal_text(), float(T))
        assert grid["is_hard_cut"][3] == 1
        assert grid["is_hard_cut"][0] == 0

    def test_clip_labels_create_grid_columns(self):
        T = 3
        visual = _make_visual([
            dict(frame_index=t, timestamp_sec=float(t),
                 hist_diff_to_previous=0.0, is_hard_cut=False,
                 clip_labels={"person talking": 0.9, "blank screen": 0.1})
            for t in range(T)
        ])
        grid = build_per_second_grid(visual, _minimal_audio(T), _minimal_text(), float(T))
        assert "clip_person talking" in grid
        assert "clip_blank screen" in grid

    def test_text_segment_sets_has_text(self):
        T = 10
        text = _make_text([
            dict(id=0, start=2.0, end=5.0, text="hello",
                 similarity_to_next=None, is_potential_boundary=False, matched_keywords=[])
        ])
        grid = build_per_second_grid(_minimal_visual(T), _minimal_audio(T), text, float(T))
        assert grid["has_text"][2] == 1
        assert grid["has_text"][5] == 0  # end is exclusive
        assert grid["has_text"][0] == 0

    def test_audio_segment_beyond_T_ignored(self):
        T = 3
        audio = _make_audio([
            dict(start=float(t), end=float(t+1), is_speech=(t == 5), rms_energy=0.0,
                 spectral_centroid=0.0, spectral_bandwidth=0.0,
                 zero_crossing_rate=0.0, spectral_entropy=0.0, audio_class="silence")
            for t in range(6)
        ])
        grid = build_per_second_grid(_minimal_visual(T), audio, _minimal_text(), float(T))
        assert len(grid["is_speech"]) == T

    # Sad
    def test_empty_visual_no_clip_columns(self):
        T = 5
        grid = build_per_second_grid(
            VisualFeatures(frames=[]), _minimal_audio(T), _minimal_text(), float(T)
        )
        assert not any(k.startswith("clip_") for k in grid)

    def test_empty_audio_all_zeros(self):
        T = 5
        grid = build_per_second_grid(
            _minimal_visual(T), AudioFeatures(segments=[]), _minimal_text(), float(T)
        )
        assert np.all(grid["is_speech"] == 0)
        assert np.all(grid["rms_energy"] == 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# boundaries.py — Signals 1-4 (Signal 5 tested in test_fusion_phase3.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestBoundariesSignals:
    def test_always_includes_0_and_T(self):
        T = 20
        b = find_boundaries(_grid(T))
        assert b[0] == 0
        assert b[-1] == T

    def test_signal1_hard_cut_adds_boundary(self):
        T = 20
        g = _grid(T, is_hard_cut=np.array([1 if t == 10 else 0 for t in range(T)], dtype=np.int8))
        assert 10 in find_boundaries(g)

    def test_signal2_clip_kl_shift_adds_boundary(self):
        T = 20
        # First half: all probability on label A; second half: all on label B
        clip_a = np.array([1.0 if t < 10 else 0.0 for t in range(T)], dtype=np.float32)
        clip_b = np.array([0.0 if t < 10 else 1.0 for t in range(T)], dtype=np.float32)
        g = _grid(T, **{"clip_labelA": clip_a, "clip_labelB": clip_b})
        assert 10 in find_boundaries(g)

    def test_signal3_low_text_sim_adds_boundary(self):
        T = 20
        sim = np.full(T, np.nan, dtype=np.float64)
        sim[8] = 0.1  # very dissimilar → boundary at t+1=9
        g = _grid(T, text_sim_to_next=sim)
        assert 9 in find_boundaries(g)

    def test_signal4_speech_transition_adds_boundary(self):
        T = 20
        is_speech = np.zeros(T, dtype=np.int8)
        is_speech[10:] = 1  # speech starts at 10
        g = _grid(T, is_speech=is_speech)
        assert 10 in find_boundaries(g)

    def test_min_gap_filter_removes_nearby_duplicates(self):
        T = 30
        # Two hard cuts 2 seconds apart — gap < MIN_BOUNDARY_GAP_SEC(3) → one removed
        is_hard_cut = np.zeros(T, dtype=np.int8)
        is_hard_cut[10] = 1
        is_hard_cut[11] = 1
        g = _grid(T, is_hard_cut=is_hard_cut)
        b = find_boundaries(g)
        # Both 10 and 11 should not both survive after gap filter
        assert not (10 in b and 11 in b)

    # Sad
    def test_no_signals_returns_only_endpoints(self):
        T = 20
        b = find_boundaries(_grid(T))
        assert b == [0, T]


# ─────────────────────────────────────────────────────────────────────────────
# classify.py — _resolve_rule_label_for_segment, classify_segments
# ─────────────────────────────────────────────────────────────────────────────

class TestResolveRuleLabel:
    def _hit(self, s, e, label="sponsorship", conf=0.9):
        return RuleHit(s, e, label, "test_rule", conf)

    def test_full_coverage_returns_rule_label(self):
        label, conf, rules = _resolve_rule_label_for_segment(
            [self._hit(0, 10)], 0, 10
        )
        assert label == "sponsorship"
        assert conf == 0.9

    def test_partial_coverage_above_threshold_returns_label(self):
        label, conf, _ = _resolve_rule_label_for_segment(
            [self._hit(0, 8)], 0, 10  # 80% coverage ≥ 50% threshold
        )
        assert label == "sponsorship"

    def test_partial_coverage_below_threshold_returns_empty(self):
        label, conf, _ = _resolve_rule_label_for_segment(
            [self._hit(0, 4)], 0, 10  # 40% coverage < 50% threshold
        )
        assert label == ""
        assert conf == 0.0

    def test_no_hits_returns_empty(self):
        label, conf, rules = _resolve_rule_label_for_segment([], 0, 10)
        assert label == ""
        assert rules == []

    def test_highest_confidence_wins_when_multiple_rules(self):
        hits = [self._hit(0, 10, "intro", 0.6), self._hit(0, 10, "sponsorship", 0.9)]
        label, conf, _ = _resolve_rule_label_for_segment(hits, 0, 10)
        assert label == "sponsorship"
        assert conf == 0.9


class TestClassifyByClipAndAudio:
    def test_no_clip_keys_falls_back_to_core_content(self):
        label, _, _ = _classify_by_clip_and_audio({"is_speech": 0.8, "has_text": 0.0})
        assert label == "core_content"

    def test_blank_screen_with_no_speech_maps_to_transition(self):
        label, _, _ = _classify_by_clip_and_audio(
            {"clip_blank screen": 0.9, "is_speech": 0.0, "has_text": 0.0}
        )
        assert label == "transition"

    def test_person_talking_low_speech_demoted_to_filler(self):
        # CLIP says "person talking" but speech ratio is very low → filler
        label, _, _ = _classify_by_clip_and_audio(
            {"clip_person talking": 0.9, "is_speech": 0.0, "has_text": 0.0}
        )
        assert label == "filler"

    def test_confidence_is_between_0_and_1(self):
        _, conf, _ = _classify_by_clip_and_audio(
            {"clip_presentation slide": 0.8, "is_speech": 0.5, "has_text": 0.3}
        )
        assert 0.0 <= conf <= 1.0


class TestClassifySegments:
    def _minimal_grid(self, T: int) -> Dict[str, np.ndarray]:
        return _grid(T, **{"clip_presentation slide": np.ones(T, dtype=np.float32)})

    def test_returns_correct_segment_count(self):
        T = 30
        boundaries = [0, 10, 20, 30]
        segs = classify_segments(self._minimal_grid(T), boundaries, [], [])
        assert len(segs) == 3

    def test_segment_ids_are_sequential(self):
        T = 20
        segs = classify_segments(self._minimal_grid(T), [0, 10, 20], [], [])
        assert [s.segment_id for s in segs] == [0, 1]

    def test_rule_hit_overrides_clip(self):
        T = 20
        # CLIP strongly says "presentation slide" (core_content)
        # but a dead_air rule covers the whole segment
        hits = [RuleHit(0, 20, "dead_air", "dead_air", confidence=0.95)]
        segs = classify_segments(self._minimal_grid(T), [0, 20], hits, [])
        assert segs[0].label == "dead_air"

    def test_rule_not_covering_segment_does_not_override(self):
        T = 20
        # Rule covers seconds 15–20 only, segment is 0–10
        hits = [RuleHit(15, 20, "sponsorship", "test", confidence=0.9)]
        segs = classify_segments(self._minimal_grid(T), [0, 10, 20], hits, [])
        # First segment (0-10) should NOT be sponsorship
        assert segs[0].label != "sponsorship"

    def test_empty_boundaries_returns_no_segments(self):
        segs = classify_segments(_grid(10), [0, 10], [], [])
        # With only [0, 10] there is exactly 1 segment
        assert len(segs) == 1

    # Sad
    def test_degenerate_boundary_pair_skipped(self):
        # If boundaries contain [5, 5] (zero-length), it should be skipped
        T = 20
        segs = classify_segments(self._minimal_grid(T), [0, 5, 5, 20], [], [])
        # The 5→5 interval should be silently skipped
        assert all(s.end_sec > s.start_sec for s in segs)


# ─────────────────────────────────────────────────────────────────────────────
# rules.py — rule_holding_screen, run_all_rules
# ─────────────────────────────────────────────────────────────────────────────

class TestRuleHoldingScreen:
    def test_long_static_silence_triggers_holding_screen(self):
        T = 20
        g = _grid(T,
                  hist_diff=np.zeros(T, dtype=np.float32),
                  is_speech=np.zeros(T, dtype=np.int8))
        hits = rule_holding_screen(g)
        assert len(hits) == 1
        assert hits[0].label == "holding_screen"

    def test_motion_present_does_not_trigger(self):
        T = 20
        hist = np.full(T, 0.5, dtype=np.float32)  # lots of motion
        g = _grid(T, hist_diff=hist, is_speech=np.zeros(T, dtype=np.int8))
        hits = rule_holding_screen(g)
        assert hits == []

    def test_speech_present_does_not_trigger(self):
        T = 20
        g = _grid(T,
                  hist_diff=np.zeros(T, dtype=np.float32),
                  is_speech=np.ones(T, dtype=np.int8))
        hits = rule_holding_screen(g)
        assert hits == []

    def test_short_static_silence_below_min_duration_not_flagged(self):
        T = 10
        is_speech = np.zeros(T, dtype=np.int8)
        is_speech[7:] = 1  # only 7 static seconds, below MIN=8
        g = _grid(T, hist_diff=np.zeros(T, dtype=np.float32), is_speech=is_speech)
        hits = rule_holding_screen(g)
        assert hits == []


class TestRunAllRules:
    def test_returns_list(self):
        T = 20
        g = _grid(T)
        tf = TextFeatures(segments=[])
        hits = run_all_rules(g, tf)
        assert isinstance(hits, list)

    def test_dead_air_and_holding_screen_can_both_fire(self):
        T = 20
        g = _grid(T,
                  is_speech=np.zeros(T, dtype=np.int8),
                  rms_energy=np.zeros(T, dtype=np.float32),
                  hist_diff=np.zeros(T, dtype=np.float32))
        tf = TextFeatures(segments=[])
        hits = run_all_rules(g, tf)
        labels = {h.label for h in hits}
        assert "dead_air" in labels
        assert "holding_screen" in labels

    def test_empty_text_no_keyword_hits(self):
        T = 10
        g = _grid(T)
        hits = run_all_rules(g, TextFeatures(segments=[]))
        keyword_hits = [h for h in hits if "keyword" in h.rule_name or "intro" in h.rule_name]
        assert keyword_hits == []


# ─────────────────────────────────────────────────────────────────────────────
# smooth.py
# ─────────────────────────────────────────────────────────────────────────────

class TestMergeAdjacentSameLabel:
    def test_two_same_label_merged(self):
        segs = [_seg(0, 0, 10, "core_content"), _seg(1, 10, 20, "core_content")]
        result = merge_adjacent_same_label(segs)
        assert len(result) == 1
        assert result[0].start_sec == 0
        assert result[0].end_sec == 20

    def test_different_labels_not_merged(self):
        segs = [_seg(0, 0, 10, "core_content"), _seg(1, 10, 20, "sponsorship")]
        result = merge_adjacent_same_label(segs)
        assert len(result) == 2

    def test_merged_confidence_is_max(self):
        segs = [_seg(0, 0, 10, "intro", 0.6), _seg(1, 10, 20, "intro", 0.9)]
        result = merge_adjacent_same_label(segs)
        assert result[0].confidence == 0.9

    def test_empty_list_returns_empty(self):
        assert merge_adjacent_same_label([]) == []

    def test_single_segment_unchanged(self):
        segs = [_seg(0, 0, 10, "outro")]
        result = merge_adjacent_same_label(segs)
        assert len(result) == 1


class TestAbsorbShortSegments:
    def _short(self) -> float:
        return MIN_SEGMENT_DURATION - 0.5

    def test_short_segment_absorbed_into_neighbor(self):
        segs = [
            _seg(0, 0, 30, "core_content", 0.9),
            _seg(1, 30, 30 + self._short(), "sponsorship", 0.5),
            _seg(2, 30 + self._short(), 60, "core_content", 0.9),
        ]
        result = absorb_short_segments(segs)
        # Short middle segment should be absorbed
        assert len(result) < 3

    def test_long_segment_not_absorbed(self):
        segs = [
            _seg(0, 0, 10, "core_content"),
            _seg(1, 10, 20, "sponsorship"),
            _seg(2, 20, 30, "core_content"),
        ]
        result = absorb_short_segments(segs)
        assert len(result) == 3

    def test_single_short_segment_untouched(self):
        segs = [_seg(0, 0, self._short(), "filler")]
        result = absorb_short_segments(segs)
        assert len(result) == 1

    def test_empty_list_returns_empty(self):
        assert absorb_short_segments([]) == []


class TestRenumber:
    def test_ids_reset_to_sequential(self):
        segs = [_seg(5, 0, 10, "intro"), _seg(99, 10, 20, "outro")]
        result = renumber(segs)
        assert [s.segment_id for s in result] == [0, 1]

    def test_content_preserved(self):
        segs = [_seg(7, 0, 10, "recap", 0.75, "summary text")]
        result = renumber(segs)
        assert result[0].label == "recap"
        assert result[0].summary == "summary text"
        assert result[0].confidence == 0.75


class TestSmoothPipeline:
    def test_full_pipeline_merges_and_renumbers(self):
        segs = [
            _seg(0, 0, 10, "core_content"),
            _seg(1, 10, 20, "core_content"),  # will merge with previous
            _seg(2, 20, 30, "outro"),
        ]
        result = smooth_pipeline(segs)
        assert result[0].segment_id == 0
        assert result[0].label == "core_content"
        assert result[0].end_sec == 20.0

    def test_empty_input_returns_empty(self):
        assert smooth_pipeline([]) == []


# ─────────────────────────────────────────────────────────────────────────────
# export.py
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildChapters:
    def test_consecutive_core_content_grouped(self):
        segs = [
            _seg(0, 0, 10, "core_content", summary="First topic"),
            _seg(1, 10, 20, "core_content", summary="Second topic"),
            _seg(2, 20, 30, "outro"),
        ]
        chapters = build_chapters(segs)
        assert len(chapters) == 1
        assert chapters[0].start_sec == 0
        assert chapters[0].end_sec == 20

    def test_interrupted_core_content_creates_two_chapters(self):
        segs = [
            _seg(0, 0, 10, "core_content", summary="Part one"),
            _seg(1, 10, 20, "sponsorship"),
            _seg(2, 20, 30, "core_content", summary="Part two"),
        ]
        chapters = build_chapters(segs)
        assert len(chapters) == 2
        assert chapters[0].end_sec == 10
        assert chapters[1].start_sec == 20

    def test_no_core_content_returns_empty(self):
        segs = [_seg(0, 0, 10, "sponsorship"), _seg(1, 10, 20, "outro")]
        chapters = build_chapters(segs)
        assert chapters == []

    def test_chapter_title_from_summary(self):
        segs = [_seg(0, 0, 30, "core_content", summary="Signal processing basics")]
        chapters = build_chapters(segs)
        assert "Signal" in chapters[0].title

    def test_empty_summary_gets_fallback_title(self):
        segs = [_seg(0, 0, 30, "core_content", summary="")]
        chapters = build_chapters(segs)
        assert chapters[0].title != ""

    # Sad
    def test_empty_segments_returns_empty(self):
        assert build_chapters([]) == []


class TestBuildSkipSuggestions:
    def test_non_core_content_ids_returned(self):
        segs = [
            _seg(0, 0, 10, "core_content"),
            _seg(1, 10, 20, "sponsorship"),
            _seg(2, 20, 30, "outro"),
        ]
        skip = build_skip_suggestions(segs)
        assert skip == [1, 2]

    def test_all_core_content_returns_empty(self):
        segs = [_seg(0, 0, 10, "core_content"), _seg(1, 10, 20, "core_content")]
        assert build_skip_suggestions(segs) == []

    def test_all_non_core_returned(self):
        segs = [_seg(0, 0, 5, "intro"), _seg(1, 5, 10, "dead_air")]
        skip = build_skip_suggestions(segs)
        assert 0 in skip and 1 in skip

    # Sad
    def test_empty_list_returns_empty(self):
        assert build_skip_suggestions([]) == []


class TestExportMetadata:
    def _meta_raw(self) -> MetaRaw:
        return MetaRaw(
            filename="test.mp4", duration_sec=30.0, fps=30.0,
            width=1280, height=720,
            video_codec="h264", audio_codec="aac",
            has_audio=True, ingested_at="2026-01-01T00:00:00Z",
        )

    def test_metadata_json_written(self, tmp_path):
        ws = Workspace.for_video(Path("test.mp4"), root=tmp_path / "ws")
        ws.ensure()
        segs = [_seg(0, 0, 30, "core_content")]
        path = export_metadata(ws, self._meta_raw(), segs)
        assert path.exists()

    def test_metadata_json_is_valid(self, tmp_path):
        from backend.pipeline.schemas import Metadata
        ws = Workspace.for_video(Path("test.mp4"), root=tmp_path / "ws")
        ws.ensure()
        segs = [_seg(0, 0, 10, "core_content"), _seg(1, 10, 30, "sponsorship")]
        export_metadata(ws, self._meta_raw(), segs)
        data = json.loads(ws.metadata_path.read_text())
        meta = Metadata.model_validate(data)
        assert meta.video_info.filename == "test.mp4"
        assert len(meta.segments) == 2

    def test_skip_suggestions_in_output(self, tmp_path):
        from backend.pipeline.schemas import Metadata
        ws = Workspace.for_video(Path("test.mp4"), root=tmp_path / "ws")
        ws.ensure()
        segs = [_seg(0, 0, 10, "core_content"), _seg(1, 10, 20, "sponsorship")]
        export_metadata(ws, self._meta_raw(), segs)
        data = json.loads(ws.metadata_path.read_text())
        meta = Metadata.model_validate(data)
        assert 1 in meta.skip_suggestions
        assert 0 not in meta.skip_suggestions

    def test_verified_by_human_defaults_false(self, tmp_path):
        from backend.pipeline.schemas import Metadata
        ws = Workspace.for_video(Path("test.mp4"), root=tmp_path / "ws")
        ws.ensure()
        export_metadata(ws, self._meta_raw(), [_seg(0, 0, 30, "core_content")])
        data = json.loads(ws.metadata_path.read_text())
        meta = Metadata.model_validate(data)
        assert meta.video_info.verified_by_human is False


# ─────────────────────────────────────────────────────────────────────────────
# Boundary-improvement-plan fixes
# ─────────────────────────────────────────────────────────────────────────────

def _seg_hc(sid: int, start: float, end: float, label: str,
            hard_cut: bool = False, confidence: float = 0.8) -> Segment:
    """Helper that also sets has_hard_cut_before."""
    return Segment(
        segment_id=sid, start_sec=start, end_sec=end,
        label=label, confidence=confidence,
        evidence=SegmentEvidence(), summary="",
        has_hard_cut_before=hard_cut,
    )


class TestFix1HardCutBoundaryPreserved:
    """Fix 1: merge_adjacent_same_label must not merge across a hard-cut boundary."""

    def test_hard_cut_blocks_same_label_merge(self):
        # Two core_content segments; the right one has has_hard_cut_before=True.
        segs = [
            _seg_hc(0, 0, 100, "core_content", hard_cut=False),
            _seg_hc(1, 100, 200, "core_content", hard_cut=True),
        ]
        result = merge_adjacent_same_label(segs)
        assert len(result) == 2, "Hard-cut boundary must NOT be merged away"
        assert result[0].end_sec == 100.0
        assert result[1].start_sec == 100.0

    def test_no_hard_cut_allows_merge(self):
        segs = [
            _seg_hc(0, 0, 50, "core_content", hard_cut=False),
            _seg_hc(1, 50, 100, "core_content", hard_cut=False),
        ]
        result = merge_adjacent_same_label(segs)
        assert len(result) == 1, "Without hard-cut flag, same-label segments must merge"
        assert result[0].end_sec == 100.0

    def test_different_labels_always_split(self):
        segs = [
            _seg_hc(0, 0, 50, "core_content", hard_cut=False),
            _seg_hc(1, 50, 100, "filler", hard_cut=False),
        ]
        result = merge_adjacent_same_label(segs)
        assert len(result) == 2

    def test_hard_cut_flag_preserved_on_merge(self):
        # When three segments are involved: A (no cut) + B (no cut, same label as A)
        # + C (hard cut, different label), after merge A+B the result should not have
        # hard_cut_before=True (it came from A, which has no cut at its start).
        segs = [
            _seg_hc(0, 0, 10, "core_content", hard_cut=False),
            _seg_hc(1, 10, 20, "core_content", hard_cut=False),
            _seg_hc(2, 20, 30, "filler", hard_cut=True),
        ]
        result = merge_adjacent_same_label(segs)
        assert len(result) == 2
        assert result[0].has_hard_cut_before is False
        assert result[1].has_hard_cut_before is True

    def test_classify_stamps_hard_cut_flag(self):
        """classify_segments must set has_hard_cut_before when hard_cut_set provided."""
        T = 20
        grid = _grid(T, is_hard_cut=np.array([0]*10 + [1] + [0]*9, dtype=np.int8))
        segs = classify_segments(grid, [0, 10, 20], [], [], hard_cut_set={10})
        # Segment starting at 10 should be flagged.
        seg_at_10 = next(s for s in segs if s.start_sec == 10.0)
        assert seg_at_10.has_hard_cut_before is True
        # Segment starting at 0 should NOT be flagged.
        seg_at_0 = next(s for s in segs if s.start_sec == 0.0)
        assert seg_at_0.has_hard_cut_before is False


class TestFix2NaturalBreakCandidates:
    """Fix 2: export_metadata must persist raw boundaries as natural_break_candidates."""

    def _meta_raw(self) -> MetaRaw:
        return MetaRaw(
            filename="test.mp4", duration_sec=100.0, fps=30.0,
            width=640, height=360,
            video_codec="h264", audio_codec="aac",
            has_audio=True, ingested_at="2026-01-01T00:00:00Z",
        )

    def test_break_candidates_populated(self, tmp_path):
        from backend.pipeline.schemas import Metadata
        ws = Workspace.for_video(Path("test.mp4"), root=tmp_path / "ws")
        ws.ensure()
        segs = [_seg(0, 0, 100, "core_content")]
        # Pass raw_boundaries that include internal points.
        export_metadata(ws, self._meta_raw(), segs, raw_boundaries=[0, 30, 60, 100])
        data = json.loads(ws.metadata_path.read_text())
        meta = Metadata.model_validate(data)
        # 0 and 100 (== T) should be excluded; 30 and 60 should appear.
        assert 30.0 in meta.natural_break_candidates
        assert 60.0 in meta.natural_break_candidates

    def test_endpoints_excluded(self, tmp_path):
        from backend.pipeline.schemas import Metadata
        ws = Workspace.for_video(Path("test.mp4"), root=tmp_path / "ws")
        ws.ensure()
        segs = [_seg(0, 0, 100, "core_content")]
        export_metadata(ws, self._meta_raw(), segs, raw_boundaries=[0, 50, 100])
        data = json.loads(ws.metadata_path.read_text())
        meta = Metadata.model_validate(data)
        assert 0.0 not in meta.natural_break_candidates
        assert 100.0 not in meta.natural_break_candidates

    def test_no_raw_boundaries_gives_empty_list(self, tmp_path):
        from backend.pipeline.schemas import Metadata
        ws = Workspace.for_video(Path("test.mp4"), root=tmp_path / "ws")
        ws.ensure()
        segs = [_seg(0, 0, 100, "core_content")]
        export_metadata(ws, self._meta_raw(), segs)  # no raw_boundaries arg
        data = json.loads(ws.metadata_path.read_text())
        meta = Metadata.model_validate(data)
        assert meta.natural_break_candidates == []

    def test_candidates_are_sorted(self, tmp_path):
        from backend.pipeline.schemas import Metadata
        ws = Workspace.for_video(Path("test.mp4"), root=tmp_path / "ws")
        ws.ensure()
        segs = [_seg(0, 0, 100, "core_content")]
        export_metadata(ws, self._meta_raw(), segs, raw_boundaries=[0, 80, 20, 50, 100])
        data = json.loads(ws.metadata_path.read_text())
        meta = Metadata.model_validate(data)
        cands = meta.natural_break_candidates
        assert cands == sorted(cands)


class TestFix3SplitOversized:
    """Fix 3: split_oversized_segments must bisect segments longer than MAX_SEG_DURATION."""

    def _long_grid(self, T: int, cut_at: int) -> Dict[str, np.ndarray]:
        g = _grid(T)
        g["hist_diff"] = np.zeros(T, dtype=np.float32)
        g["hist_diff"][cut_at] = 2.0  # strong hard cut at cut_at
        return g

    def test_long_segment_split_at_hardcut(self):
        T = int(MAX_SEG_DURATION) + 100
        cut = T // 2
        g = self._long_grid(T, cut)
        segs = [_seg(0, 0.0, float(T), "core_content")]
        raw_boundaries = [0, cut, T]
        result = split_oversized_segments(segs, raw_boundaries, g)
        assert len(result) == 2
        assert result[0].end_sec == float(cut)
        assert result[1].start_sec == float(cut)

    def test_short_segment_not_split(self):
        T = 100
        g = _grid(T)
        segs = [_seg(0, 0.0, float(T), "core_content")]
        raw_boundaries = [0, 50, T]
        result = split_oversized_segments(segs, raw_boundaries, g)
        assert len(result) == 1, "Segments ≤ MAX_SEG_DURATION must not be split"

    def test_no_internal_boundaries_leaves_segment_intact(self):
        T = int(MAX_SEG_DURATION) + 50
        g = _grid(T)
        segs = [_seg(0, 0.0, float(T), "core_content")]
        raw_boundaries = [0, T]  # only endpoints, no internal candidate
        result = split_oversized_segments(segs, raw_boundaries, g)
        assert len(result) == 1

    def test_right_half_flagged_as_hard_cut(self):
        T = int(MAX_SEG_DURATION) + 100
        cut = 200
        g = self._long_grid(T, cut)
        segs = [_seg(0, 0.0, float(T), "core_content")]
        result = split_oversized_segments(segs, [0, cut, T], g)
        assert result[1].has_hard_cut_before is True


class TestFix4PropagateContext:
    """Fix 4: short filler surrounded by core_content must be relabeled core_content."""

    def test_short_filler_between_core_relabeled(self):
        segs = [
            _seg(0, 0, 100, "core_content"),
            _seg(1, 100, 115, "filler"),      # 15 s < CONTEXT_FILLER_MAX_SEC=30
            _seg(2, 115, 300, "core_content"),
        ]
        result = propagate_context(segs)
        assert result[1].label == "core_content"

    def test_long_filler_between_core_not_relabeled(self):
        segs = [
            _seg(0, 0, 100, "core_content"),
            _seg(1, 100, 150, "filler"),      # 50 s > CONTEXT_FILLER_MAX_SEC=30
            _seg(2, 150, 300, "core_content"),
        ]
        result = propagate_context(segs)
        assert result[1].label == "filler"

    def test_filler_at_edge_not_relabeled(self):
        # Only one neighbor is core_content.
        segs = [
            _seg(0, 0, 10, "filler"),
            _seg(1, 10, 20, "core_content"),
        ]
        result = propagate_context(segs)
        assert result[0].label == "filler"

    def test_non_filler_not_affected(self):
        segs = [
            _seg(0, 0, 100, "core_content"),
            _seg(1, 100, 110, "sponsorship"),
            _seg(2, 110, 300, "core_content"),
        ]
        result = propagate_context(segs)
        assert result[1].label == "sponsorship"

    def test_relabeled_confidence_is_average_of_neighbors(self):
        segs = [
            _seg(0, 0, 100, "core_content", confidence=0.9),
            _seg(1, 100, 110, "filler", confidence=0.3),
            _seg(2, 110, 300, "core_content", confidence=0.7),
        ]
        result = propagate_context(segs)
        assert result[1].confidence == pytest.approx((0.9 + 0.7) / 2, abs=1e-3)

    def test_two_adjacent_fillers_not_individually_eligible(self):
        # Neither filler is flanked by core_content on BOTH sides, so neither
        # is eligible for single-pass relabeling. Both stay as filler.
        segs = [
            _seg(0, 0, 100, "core_content"),
            _seg(1, 100, 110, "filler"),
            _seg(2, 110, 120, "filler"),
            _seg(3, 120, 300, "core_content"),
        ]
        result = propagate_context(segs)
        assert result[1].label == "filler"
        assert result[2].label == "filler"

    def test_isolated_filler_flanked_on_both_sides_is_relabeled(self):
        # Single filler with core_content on both sides → relabeled.
        segs = [
            _seg(0, 0, 100, "core_content"),
            _seg(1, 100, 110, "filler"),
            _seg(2, 110, 300, "core_content"),
        ]
        result = propagate_context(segs)
        assert result[1].label == "core_content"


class TestFix5PromoteStrandedCandidates:
    """Fix 5: break candidates > PROMOTE_THRESHOLD s from any seg boundary become seg boundaries."""

    def test_stranded_candidate_is_promoted(self):
        # One long segment; raw boundary at midpoint with no nearby seg boundary.
        gap = int(PROMOTE_THRESHOLD) + 10  # well above threshold
        split_at = gap
        T = gap * 2
        segs = [_seg(0, 0.0, float(T), "core_content")]
        raw_boundaries = [0, split_at, T]
        result = promote_stranded_candidates(segs, raw_boundaries)
        assert len(result) == 2
        assert result[0].end_sec == float(split_at)
        assert result[1].start_sec == float(split_at)

    def test_nearby_candidate_not_promoted(self):
        # Segment boundary already exists within PROMOTE_THRESHOLD of the candidate.
        segs = [
            _seg(0, 0.0, 100.0, "core_content"),
            _seg(1, 100.0, 300.0, "core_content"),
        ]
        # Candidate at 110 — only 10 s from the seg boundary at 100 → no split.
        result = promote_stranded_candidates(segs, raw_boundaries=[0, 110, 300])
        assert len(result) == 2

    def test_promoted_segment_inherits_parent_label(self):
        T = int(PROMOTE_THRESHOLD) * 3
        split_at = T // 2
        segs = [_seg(0, 0.0, float(T), "sponsorship")]
        result = promote_stranded_candidates(segs, [0, split_at, T])
        assert result[0].label == "sponsorship"
        assert result[1].label == "sponsorship"

    def test_right_half_has_no_hard_cut_flag(self):
        T = int(PROMOTE_THRESHOLD) * 3
        split_at = T // 2
        segs = [_seg_hc(0, 0.0, float(T), "core_content", hard_cut=True)]
        result = promote_stranded_candidates(segs, [0, split_at, T])
        # Left half keeps parent's hard_cut_before; right half gets False (it's a speech split).
        assert result[1].has_hard_cut_before is False

    def test_empty_raw_boundaries_returns_unchanged(self):
        segs = [_seg(0, 0.0, 500.0, "core_content")]
        result = promote_stranded_candidates(segs, [])
        assert len(result) == 1

    def test_multiple_stranded_candidates_all_promoted(self):
        # Two stranded candidates, both far from any seg boundary.
        gap = int(PROMOTE_THRESHOLD) + 5
        T = gap * 3
        segs = [_seg(0, 0.0, float(T), "core_content")]
        result = promote_stranded_candidates(segs, [0, gap, gap * 2, T])
        assert len(result) == 3
        assert result[0].end_sec == float(gap)
        assert result[1].end_sec == float(gap * 2)

    def test_smooth_pipeline_includes_promotion(self):
        # Integration: smooth_pipeline with raw_boundaries promotes a stranded candidate.
        gap = int(PROMOTE_THRESHOLD) + 10
        T = gap * 2
        segs = [
            _seg(0, 0.0, float(T), "core_content"),
        ]
        result = smooth_pipeline(segs, raw_boundaries=[0, gap, T])
        boundaries = sorted({s.start_sec for s in result} | {s.end_sec for s in result})
        assert float(gap) in boundaries
