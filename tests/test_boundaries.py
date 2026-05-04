"""
Tests for backend/pipeline/fusion/boundaries.py (Phase 3: boundary candidate
generation).
"""
from __future__ import annotations
import numpy as np
import pytest

from backend.pipeline.fusion.boundaries import (
    HIST_DIFF_THRESHOLD,
    CLIP_KL_THRESHOLD,
    TEXT_SIM_THRESHOLD,
    MIN_BOUNDARY_GAP_SEC,
    _kl_divergence,
    find_boundaries,
)


def _grid(T: int, **arrays) -> dict:
    g = {
        "is_speech": np.zeros(T, dtype=np.int8),
        "is_music": np.zeros(T, dtype=np.int8),
        "is_hard_cut": np.zeros(T, dtype=np.int8),
        "hist_diff": np.zeros(T, dtype=np.float32),
        "text_sim_to_next": np.full(T, np.nan, dtype=np.float32),
    }
    g.update(arrays)
    return g


class TestKLDivergence:
    def test_identical_distributions_zero(self):
        p = np.array([0.5, 0.5])
        assert _kl_divergence(p, p) == pytest.approx(0.0, abs=1e-6)

    def test_different_distributions_positive(self):
        p = np.array([0.9, 0.1])
        q = np.array([0.1, 0.9])
        assert _kl_divergence(p, q) > 0

    def test_symmetric(self):
        p = np.array([0.7, 0.3])
        q = np.array([0.4, 0.6])
        assert _kl_divergence(p, q) == pytest.approx(_kl_divergence(q, p))


class TestFindBoundaries:
    def test_includes_start_and_end(self):
        T = 30
        boundaries = find_boundaries(_grid(T))
        assert 0 in boundaries
        assert T in boundaries

    def test_hard_cut_creates_boundary(self):
        T = 30
        cuts = np.zeros(T, dtype=np.int8)
        cuts[15] = 1  # one cut in the middle
        grid = _grid(T, is_hard_cut=cuts)
        boundaries = find_boundaries(grid)
        # Boundary should be at or near 15 (gap-filtered to within MIN_GAP)
        assert any(abs(b - 15) <= MIN_BOUNDARY_GAP_SEC for b in boundaries)

    def test_speech_silence_transition_creates_boundary(self):
        T = 30
        speech = np.ones(T, dtype=np.int8)
        speech[15:] = 0  # speech ends at 15
        grid = _grid(T, is_speech=speech)
        boundaries = find_boundaries(grid)
        assert any(abs(b - 15) <= MIN_BOUNDARY_GAP_SEC for b in boundaries)

    def test_music_speech_transition_creates_boundary(self):
        T = 30
        music = np.ones(T, dtype=np.int8)
        music[15:] = 0
        grid = _grid(T, is_music=music)
        boundaries = find_boundaries(grid)
        assert any(abs(b - 15) <= MIN_BOUNDARY_GAP_SEC for b in boundaries)

    def test_text_topic_jump_creates_boundary(self):
        T = 30
        sims = np.full(T, 0.9, dtype=np.float32)  # most sentences similar
        sims[10] = 0.05  # one big topic jump (well below TEXT_SIM_THRESHOLD)
        grid = _grid(T, text_sim_to_next=sims)
        boundaries = find_boundaries(grid)
        # Boundary is placed AFTER the dissimilar sentence (t+1 = 11)
        assert any(abs(b - 11) <= MIN_BOUNDARY_GAP_SEC for b in boundaries)

    def test_clip_scene_shift_creates_boundary(self):
        T = 30
        # Two CLIP labels — both stable except for a sudden swap at t=15
        clip_a = np.zeros(T, dtype=np.float32)
        clip_b = np.zeros(T, dtype=np.float32)
        clip_a[:15] = 0.95
        clip_b[:15] = 0.05
        clip_a[15:] = 0.05
        clip_b[15:] = 0.95
        grid = _grid(T, **{"clip_label_a": clip_a, "clip_label_b": clip_b})
        boundaries = find_boundaries(grid)
        assert any(abs(b - 15) <= MIN_BOUNDARY_GAP_SEC for b in boundaries)

    def test_no_signals_yields_only_endpoints(self):
        T = 30
        boundaries = find_boundaries(_grid(T))
        # Just 0 and T (and maybe one interior because of gap-filter logic)
        assert boundaries[0] == 0
        assert boundaries[-1] == T

    def test_minimum_gap_enforced(self):
        T = 30
        # Two cuts close together — should be coalesced
        cuts = np.zeros(T, dtype=np.int8)
        cuts[5] = 1
        cuts[6] = 1  # within MIN_BOUNDARY_GAP_SEC of previous
        grid = _grid(T, is_hard_cut=cuts)
        boundaries = find_boundaries(grid)
        # Verify no two boundaries are within MIN_BOUNDARY_GAP_SEC except possibly at
        # the very end (where T may be appended even if filtered out)
        for i in range(len(boundaries) - 2):
            assert boundaries[i + 1] - boundaries[i] >= MIN_BOUNDARY_GAP_SEC

    def test_boundaries_sorted(self):
        T = 30
        cuts = np.zeros(T, dtype=np.int8)
        cuts[20] = 1
        cuts[10] = 1
        grid = _grid(T, is_hard_cut=cuts)
        boundaries = find_boundaries(grid)
        assert boundaries == sorted(boundaries)