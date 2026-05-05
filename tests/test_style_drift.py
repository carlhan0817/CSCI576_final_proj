"""Tests for backend/pipeline/fusion/style_drift.py."""
from __future__ import annotations

import numpy as np

from backend.pipeline.fusion.style_drift import compute_drift


def _normalise(v):
    return v / np.linalg.norm(v)


class TestComputeDrift:
    def test_uniform_embeddings_have_zero_drift(self):
        T, D = 60, 16
        emb = np.tile(_normalise(np.ones(D)), (T, 1)).astype(np.float32)
        drift = compute_drift(emb, window_sec=15)
        assert drift.shape == (T,)
        assert np.allclose(drift, 0.0, atol=1e-5)

    def test_inserted_block_produces_drift_peak(self):
        T, D = 100, 16
        host = _normalise(np.array([1.0] + [0.0] * (D - 1)))
        ad = _normalise(np.array([0.0, 1.0] + [0.0] * (D - 2)))

        emb = np.tile(host, (T, 1)).astype(np.float32)
        emb[40:60] = ad  # 20-second ad block in the middle

        drift = compute_drift(emb, window_sec=15)

        host_mean_drift = drift[5:35].mean()  # quiet host region
        ad_mean_drift = drift[42:58].mean()   # interior of ad block
        # Cosine distance is bounded by context composition; with orthogonal
        # one-hot host/ad and balanced past+future windows, interior drift
        # peaks around 0.2-0.3 (far above the flat host baseline).
        assert ad_mean_drift > 0.15, f"ad drift too low: {ad_mean_drift}"
        assert host_mean_drift < 0.1, f"host drift not flat: {host_mean_drift}"
        # Also verify the boundary spike is unambiguous.
        assert drift.max() > 5 * (host_mean_drift + 1e-6)

    def test_short_video_does_not_crash(self):
        emb = np.random.RandomState(0).randn(5, 16).astype(np.float32)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        drift = compute_drift(emb, window_sec=15)
        assert drift.shape == (5,)
        assert np.all(np.isfinite(drift))

    def test_handles_zero_vector_safely(self):
        T, D = 30, 16
        emb = np.tile(_normalise(np.ones(D)), (T, 1)).astype(np.float32)
        emb[10] = 0.0
        drift = compute_drift(emb, window_sec=10)
        assert np.all(np.isfinite(drift))
