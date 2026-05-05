"""Phase 3: per-second visual style discontinuity from CLIP embeddings.

For each second t, drift[t] is the cosine distance between the embedding at
second t and the mean embedding of its surrounding context (past and future
window of `window_sec` seconds, excluding the immediate ±2-second neighbours
to prevent self-leakage at boundaries).

A sustained run of high drift is the visual fingerprint of "this block is
foreign to its neighbourhood" — the most reliable cross-genre ad signal.
"""
from __future__ import annotations
import numpy as np


def _l2_normalise(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < eps:
        return np.zeros_like(v)
    return v / norm


def compute_drift(embeddings: np.ndarray, window_sec: int = 15) -> np.ndarray:
    """Compute per-second neighborhood drift.

    Args:
        embeddings: shape (T, D), one row per second. Need not be pre-normalised.
        window_sec: half-window in seconds. Context = past[t-window..t-2] ∪ future[t+2..t+window].

    Returns:
        drift: shape (T,) float32 array in [0, 2]. Higher = more anomalous.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D (T, D); got {embeddings.shape}")
    T = embeddings.shape[0]
    drift = np.zeros(T, dtype=np.float32)

    # Pre-normalise once.
    normed = np.zeros_like(embeddings, dtype=np.float32)
    for t in range(T):
        normed[t] = _l2_normalise(embeddings[t])

    GAP = 2  # immediate-neighbor exclusion radius
    for t in range(T):
        past_lo, past_hi = max(0, t - window_sec), max(0, t - GAP)
        fut_lo, fut_hi = min(T, t + GAP + 1), min(T, t + window_sec + 1)
        ctx_chunks = []
        if past_hi > past_lo:
            ctx_chunks.append(normed[past_lo:past_hi])
        if fut_hi > fut_lo:
            ctx_chunks.append(normed[fut_lo:fut_hi])
        if not ctx_chunks:
            drift[t] = 0.0
            continue
        ctx = np.concatenate(ctx_chunks, axis=0)
        ctx_mean = _l2_normalise(ctx.mean(axis=0))
        drift[t] = float(1.0 - np.dot(normed[t], ctx_mean))

    return drift
