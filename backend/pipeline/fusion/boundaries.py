"""
Phase 3 Step 3: Boundary candidate generation.

Find time points where "something changes" — visual cuts, CLIP scene shifts,
text topic shifts, audio speech-music transitions. These become the candidate
segment boundaries.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List


# Tunable thresholds
HIST_DIFF_THRESHOLD = 0.4
CLIP_KL_THRESHOLD = 0.3
TEXT_SIM_THRESHOLD = 0.35
AUDIO_STATE_CHANGE_THRESHOLD = 1  # is_speech flips
MIN_BOUNDARY_GAP_SEC = 3  # don't put two boundaries within 3 seconds of each other


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    """Symmetric KL between two prob distributions."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(0.5 * (np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p))))


def find_boundaries(grid: Dict[str, np.ndarray]) -> List[int]:
    """
    Return a sorted list of boundary timestamps (in seconds).
    Always includes 0 and T-1 as the ends.
    """
    T = len(grid["is_speech"])
    candidate_seconds = set()
    candidate_seconds.add(0)
    candidate_seconds.add(T)

    # Signal 1: visual hard cuts
    for t in range(T):
        if grid["is_hard_cut"][t]:
            candidate_seconds.add(t)

    # Signal 2: CLIP scene probability shifts (KL divergence between adjacent seconds)
    clip_keys = sorted([k for k in grid.keys() if k.startswith("clip_")])
    if clip_keys:
        clip_matrix = np.stack([grid[k] for k in clip_keys], axis=1)  # (T, n_labels)
        for t in range(1, T):
            kl = _kl_divergence(clip_matrix[t - 1], clip_matrix[t])
            if kl > CLIP_KL_THRESHOLD:
                candidate_seconds.add(t)

    # Signal 3: text topic shifts (low cosine sim between adjacent sentences)
    text_sim = grid["text_sim_to_next"]
    for t in range(T):
        if not np.isnan(text_sim[t]) and text_sim[t] < TEXT_SIM_THRESHOLD:
            candidate_seconds.add(t + 1)  # boundary is AFTER the dissimilar sentence

    # Signal 4: audio speech ↔ silence transitions
    is_speech = grid["is_speech"]
    for t in range(1, T):
        if is_speech[t] != is_speech[t - 1]:
            candidate_seconds.add(t)

    # Sort and apply minimum-gap filter
    sorted_boundaries = sorted(candidate_seconds)
    filtered = [sorted_boundaries[0]]
    for b in sorted_boundaries[1:]:
        if b - filtered[-1] >= MIN_BOUNDARY_GAP_SEC:
            filtered.append(b)
    
    # Make sure end is included even if filtered out
    if filtered[-1] != T:
        filtered.append(T)
    
    return filtered