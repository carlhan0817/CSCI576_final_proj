"""
scripts/diagnose_ad_block.py

Per-GT-ad diagnostic for backend.pipeline.fusion.rules.rule_ad_block.

For each inserted ad in ground_truth/test_*.json, walk the four gates
(A drift threshold, B sustained run, C commercial signal, D boundary cuts)
and report which one(s) fail. Also measures the L-cut/J-cut tolerance
window (Lcut_min_W) needed for a 'both drifts hot' branch to fire.

Adds a length-matched random non-ad baseline (1 sample per GT ad) plus a
full-track drift histogram for each video.

Output: stdout. No assertions. Pure exploration.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from backend.pipeline.workspace import Workspace
from backend.pipeline.fusion.align import load_phase2_features, build_per_second_grid
from backend.pipeline.fusion.rules import (
    AD_BLOCK_MIN_DURATION,
    AD_BLOCK_BOUNDARY_TOLERANCE,
    VISUAL_BLOCK_DRIFT_THRESHOLD,
    AUDIO_BLOCK_DRIFT_THRESHOLD,
)
# Legacy gate-A/B threshold for the (now-deprecated) raw drift signals.
# The old `rule_ad_block` used a single 0.30 cutoff; we keep it here purely so
# the original A/B/C/D diagnostic table still renders for historical context.
AD_BLOCK_DRIFT_THRESHOLD = 0.30
from backend.pipeline.schemas import MetaRaw, VisualFeatures, AudioFeatures


# --- Step 1 (Path X): block_drift exploration configs ---------------------- #
# Visual side: CLIP embeddings (already L2-normalised by compute_drift).
# Each tuple is (label, near_skip, far_window). near_skip = exclude ±near_skip
# seconds around t; far_window = total half-window. The current production
# `compute_drift` is equivalent to (near_skip=2, far_window=15).
VISUAL_BLOCK_CONFIGS = [
    ("local_15",   2,  15),  # baseline: current production drift
    ("far_20_60",  20, 60),
    ("far_30_90",  30, 90),
]
# Audio side: same configs but tested twice — raw MFCC vs per-column z-scored
# MFCC. z-score is orthogonal to the window choice (axis 2), so we run the
# full cartesian product to see which knob (window or normalisation) does
# the work.
AUDIO_BLOCK_CONFIGS = VISUAL_BLOCK_CONFIGS  # same windows, replicated raw + zscore


COMMERCIAL_SIGNAL_KEYS = (
    "has_url", "has_price", "has_phone", "has_cta",
    "has_brand_lockup", "text_has_cta",
)
SIGNAL_SHORT = {
    "has_url": "URL", "has_price": "PRC", "has_phone": "PHN",
    "has_cta": "CTA", "has_brand_lockup": "BRD", "text_has_cta": "TXT",
}
W_MAX_SEARCH = 30
HIST_BINS = 20
HIST_MAX = 1.0


# --------------------------- Binary array utilities --------------------------- #

def morph_dilate_1d(x: np.ndarray, w: int) -> np.ndarray:
    """1D binary dilation by w on each side."""
    out = x.astype(bool).copy()
    if w <= 0:
        return out
    for offset in range(1, w + 1):
        out[offset:] |= x[:-offset].astype(bool)
        out[:-offset] |= x[offset:].astype(bool)
    return out


def longest_run(arr: np.ndarray) -> int:
    """Length of the longest run of 1s."""
    best = cur = 0
    for v in arr:
        if v:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


# --------------------------- GT loaders --------------------------- #

def load_gt(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def gt_ad_windows(gt: dict) -> List[Tuple[int, int, str]]:
    out = []
    for ad in gt["inserted_ads"]:
        s = int(np.floor(ad["final_video_ad_start_seconds"]))
        e = int(np.ceil(ad["final_video_ad_end_seconds"]))
        out.append((s, e, ad["ad_filename"]))
    return out


def non_ad_intervals(gt: dict) -> List[Tuple[int, int]]:
    out = []
    for seg in gt["timeline_segments"]:
        if seg["type"] == "video_content":
            s = int(np.ceil(seg["final_video_start_seconds"]))
            e = int(np.floor(seg["final_video_end_seconds"]))
            if e > s:
                out.append((s, e))
    return out


def sample_baseline(rng: random.Random,
                    intervals: List[Tuple[int, int]],
                    length: int) -> Optional[Tuple[int, int]]:
    candidates = [(s, e) for s, e in intervals if (e - s) >= length]
    if not candidates:
        return None
    s_iv, e_iv = rng.choice(candidates)
    s = rng.randint(s_iv, e_iv - length)
    return s, s + length


# --------------------------- Per-window analysis --------------------------- #

def analyze_window(grid: Dict[str, np.ndarray], s: int, e: int) -> Dict:
    T = len(grid["is_speech"])
    s = max(0, s)
    e = min(T, e)
    if e <= s:
        return {}

    v = grid.get("style_drift", np.zeros(T))[s:e]
    a = grid.get("audio_drift", np.zeros(T))[s:e]
    cuts = grid.get("is_hard_cut", np.zeros(T, dtype=np.int8))

    v_hot = v >= AD_BLOCK_DRIFT_THRESHOLD
    a_hot = a >= AD_BLOCK_DRIFT_THRESHOLD
    or_hot = (v_hot | a_hot).astype(np.int8)

    c_signals = {}
    for key in COMMERCIAL_SIGNAL_KEYS:
        arr = grid.get(key)
        c_signals[key] = int(arr[s:e].sum()) if arr is not None else 0
    c_total = sum(c_signals.values())

    def cut_near(t: int) -> bool:
        lo = max(0, t - AD_BLOCK_BOUNDARY_TOLERANCE)
        hi = min(len(cuts), t + AD_BLOCK_BOUNDARY_TOLERANCE + 1)
        return bool(cuts[lo:hi].any())

    lag = int(np.argmax(a) - np.argmax(v)) if (v.size and a.size) else 0

    # Lcut_min_W: smallest W such that dilate(v_hot,W) & dilate(a_hot,W)
    # has a >= AD_BLOCK_MIN_DURATION run.
    min_W: Optional[int] = None
    if v_hot.any() and a_hot.any():
        v_int = v_hot.astype(np.int8)
        a_int = a_hot.astype(np.int8)
        for W in range(0, W_MAX_SEARCH + 1):
            both = (morph_dilate_1d(v_int, W) & morph_dilate_1d(a_int, W)).astype(np.int8)
            if longest_run(both) >= AD_BLOCK_MIN_DURATION:
                min_W = W
                break

    B_max_run = longest_run(or_hot)
    return {
        "len": e - s,
        "A_v_max": float(v.max()) if v.size else 0.0,
        "A_a_max": float(a.max()) if a.size else 0.0,
        "A_v_hot": int(v_hot.sum()),
        "A_a_hot": int(a_hot.sum()),
        "B_max_run": B_max_run,
        "B_pass": B_max_run >= AD_BLOCK_MIN_DURATION,
        "C_signals": c_signals,
        "C_total": c_total,
        "D_left": cut_near(s),
        "D_right": cut_near(e - 1),
        "Lcut_lag": lag,
        "Lcut_min_W": min_W,
    }


def verdict(row: Dict) -> str:
    if row["A_v_hot"] == 0 and row["A_a_hot"] == 0:
        return "A FAIL  (drift never reaches threshold)"
    if not row["B_pass"]:
        return "B FAIL  (drift hot but not sustained 10s)"
    if row["C_total"] == 0:
        if row["Lcut_min_W"] is not None and row["D_left"] and row["D_right"]:
            return f"C FAIL → alt-path candidate (W={row['Lcut_min_W']}, both bounds cut)"
        return "C FAIL  (no commercial signal; alt-path NOT viable)"
    return "PASSES (should fire)"


# --------------------------- Histograms --------------------------- #

def drift_histogram(arr: np.ndarray, bins: int = HIST_BINS, hi: float = HIST_MAX) -> List[int]:
    edges = np.linspace(0, hi, bins + 1)
    counts, _ = np.histogram(arr, bins=edges)
    return counts.tolist()


def render_histogram(counts: List[int], hi: float = HIST_MAX) -> str:
    edges = np.linspace(0, hi, len(counts) + 1)
    total = sum(counts) or 1
    max_count = max(counts) or 1
    lines = []
    for i, c in enumerate(counts):
        bar_len = int(40 * c / max_count)
        pct = 100 * c / total
        lines.append(f"  [{edges[i]:.2f}, {edges[i+1]:.2f})  {c:6d}  {pct:5.1f}%  {'#' * bar_len}")
    return "\n".join(lines)


# --------------------------- Pretty-print --------------------------- #

HEADER = (
    "  {tag:<12}  {win:<14}  {ln:>4}  "
    "{vmax:>6}  {amax:>6}  {vh:>5}  {ah:>5}  {br:>5}  {bp:<3}  "
    "{sig:<24}  {bd:<5}  {lag:>4}  {mw:>5}  {verdict}"
)
HEADER_TITLE = (
    "  ad#/file      [s, e)              len  "
    "v_max  a_max  v_hot  a_hot  B_run  B_ok "
    "URL/PRC/PHN/CTA/BRD/TXT   L/R    lag  minW   verdict"
)


def fmt_signals(sig: Dict[str, int]) -> str:
    return "/".join(str(sig[k]) for k in COMMERCIAL_SIGNAL_KEYS)


def render_row(tag: str, s: int, e: int, row: Dict) -> str:
    return HEADER.format(
        tag=tag,
        win=f"[{s},{e})",
        ln=row["len"],
        vmax=f"{row['A_v_max']:.2f}",
        amax=f"{row['A_a_max']:.2f}",
        vh=row["A_v_hot"],
        ah=row["A_a_hot"],
        br=row["B_max_run"],
        bp="Y" if row["B_pass"] else "-",
        sig=fmt_signals(row["C_signals"]),
        bd=("Y" if row["D_left"] else "-") + "/" + ("Y" if row["D_right"] else "-"),
        lag=f"{row['Lcut_lag']:+d}",
        mw=str(row["Lcut_min_W"]) if row["Lcut_min_W"] is not None else "n/a",
        verdict=verdict(row),
    )


# --------------------------- Per-video driver --------------------------- #

def diagnose_video(workspace: Workspace, gt_path: Path, rng: random.Random) -> None:
    meta = MetaRaw.model_validate(json.loads(workspace.meta_raw_path.read_text(encoding="utf-8")))
    visual, audio, text = load_phase2_features(workspace)
    grid = build_per_second_grid(visual, audio, text, meta.duration_sec)
    T = len(grid["is_speech"])

    gt = load_gt(gt_path)
    ads = gt_ad_windows(gt)
    non_ad = non_ad_intervals(gt)

    print(f"\n{'=' * 90}")
    print(f"=== {gt_path.stem}  T={T}s  thresh={AD_BLOCK_DRIFT_THRESHOLD}  "
          f"min_dur={AD_BLOCK_MIN_DURATION}s  bound_tol={AD_BLOCK_BOUNDARY_TOLERANCE}s")
    print('=' * 90)

    print("\nGT ads:")
    print(HEADER_TITLE)
    for i, (s, e, fn) in enumerate(ads, 1):
        row = analyze_window(grid, s, e)
        print(render_row(f"ad{i} {fn}", s, e, row))

    print("\nBaseline (length-matched random non-ad windows):")
    print(HEADER_TITLE)
    for i, (s, e, _fn) in enumerate(ads, 1):
        length = e - s
        win = sample_baseline(rng, non_ad, length)
        if win is None:
            print(f"  base{i}        (no window of length {length}s available)")
            continue
        bs, be = win
        row = analyze_window(grid, bs, be)
        print(render_row(f"base{i}", bs, be, row))

    print("\nFull-track drift histograms:")
    v_full = grid.get("style_drift", np.zeros(T))
    a_full = grid.get("audio_drift", np.zeros(T))
    print(f"  style_drift  (max={v_full.max():.3f}, mean={v_full.mean():.3f}, "
          f">={AD_BLOCK_DRIFT_THRESHOLD}: {int((v_full >= AD_BLOCK_DRIFT_THRESHOLD).sum())}s "
          f"of {T}s = {100 * (v_full >= AD_BLOCK_DRIFT_THRESHOLD).sum() / T:.1f}%)")
    print(render_histogram(drift_histogram(v_full)))
    print(f"  audio_drift  (max={a_full.max():.3f}, mean={a_full.mean():.3f}, "
          f">={AD_BLOCK_DRIFT_THRESHOLD}: {int((a_full >= AD_BLOCK_DRIFT_THRESHOLD).sum())}s "
          f"of {T}s = {100 * (a_full >= AD_BLOCK_DRIFT_THRESHOLD).sum() / T:.1f}%)")
    print(render_histogram(drift_histogram(a_full)))


# --------------------------- Step 1: block_drift exploration --------------- #

def _l2_normalise(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


def compute_block_drift(matrix: np.ndarray, near_skip: int, far_window: int) -> np.ndarray:
    """Asymmetric-context drift.

    For each second t, drift[t] = 1 - cos(row_t, mean(ctx)) where
    ctx = rows in [t-far_window, t-near_skip] U [t+near_skip+1, t+far_window+1].

    Setting near_skip > the longest expected ad duration / 2 ensures the
    context is dominated by lecture content even when t lies inside an ad
    body, which is what makes drift sustained (not just a boundary spike).
    """
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2D, got {matrix.shape}")
    T = matrix.shape[0]
    drift = np.zeros(T, dtype=np.float32)
    normed = np.zeros_like(matrix, dtype=np.float32)
    for t in range(T):
        normed[t] = _l2_normalise(matrix[t])

    for t in range(T):
        past_lo = max(0, t - far_window)
        past_hi = max(0, t - near_skip)
        fut_lo = min(T, t + near_skip + 1)
        fut_hi = min(T, t + far_window + 1)
        chunks = []
        if past_hi > past_lo:
            chunks.append(normed[past_lo:past_hi])
        if fut_hi > fut_lo:
            chunks.append(normed[fut_lo:fut_hi])
        if not chunks:
            drift[t] = 0.0
            continue
        ctx = np.concatenate(chunks, axis=0)
        ctx_mean = _l2_normalise(ctx.mean(axis=0))
        drift[t] = float(1.0 - np.dot(normed[t], ctx_mean))

    return drift


def zscore_columns(matrix: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Per-column standardisation: each MFCC dim → mean 0, std 1."""
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    return (matrix - mean) / (std + eps)


def build_visual_embedding_matrix(visual: VisualFeatures, T: int) -> Optional[np.ndarray]:
    if not visual.frames or not visual.frames[0].clip_embedding:
        return None
    D = len(visual.frames[0].clip_embedding)
    M = np.zeros((T, D), dtype=np.float32)
    for f in visual.frames:
        t = int(f.timestamp_sec)
        if t >= T or not f.clip_embedding:
            continue
        M[t] = np.array(f.clip_embedding, dtype=np.float32)
    return M


def build_mfcc_matrix(audio: AudioFeatures, T: int) -> Optional[np.ndarray]:
    if not audio.segments or not audio.segments[0].mfcc:
        return None
    D = len(audio.segments[0].mfcc)
    M = np.zeros((T, D), dtype=np.float32)
    for seg in audio.segments:
        t = int(np.floor(seg.start))
        if t >= T or not seg.mfcc:
            continue
        M[t] = np.array(seg.mfcc, dtype=np.float32)
    return M


def non_ad_mask(gt: dict, T: int) -> np.ndarray:
    """Boolean mask of length T: True where the second is non-ad video_content."""
    mask = np.zeros(T, dtype=bool)
    for s, e in non_ad_intervals(gt):
        mask[max(0, s):min(T, e)] = True
    return mask


def summarise_window(drift: np.ndarray, s: int, e: int, threshold: float) -> Dict:
    seg = drift[max(0, s):min(len(drift), e)]
    if seg.size == 0:
        return {"max": 0.0, "p50": 0.0, "p25": 0.0, "hot_sec": 0, "max_run": 0}
    hot = (seg >= threshold).astype(np.int8)
    return {
        "max": float(seg.max()),
        "p50": float(np.percentile(seg, 50)),
        "p25": float(np.percentile(seg, 25)),
        "hot_sec": int(hot.sum()),
        "max_run": longest_run(hot),
    }


def explore_block_drift(workspace: Workspace, gt_path: Path, rng: random.Random) -> None:
    """Step 1: compare local_15 vs far_20_60 vs far_30_90, plus MFCC z-score axis."""
    meta = MetaRaw.model_validate(json.loads(workspace.meta_raw_path.read_text(encoding="utf-8")))
    visual, audio, _text = load_phase2_features(workspace)
    T = int(np.ceil(meta.duration_sec))

    gt = load_gt(gt_path)
    ads = gt_ad_windows(gt)
    non_ad_iv = non_ad_intervals(gt)
    non_ad = non_ad_mask(gt, T)

    print(f"\n{'#' * 90}")
    print(f"### Step 1 block_drift exploration — {gt_path.stem}  T={T}s")
    print('#' * 90)

    # Pre-build sampled baseline windows once so each config sees the same set
    baselines: List[Tuple[int, int]] = []
    for s, e, _fn in ads:
        win = sample_baseline(rng, non_ad_iv, e - s)
        baselines.append(win if win else (-1, -1))

    # ---- Visual side ---- #
    emb_v = build_visual_embedding_matrix(visual, T)
    if emb_v is None:
        print("  [skip] no CLIP embeddings available")
    else:
        print("\n--- VISUAL (CLIP embeddings) ---")
        for label, near, far in VISUAL_BLOCK_CONFIGS:
            drift = compute_block_drift(emb_v, near, far)
            _print_drift_block(label, "visual", drift, ads, baselines, non_ad)

    # ---- Audio side: raw MFCC ---- #
    mfcc = build_mfcc_matrix(audio, T)
    if mfcc is None:
        print("  [skip] no MFCC available")
    else:
        print("\n--- AUDIO (MFCC, raw) ---")
        for label, near, far in AUDIO_BLOCK_CONFIGS:
            drift = compute_block_drift(mfcc, near, far)
            _print_drift_block(label, "audio_raw", drift, ads, baselines, non_ad)

        print("\n--- AUDIO (MFCC, per-dim z-score) ---")
        mfcc_z = zscore_columns(mfcc)
        for label, near, far in AUDIO_BLOCK_CONFIGS:
            drift = compute_block_drift(mfcc_z, near, far)
            _print_drift_block(label, "audio_zsc", drift, ads, baselines, non_ad)


def _print_drift_block(
    cfg_label: str,
    modality: str,
    drift: np.ndarray,
    ads: List[Tuple[int, int, str]],
    baselines: List[Tuple[int, int]],
    non_ad: np.ndarray,
) -> None:
    """Render one (modality, config) section."""
    non_ad_vals = drift[non_ad]
    if non_ad_vals.size == 0:
        threshold = 0.30  # fallback
    else:
        threshold = float(np.percentile(non_ad_vals, 90))
    stats_full = (
        f"max={drift.max():.3f} mean={drift.mean():.3f} "
        f"non_ad_p90(=thresh)={threshold:.3f}"
    )
    print(f"  [{modality}] {cfg_label:<10}  ({stats_full})")
    for i, (s, e, fn) in enumerate(ads, 1):
        stats = summarise_window(drift, s, e, threshold)
        verdict_str = _ad_verdict(stats)
        print(f"    ad{i:<2} [{s},{e})  len={e-s:>3}  "
              f"max={stats['max']:.2f} p50={stats['p50']:.2f} p25={stats['p25']:.2f} "
              f"hot={stats['hot_sec']:>3} run={stats['max_run']:>3}  {verdict_str}")
    for i, (s, e) in enumerate(baselines, 1):
        if s < 0:
            continue
        stats = summarise_window(drift, s, e, threshold)
        verdict_str = _baseline_verdict(stats)
        print(f"    bs{i:<2} [{s},{e})  len={e-s:>3}  "
              f"max={stats['max']:.2f} p50={stats['p50']:.2f} p25={stats['p25']:.2f} "
              f"hot={stats['hot_sec']:>3} run={stats['max_run']:>3}  {verdict_str}")


def _ad_verdict(stats: Dict) -> str:
    if stats["max_run"] >= AD_BLOCK_MIN_DURATION:
        return "BLOCK_HOLDS"
    if stats["max_run"] >= AD_BLOCK_MIN_DURATION // 2:
        return "marginal"
    return "BLOCK_FAILS"


def _baseline_verdict(stats: Dict) -> str:
    if stats["max_run"] >= AD_BLOCK_MIN_DURATION:
        return "FP_RISK"
    if stats["max_run"] >= AD_BLOCK_MIN_DURATION // 2:
        return "FP_marginal"
    return "clean"


# --------------------------- CLI --------------------------- #

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1],
                   help="repo root (defaults to script's parent)")
    p.add_argument("--videos", nargs="+", default=["test_001", "test_004"],
                   help="video stems to diagnose (default: test_001 test_004)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    repo: Path = args.repo
    workspace_root = repo / "workspace"
    gt_root = repo / "ground_truth"
    videos_dir = repo / "videos"

    for stem in args.videos:
        gt_path = gt_root / f"{stem}.json"
        if not gt_path.exists():
            print(f"[skip] no ground truth at {gt_path}", file=sys.stderr)
            continue
        # video file may not actually exist on disk; Workspace just uses the stem.
        ws = Workspace.for_video(videos_dir / f"{stem}.mp4", root=workspace_root)
        if not ws.audio_features_path.exists():
            print(f"[skip] no Phase 2 features for {stem}", file=sys.stderr)
            continue
        diagnose_video(ws, gt_path, rng)
        explore_block_drift(ws, gt_path, rng)

    return 0


if __name__ == "__main__":
    sys.exit(main())
