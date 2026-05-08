"""
Phase 3 Step 6: Generate the final metadata.json artifact.

- Build VideoInfo from meta_raw.json.
- Build chapters by merging adjacent core_content segments.
- Build skip_suggestions list of all non-content segment_ids.
- Validate the whole structure with Pydantic.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from backend.pipeline.schemas import (
    MetaRaw, Metadata, VideoInfo, Segment, Chapter, AdInsertionCandidate,
)
from backend.pipeline.workspace import Workspace
from backend.pipeline.fusion.ad_slots import score_candidates


# Minimum |lum_context_delta| at a hard cut that signals "content resuming after ad".
# The ad boundary creates a large luminance shift in one direction; when the ad ends,
# the luminance shifts back in the OPPOSITE direction by at least this much.
_LUM_END_THRESHOLD = 30.0
_MIN_AD_SEC = 15.0   # shortest ad to consider
_MAX_AD_SEC = 180.0  # longest ad to scan for end

# Number of consecutive speech-positive seconds required at a hard cut to terminate an ad.
# Ad voiceover bursts last 1-2s per product shot; a content host resumes for much longer.
_SUSTAINED_SPEECH_SEC = 3

# hc_burst_density value below which the cut rhythm has clearly returned to content levels.
# BURST_DENSITY_THRESHOLD (in burst-start detection) is 0.10; this is a clear exit margin.
_BURST_EXIT_THRESHOLD = 0.04

# CLIP content-rollup signal: combined 3s rolling probability across content-facing labels.
_CONTENT_CLIP_KEYS = (
    "clip_a person talking to a camera",
    "clip_a presentation slide",
    "clip_a screen recording of software",
)
_CLIP_CONTENT_THRESHOLD = 0.40


def _clip_content_score(t: int, grid: Dict[str, np.ndarray], T: int, window: int = 3) -> float:
    """Mean content-CLIP probability summed across content keys over [t, t+window)."""
    total = 0.0
    for k in _CONTENT_CLIP_KEYS:
        arr = grid.get(k)
        if arr is not None:
            total += float(arr[min(t, T - 1) : min(t + window, T)].mean())
    return total


def _relabel_ad_blocks(
    segments: List[Segment],
    ad_candidates: List[AdInsertionCandidate],
    grid: Optional[Dict[str, np.ndarray]],
) -> List[Segment]:
    """
    Relabel segments within detected ad blocks using insertion candidates as anchors.

    Each ad_insertion_candidate marks a visually confirmed ad start.  From that
    anchor the function scans forward for the ad END using four signals:

    Primary    — SUSTAINED speech (≥ _SUSTAINED_SPEECH_SEC consecutive seconds) at
                 a hard cut. Single-second voiceover bursts inside the ad no longer
                 trigger the end; only a host genuinely resuming does.

    Secondary  — lum_context_delta flips sign AND |delta| ≥ _LUM_END_THRESHOLD at a
                 hard cut: the screen brightness snaps back to the content level.

    Tertiary   — hc_burst_density drops below _BURST_EXIT_THRESHOLD at a hard cut:
                 the dense-cut visual edit rhythm of the ad has ended.

    Quaternary — content-facing CLIP labels (host / slide / screen-recording) sum
                 to ≥ _CLIP_CONTENT_THRESHOLD over a 3s window: visual semantics
                 confirm content has returned.

    Whichever fires first (past _MIN_AD_SEC from the anchor) ends the window.
    Falls back to _MAX_AD_SEC if no signal fires.

    Using candidates as anchors limits relabeling to at most n_ads small windows
    and avoids the false positives that arise from independent scanning.
    """
    if grid is None or not ad_candidates:
        return segments

    is_hard_cut   = grid.get("is_hard_cut")
    lum_delta     = grid.get("lum_context_delta")
    is_speech     = grid.get("is_speech")
    burst_density = grid.get("hc_burst_density")
    T = len(grid["is_speech"])

    ad_windows: List[tuple] = []  # (start_sec, end_sec)

    for cand in ad_candidates:
        t0 = int(cand.timestamp_sec)
        if t0 >= T:
            continue

        start_ld   = float(lum_delta[t0]) if lum_delta is not None else 0.0
        start_sign = 1 if start_ld >= 0 else -1  # +1 = entered brighter, -1 = entered darker

        t_end = min(t0 + int(_MAX_AD_SEC) + 1, T)  # default: cover full max window
        for t in range(t0 + int(_MIN_AD_SEC), min(t0 + int(_MAX_AD_SEC) + 1, T)):
            hc_t  = is_hard_cut is not None and bool(is_hard_cut[t])
            ld_t  = float(lum_delta[t]) if lum_delta is not None else 0.0
            sp_t  = is_speech is not None and bool(is_speech[t])

            # Primary: SUSTAINED speech at a hard cut → host has resumed (Fix A).
            # Single-second bursts inside ad voiceover no longer trigger the end.
            if hc_t and sp_t and is_speech is not None:
                run = sum(
                    bool(is_speech[min(t + k, T - 1)])
                    for k in range(_SUSTAINED_SPEECH_SEC)
                )
                if run >= _SUSTAINED_SPEECH_SEC:
                    t_end = t + 1
                    break

            # Secondary: luminance flips back with magnitude ≥ threshold at a hard cut.
            if hc_t and (ld_t * start_sign) < 0 and abs(ld_t) >= _LUM_END_THRESHOLD:
                t_end = t + 1
                break

            # Tertiary: hc_burst_density drops below content level at a hard cut (Fix B).
            # Targets burst-detected ads (e.g. UberEats) where the visual edit rhythm
            # change is clearer than any audio signal.
            if hc_t and burst_density is not None:
                bd_t = float(burst_density[min(t, T - 1)])
                if bd_t < _BURST_EXIT_THRESHOLD:
                    t_end = t + 1
                    break

            # Quaternary: CLIP content-label rollup crosses threshold at a hard cut (Fix C).
            # Audio-independent visual semantic signal — fires when the screen is clearly
            # back on host/slide/screen-recording content.
            if hc_t and _clip_content_score(t, grid, T) >= _CLIP_CONTENT_THRESHOLD:
                t_end = t + 1
                break

        ad_windows.append((float(t0), float(t_end)))

    result: List[Segment] = []
    for seg in segments:
        mid = (seg.start_sec + seg.end_sec) / 2.0
        in_ad = any(w_start <= mid < w_end for w_start, w_end in ad_windows)
        if in_ad and seg.label != "ad":
            seg = seg.model_copy(update={"label": "ad", "confidence": 0.9})
        result.append(seg)
    return result


def build_chapters(segments: List[Segment]) -> List[Chapter]:
    """
    Group consecutive core_content segments into chapters.
    Title = first 60 chars of the first segment's summary (or "Chapter N" if empty).
    """
    chapters: List[Chapter] = []
    chapter_id = 0
    current_start = None
    current_summary = ""
    
    for seg in segments:
        if seg.label == "core_content":
            if current_start is None:
                current_start = seg.start_sec
                current_summary = seg.summary
            current_end = seg.end_sec
        else:
            if current_start is not None:
                title = (current_summary[:60].rsplit(" ", 1)[0] + "...") if current_summary else f"Chapter {chapter_id + 1}"
                chapters.append(Chapter(
                    chapter_id=chapter_id,
                    start_sec=current_start,
                    end_sec=current_end,
                    title=title.strip(),
                ))
                chapter_id += 1
                current_start = None
                current_summary = ""
    
    # Tail
    if current_start is not None:
        title = (current_summary[:60].rsplit(" ", 1)[0] + "...") if current_summary else f"Chapter {chapter_id + 1}"
        chapters.append(Chapter(
            chapter_id=chapter_id,
            start_sec=current_start,
            end_sec=current_end,
            title=title.strip(),
        ))
    
    return chapters


def build_skip_suggestions(segments: List[Segment]) -> List[int]:
    """All non-core_content segment_ids."""
    return [s.segment_id for s in segments if s.label != "core_content"]


def export_metadata(
    workspace: Workspace,
    meta_raw: MetaRaw,
    segments: List[Segment],
    raw_boundaries: Optional[List[int]] = None,
    grid: Optional[Dict[str, np.ndarray]] = None,
    n_ads: int = 3,
) -> Path:
    """Write metadata.json and return its path.

    raw_boundaries: the full boundary list from find_boundaries() before smoothing.
        Timestamps at 0 and video-end are excluded from natural_break_candidates
        since they are trivial endpoints, not content transitions.
    grid: per-second feature grid used to score ad insertion candidates.
    n_ads: number of ad insertion slots to recommend (default 3).
    """
    video_info = VideoInfo(
        filename=meta_raw.filename,
        duration_sec=meta_raw.duration_sec,
        fps=meta_raw.fps,
        width=meta_raw.width,
        height=meta_raw.height,
        analysis_version="0.3.0",
        verified_by_human=False,
    )

    # Build natural_break_candidates: every pre-smooth boundary except 0 and T.
    if raw_boundaries:
        T = int(meta_raw.duration_sec)
        break_candidates = sorted(
            float(b) for b in raw_boundaries if b != 0 and b != T
        )
    else:
        break_candidates = []

    # Score and rank external ad insertion slots (uses original segment labels).
    ad_candidates = score_candidates(
        segments, break_candidates, grid, meta_raw.duration_sec, n_ads
    )

    # Relabel segments inside visually-detected ad blocks AFTER scoring so that
    # the transition bonus and sponsor penalty used during scoring see the original
    # classifier labels, while the stored metadata shows the correct "ad" labels.
    segments = _relabel_ad_blocks(segments, ad_candidates, grid)

    # Rebuild chapters and skip suggestions from the corrected segment labels.
    chapters = build_chapters(segments)
    skip_suggestions = build_skip_suggestions(segments)

    metadata = Metadata(
        video_info=video_info,
        segments=segments,
        chapters=chapters,
        skip_suggestions=skip_suggestions,
        natural_break_candidates=break_candidates,
        ad_insertion_candidates=ad_candidates,
    )

    workspace.metadata_path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
    return workspace.metadata_path