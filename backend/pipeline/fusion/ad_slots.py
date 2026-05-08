"""
Ad insertion slot scoring.

Combines two signals:
  1. In-video ad reads  — already captured as `ad` segments by CLIP + sponsor keywords.
  2. External break slots — natural seams where a standalone ad video can be inserted.

score_candidates() scores each segment boundary and returns the top-N recommended
insertion points as AdInsertionCandidate objects.

Candidate generation
--------------------
Three candidate sources, merged before scoring:

  a) Segment START times (excluding 0) — genuine content transitions validated by the
     fusion pipeline.  Used as fallback for unit tests with a single segment.

  b) Visual burst starts — first hard cut inside any 60-second window whose hard-cut
     density exceeds BURST_DENSITY_THRESHOLD (0.10 cuts/sec).  Commercial ad videos
     contain many rapid internal edits; this signature directly identifies ad block
     boundaries even when the fusion segmenter places no boundary there.

  c) Audio silence starts — first second of each non-speech run ≥ SILENCE_MIN_SEC
     that follows an active-speech second.  Natural speech pauses are clean splice
     points and complement the visual signal when no burst evidence exists.

Scoring
-------
  composite = W_SIGNAL * signal_strength(t)
            + W_TRANSITION * transition_bonus(t)
            - W_SPONSOR * sponsor_penalty(t)

  signal_strength   — tiered by burst evidence from the grid:
                        1.00  hard cut + burst_density≥0.10 + lum_delta≥+30
                        0.95  hard cut + burst_density≥0.10
                        0.90  hard cut + lum_delta≥+40
                        0.80  hard cut (plain, no burst context)
                        0.70  speech→silence
                        0.50  silence→speech
                        0.40  hist_diff > 0.3
                        0.20  baseline
                      Falls back to 1.0 (legacy) when hc_burst_density is absent.
  transition_bonus  — 1.0 when the label changes and neither side is a non-chapter label;
                      burst-detected candidates always receive 1.0 (visual break confirmed);
                      silence-detected candidates receive max(computed_bonus, 0.5) — a floor
                      that acknowledges audio evidence of a genuine content pause
  sponsor_penalty   — penalty [0,1] for proximity to an in-video ad segment;
                      burst-detected candidates bypass this (they ARE the ad boundary)

Selection filters
-----------------
  MIN_BLOCK_SEC       — both flanking content blocks must be ≥ 60 s
  MIN_AD_SPACING_SEC  — selected candidates must be ≥ 180 s apart; also enforced as a
                        minimum buffer before the video end
  Fix 2: ad_precursor_set anchors only off "ad" segments ≥ 8 s or rule-confirmed,
         so short CLIP-only blips do not falsely elevate nearby candidates.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from backend.pipeline.schemas import AdInsertionCandidate, Segment


# ── Tunable constants ──────────────────────────────────────────────────────────
MIN_BLOCK_SEC = 60.0          # minimum content duration on each side of a slot
MIN_AD_SPACING_SEC = 180.0    # minimum gap between any two selected slots
SPONSOR_EXCLUSION_SEC = 60.0  # penalty radius around in-video ad segments
# Minimum composite score required for the 4th+ candidate.  The first three slots
# are always filled (backward-compatible behaviour for 3-ad videos); beyond that,
# only select if there is strong enough evidence to justify an extra break.
MIN_COMPOSITE_SCORE_EXTRA = 0.60

W_SIGNAL     = 0.4  # weight for break signal strength
W_TRANSITION = 0.5  # weight for label-change transition bonus (primary discriminator)
W_SPONSOR    = 0.1  # weight for sponsor-proximity penalty

# Visual burst detection thresholds
BURST_DENSITY_THRESHOLD       = 0.10   # hard cuts per second in a 60s window → ad block
LUM_JUMP_THRESHOLD            = 25.0   # luminance increase at boundary → content→ad jump
LUM_JUMP_CORROBORATION_THRESHOLD = 150.0  # |lum_delta| within ±10 s of a silence candidate
                                           # → audio+visual corroboration → burst-level priority

# Audio silence detection thresholds
SILENCE_MIN_SEC = 2   # non-speech run must last ≥ N seconds to be a candidate


# ── Signal scoring ─────────────────────────────────────────────────────────────

def _signal_strength(t: float, grid: Dict[str, np.ndarray]) -> float:
    """
    Score how strong the detected boundary signal is at second t.

    When hc_burst_density is present in the grid, hard cuts are tiered by
    burst evidence (commercial ads have many rapid internal edits):
      1.00  hard cut + burst_density ≥ 0.10  AND  |lum_delta| ≥ 30
      0.95  hard cut + burst_density ≥ 0.10
      0.90  hard cut + |lum_delta| ≥ 40
      0.80  hard cut (plain, no burst context)
    Falls back to 1.0 for any hard cut when hc_burst_density is absent.
    Uses |lum_delta| (absolute value) so large brightness changes in either
    direction (content→bright-ad or content→dark-ad) receive the same boost.
      0.70  speech → silence
      0.50  silence → speech
      0.40  scene change (hist_diff > 0.3)
      0.20  baseline
    """
    T = len(grid["is_speech"])
    ti = min(int(t), T - 1)

    is_hard_cut   = grid.get("is_hard_cut")
    hist_diff     = grid.get("hist_diff")
    is_speech     = grid.get("is_speech")
    burst_density = grid.get("hc_burst_density")
    lum_delta     = grid.get("lum_context_delta")

    if is_hard_cut is not None and bool(is_hard_cut[ti]):
        if burst_density is None:
            return 1.0  # legacy: no burst data, treat all hard cuts equally
        bd = float(burst_density[ti])
        ld = abs(float(lum_delta[ti])) if lum_delta is not None else 0.0  # absolute value
        if bd >= BURST_DENSITY_THRESHOLD and ld >= LUM_JUMP_THRESHOLD:
            return 1.0
        if bd >= BURST_DENSITY_THRESHOLD:
            return 0.95
        if ld >= LUM_JUMP_THRESHOLD + 10.0:  # |ld| ≥ 40 threshold
            return 0.90
        return 0.80

    if is_speech is not None and ti > 0:
        prev = bool(is_speech[ti - 1])
        curr = bool(is_speech[ti])
        if prev and not curr:
            return 0.7
        if not prev and curr:
            return 0.5

    if hist_diff is not None and float(hist_diff[ti]) > 0.3:
        return 0.4

    return 0.2


# Labels that, when present on either side of a boundary, indicate the transition is NOT a
# genuine content chapter break and should not receive the transition bonus.
#   "ad"         — adjacent to an in-video ad read (poor UX to insert external ad there)
#   "filler"     — padding / B-roll between content blocks, not a chapter boundary
#   "dead_air"   — silent/blank frames, not a chapter boundary
#   "transition" — brief animated transition within content, not a chapter boundary
_NON_CHAPTER_LABELS = frozenset({"ad", "filler", "dead_air", "transition"})


def _transition_bonus(t: float, segments: List[Segment]) -> float:
    """
    1.0 when the segment label changes at boundary t AND neither side is a non-chapter label
    (e.g. core_content → holding_screen, recap → core_content, intro → core_content).
    0.0 when the same label continues on both sides, or when either side is in
    _NON_CHAPTER_LABELS (ad, filler, dead_air, transition).
    0.5 for edge cases where one side is unknown.
    """
    left  = next((s.label for s in segments if abs(s.end_sec   - t) < 0.5), None)
    right = next((s.label for s in segments if abs(s.start_sec - t) < 0.5), None)
    if left is None or right is None:
        return 0.5
    if left == right:
        return 0.0
    if left in _NON_CHAPTER_LABELS or right in _NON_CHAPTER_LABELS:
        return 0.0
    return 1.0


_STEP_HALF_WIN      = 30  # half-window for cut-density step calculation (seconds)
_MIN_BURST_DURATION = 20  # ignore burst regions shorter than this (content transitions)


def _find_burst_starts(grid: Optional[Dict[str, np.ndarray]]) -> List[float]:
    """
    Return one timestamp per contiguous burst region (hc_burst_density ≥ threshold).

    Within each burst region, identifies the second t where the cut-density STEP UP
    is largest:  forward_density[t:t+30s]  −  backward_density[t-30s:t].
    This locates the actual transition from sparse-cut content into the dense-cut
    ad block, even when the burst region extends seconds or minutes before the ad
    due to the forward-looking nature of hc_burst_density.

    The returned timestamp is the hard cut nearest to that step-up point.
    Falls back to the first hard cut in the region when no hard cuts are found nearby.
    """
    if grid is None:
        return []
    burst_density = grid.get("hc_burst_density")
    is_hard_cut   = grid.get("is_hard_cut")
    if burst_density is None or is_hard_cut is None:
        return []

    T = len(burst_density)
    hc_int = is_hard_cut.astype(np.int32)
    hc_cs  = np.concatenate([[0], np.cumsum(hc_int)])  # cumulative sum for O(1) range queries

    in_burst = np.asarray(burst_density) >= BURST_DENSITY_THRESHOLD
    results: List[float] = []
    i = 0
    while i < T:
        if in_burst[i]:
            j = i
            while j < T and in_burst[j]:
                j += 1

            # Skip very short burst regions — these are content transitions, not ad blocks.
            if j - i < _MIN_BURST_DURATION:
                i = j
                continue

            # For each second t in [i, j) compute the cut-density step:
            #   step[t] = (cuts in [t, t+W)) / W  −  (cuts in [t-W, t)) / W
            t_arr   = np.arange(i, j)
            fwd_end = np.minimum(t_arr + _STEP_HALF_WIN, T)
            bwd_sta = np.maximum(t_arr - _STEP_HALF_WIN, 0)
            fwd_n   = np.maximum(fwd_end - t_arr, 1)
            bwd_n   = np.maximum(t_arr - bwd_sta, 1)
            fwd_d   = (hc_cs[fwd_end] - hc_cs[t_arr]) / fwd_n
            bwd_d   = (hc_cs[t_arr] - hc_cs[bwd_sta]) / bwd_n
            steps   = fwd_d - bwd_d

            best_local = int(i + int(np.argmax(steps)))

            # Find the nearest hard cut to best_local (search ±5 s inside the region).
            best_k: Optional[int] = None
            for radius in range(0, 6):
                for delta in ([0] if radius == 0 else [radius, -radius]):
                    k = best_local + delta
                    if i <= k < j and bool(is_hard_cut[k]):
                        best_k = k
                        break
                if best_k is not None:
                    break

            if best_k is None:
                # Fallback: first hard cut in the burst region.
                for k in range(i, j):
                    if bool(is_hard_cut[k]):
                        best_k = k
                        break

            if best_k is not None:
                results.append(float(best_k))
            i = j
        else:
            i += 1
    return results


def _find_silence_candidates(grid: Optional[Dict[str, np.ndarray]]) -> List[float]:
    """
    Return timestamps where speech transitions to non-speech for ≥ SILENCE_MIN_SEC seconds.

    A candidate is emitted at the FIRST silent second of each qualifying run, provided
    that the preceding second was speech.  Only runs preceded by speech are returned
    (silence→silence continuations and leading silence are ignored).

    These natural pauses in spoken content are clean ad splice points.  They are
    especially valuable when visual burst evidence is absent (e.g. a static-screen
    interstitial or a low-cut-density ad).
    """
    if grid is None:
        return []
    speech = grid.get("is_speech")
    if speech is None:
        return []
    T  = len(speech)
    sp = np.asarray(speech, dtype=bool)
    results: List[float] = []
    i = 0
    while i < T:
        if not sp[i]:
            j = i
            while j < T and not sp[j]:
                j += 1
            if j - i >= SILENCE_MIN_SEC and i > 0 and sp[i - 1]:
                results.append(float(i))
            i = j
        else:
            i += 1
    return results



def _sponsor_penalty(t: float, segments: List[Segment]) -> float:
    """
    Penalty [0, 1] for being close to an in-video ad (host sponsor read) segment.
    Returns 0 when min_dist >= SPONSOR_EXCLUSION_SEC, rising to 1 at 0 s distance.
    """
    ad_segs = [s for s in segments if s.label == "ad"]
    if not ad_segs:
        return 0.0
    min_dist = min(
        min(abs(t - s.start_sec), abs(t - s.end_sec))
        for s in ad_segs
    )
    return max(0.0, 1.0 - min_dist / SPONSOR_EXCLUSION_SEC)


# ── Main scoring entry point ───────────────────────────────────────────────────

def score_candidates(
    segments: List[Segment],
    natural_break_candidates: List[float],
    grid: Optional[Dict[str, np.ndarray]],
    duration_sec: float,
    n_ads: int = 3,
) -> List[AdInsertionCandidate]:
    """
    Score segment boundaries and greedily select the top-N insertion points.

    Candidate pool: segment start times (excluding 0). Falls back to
    natural_break_candidates when no segment boundaries are available.

    Pre-filter: candidates within MIN_AD_SPACING_SEC of the video end are removed so
    that no ad break is inserted in the final minutes.

    Each round:
      1. Compute walls = [0, previously-selected, duration_sec].
      2. For each remaining candidate, measure left/right content duration.
      3. Skip if either side < MIN_BLOCK_SEC or candidate is within MIN_AD_SPACING_SEC
         of an already-selected point.
      4. Score = W_SIGNAL*signal + W_TRANSITION*transition_bonus - W_SPONSOR*penalty.
      5. Select highest scorer, update walls, repeat.

    Returns candidates sorted by timestamp (not score).
    """
    if n_ads <= 0:
        return []

    # Prefer segment boundaries; fall back to raw candidates for unit-test scenarios.
    seg_starts = sorted({s.start_sec for s in segments if s.start_sec > 0})
    candidates = seg_starts if seg_starts else sorted(set(natural_break_candidates))
    if not candidates:
        return []

    # Pre-filter: require at least MIN_BLOCK_SEC of content before the video ends so
    # that no insertion point is jammed into the final segment.  MIN_AD_SPACING_SEC is
    # a between-candidate spacing constraint enforced in the selection loop below; using
    # it here as an end-buffer over-filters short/dense videos (e.g. a 10-min video
    # with a valid insertion point 3 min from the end would be wrongly removed).
    candidates = [t for t in candidates if t + MIN_BLOCK_SEC <= duration_sec]
    if not candidates:
        return []

    # Inject visual burst starts: first hard cut of each high-density edit window.
    # These bypass the segment-boundary requirement and the sponsor penalty because
    # they are direct visual evidence of an inserted ad block.
    burst_set: set = set(
        t for t in _find_burst_starts(grid)
        if MIN_BLOCK_SEC <= t <= duration_sec - MIN_BLOCK_SEC
    )

    # Inject audio silence starts: first second of each ≥SILENCE_MIN_SEC non-speech
    # run that follows speech.  Complements visual burst when no hard-cut evidence exists.
    silence_set: set = set(
        t for t in _find_silence_candidates(grid)
        if MIN_BLOCK_SEC <= t <= duration_sec - MIN_BLOCK_SEC
    )

    # Identify silence candidates corroborated by a large luminance jump within ±10 s.
    # A silence candidate + nearby extreme brightness change = audio+visual confirmation
    # of a content→ad boundary → receive burst-level scoring priority.
    _hc_arr = grid.get("is_hard_cut") if grid is not None else None
    _ld_arr = grid.get("lum_context_delta") if grid is not None else None
    _T_grid = len(_hc_arr) if _hc_arr is not None else 0

    def _has_lum_corroboration(ts: float) -> bool:
        lo = max(0, int(ts) - 10)
        hi = min(_T_grid, int(ts) + 11)
        for ti in range(lo, hi):
            if bool(_hc_arr[ti]) and abs(float(_ld_arr[ti])) >= LUM_JUMP_CORROBORATION_THRESHOLD:
                return True
        return False

    silence_corroborated: set = (
        {t for t in silence_set if _has_lum_corroboration(t)}
        if _hc_arr is not None and _ld_arr is not None else set()
    )

    # High-priority set: burst candidates + silence candidates with visual corroboration.
    high_priority: set = burst_set | silence_corroborated

    all_candidates = sorted(set(candidates) | high_priority | silence_set)

    # Confirmed hard-cut candidates: segment boundaries with max signal (burst + large lum
    # jump).  These are visually the strongest content→ad boundaries in the video and should
    # compete with burst candidates even when there is no distinct audio silence or CLIP
    # ad-label nearby.  Elevating to high_priority gives trans=1.0, breaking ties against
    # weaker candidates that happen to have a good label transition.
    if grid is not None:
        hardcut_confirmed: set = {
            t for t in all_candidates
            if t not in high_priority
            and _signal_strength(t, grid) >= 1.0
        }
        high_priority = high_priority | hardcut_confirmed
    else:
        hardcut_confirmed = set()

    # Ad-precursor candidates: any candidate within _AD_PRECURSOR_SEC *before* a
    # CLIP/rule-detected "ad" segment start.  The fusion segmenter may have placed a
    # clean boundary just before the ad label begins; that boundary is a perfect splice
    # point but its transition_bonus is often 0 (same label on both sides until the ad
    # segment is resolved).  Elevating it to high_priority gives trans=1.0 so it can
    # compete with burst/corroborated candidates.
    #
    # Fix 2: only anchor off "ad" segments that are either rule-confirmed OR long enough
    # (≥ _AD_PRECURSOR_MIN_SEG_DUR) to be a genuine detected ad.  Short CLIP-only blips
    # (2–7 s) should not create precursor elevation — they are likely false positives.
    _AD_PRECURSOR_SEC = 15.0
    _AD_PRECURSOR_MIN_SEG_DUR = 8.0
    _ad_starts = frozenset(
        s.start_sec for s in segments
        if s.label == "ad"
        and (
            len(s.evidence.triggered_rules) > 0
            or (s.end_sec - s.start_sec) >= _AD_PRECURSOR_MIN_SEG_DUR
        )
    )
    ad_precursor_set: set = {
        t for t in all_candidates
        if t not in high_priority
        and any(0 < a_t - t <= _AD_PRECURSOR_SEC for a_t in _ad_starts)
    }
    high_priority = high_priority | ad_precursor_set

    # Pre-compute per-candidate scores (one pass through grid).
    signal_scores: Dict[float, float] = {}
    trans_scores:  Dict[float, float] = {}
    for t in all_candidates:
        if t in silence_corroborated:
            # Audio silence + visual luminance jump: both modalities confirm → max confidence.
            signal_scores[t] = 1.0
            trans_scores[t]  = 1.0
        elif t in high_priority:
            # Burst boundary or ad-precursor: visual/context evidence confirmed → trans=1.0.
            signal_scores[t] = _signal_strength(t, grid) if grid is not None else 0.5
            trans_scores[t]  = 1.0
        else:
            signal_scores[t] = _signal_strength(t, grid) if grid is not None else 0.5
            tb = _transition_bonus(t, segments)
            if t in silence_set:
                # Audio evidence of a genuine pause: floor transition bonus at 0.5.
                tb = max(tb, 0.5)
            trans_scores[t] = tb

    selected: List[float] = []
    result:   List[AdInsertionCandidate] = []
    remaining = list(all_candidates)

    for _round in range(n_ads):
        if not remaining:
            break

        walls = sorted([0.0] + selected + [duration_sec])
        best_t:    Optional[float]    = None
        best_score = -1.0
        best_info: Dict[str, float]   = {}

        for t in remaining:
            prev_wall = max((w for w in walls if w <= t), default=0.0)
            next_wall = min((w for w in walls if w > t), default=duration_sec)
            left_dur  = t - prev_wall
            right_dur = next_wall - t

            if left_dur < MIN_BLOCK_SEC or right_dur < MIN_BLOCK_SEC:
                continue
            if selected and min(abs(t - s) for s in selected) < MIN_AD_SPACING_SEC:
                continue

            sig   = signal_scores[t]
            trans = trans_scores[t]
            # High-priority candidates bypass sponsor penalty: they ARE the ad boundary.
            pen   = 0.0 if t in high_priority else _sponsor_penalty(t, segments)
            composite = W_SIGNAL * sig + W_TRANSITION * trans - W_SPONSOR * pen

            if composite > best_score:
                best_score = composite
                best_t = t
                best_info = {
                    "sig": sig, "trans": trans,
                    "left": left_dur, "right": right_dur,
                }

        if best_t is None:
            break
        # For the 4th+ slot, require stronger evidence so weak segment boundaries
        # don't fill slots when the video has fewer ads than n_ads.
        if _round >= 3 and best_score < MIN_COMPOSITE_SCORE_EXTRA:
            break

        selected.append(best_t)
        remaining = [c for c in remaining if c != best_t]
        result.append(AdInsertionCandidate(
            timestamp_sec=best_t,
            score=round(max(0.0, best_score), 3),
            signal_strength=round(best_info["sig"], 3),
            transition_score=round(best_info["trans"], 3),
            left_block_sec=round(best_info["left"], 1),
            right_block_sec=round(best_info["right"], 1),
        ))

    return sorted(result, key=lambda c: c.timestamp_sec)
