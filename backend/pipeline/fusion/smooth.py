"""
Phase 3 Step 5: Temporal smoothing.

Fix 1 – merge_adjacent_same_label respects has_hard_cut_before:
    Adjacent segments with the same label are NOT merged when the right segment's
    left boundary was a visual hard cut. This preserves scene-change boundaries
    inside long same-label runs (e.g., a hard cut mid-way through a core_content
    block) so downstream ad-placement can use them.

Fix 3 – split_oversized_segments:
    Any segment longer than MAX_SEG_DURATION gets split at the strongest internal
    hard cut from the pre-smooth boundary list. Belt-and-suspenders for Fix 1 when
    the hard-cut signal is present but was dropped before classification.

Fix 4 – propagate_context:
    Short filler segments (< CONTEXT_FILLER_MAX_SEC) flanked on both sides by
    core_content are re-labeled core_content. These are pauses/b-roll inside
    an ongoing content block rather than genuine padding.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

import numpy as np

from backend.pipeline.schemas import Segment, SegmentEvidence


# ── Tunable constants ──────────────────────────────────────────────────────────
MIN_SEGMENT_DURATION = 2.0   # seconds — absorb shorter segments into neighbors
MAX_SEG_DURATION = 300.0     # seconds — force-split segments longer than this
CONTEXT_FILLER_MAX_SEC = 30  # seconds — max filler duration eligible for relabeling

AD_NOISE_GAP_TOLERANCE = 15.0   # seconds — gaps ≤ this link "ad" segs into one cluster
AD_NOISE_MIN_SPAN      = 20.0   # seconds — clusters shorter than this (with no rule hits)
                                 #           are CLIP noise → relabeled filler


# ── Internal helpers ───────────────────────────────────────────────────────────

def _merge_two(a: Segment, b: Segment, new_id: int) -> Segment:
    """Merge b into a (a comes first in time)."""
    return Segment(
        segment_id=new_id,
        start_sec=a.start_sec,
        end_sec=b.end_sec,
        label=a.label,
        confidence=max(a.confidence, b.confidence),
        evidence=SegmentEvidence(
            visual_score=max(a.evidence.visual_score, b.evidence.visual_score),
            audio_score=max(a.evidence.audio_score, b.evidence.audio_score),
            text_score=max(a.evidence.text_score, b.evidence.text_score),
            triggered_rules=list(set(a.evidence.triggered_rules + b.evidence.triggered_rules)),
        ),
        summary=a.summary if len(a.summary) >= len(b.summary) else b.summary,
        # Preserve the left segment's hard-cut flag (boundary is on the left edge of a).
        has_hard_cut_before=a.has_hard_cut_before,
        user_corrected=False,
    )


def _make_split(orig: Segment, start: float, end: float, new_id: int, hard_cut_before: bool) -> Segment:
    """Slice a segment to [start, end) preserving all other fields."""
    return Segment(
        segment_id=new_id,
        start_sec=start,
        end_sec=end,
        label=orig.label,
        confidence=orig.confidence,
        evidence=orig.evidence,
        summary=orig.summary,
        has_hard_cut_before=hard_cut_before,
        user_corrected=False,
    )


# ── Fix 1: merge respects hard-cut boundaries ─────────────────────────────────

def merge_adjacent_same_label(segments: List[Segment]) -> List[Segment]:
    """
    Collapse consecutive segments that share a label into one — UNLESS the right
    segment's has_hard_cut_before flag is set, which means a visual hard cut sits
    on that boundary and should be preserved.
    """
    if not segments:
        return []
    out = [segments[0]]
    for seg in segments[1:]:
        same_label = seg.label == out[-1].label
        # Block the merge if this boundary came from a hard cut.
        protected = seg.has_hard_cut_before
        if same_label and not protected:
            out[-1] = _merge_two(out[-1], seg, out[-1].segment_id)
        else:
            out.append(seg)
    return out


# ── Fix 3: split segments that are too long ────────────────────────────────────

def split_oversized_segments(
    segments: List[Segment],
    raw_boundaries: List[int],
    grid: Dict[str, np.ndarray],
) -> List[Segment]:
    """
    Split any segment longer than MAX_SEG_DURATION at the best internal hard-cut
    boundary from the pre-smooth candidate list.

    'Best' = the internal boundary whose second has the highest hist_diff value.
    If no hist_diff column is available, pick the boundary closest to the midpoint.
    """
    if not raw_boundaries or not grid:
        return segments

    hist = grid.get("hist_diff", None)
    T = len(grid["is_speech"])
    raw_set = set(raw_boundaries)
    out: List[Segment] = []

    for seg in segments:
        duration = seg.end_sec - seg.start_sec
        if duration <= MAX_SEG_DURATION:
            out.append(seg)
            continue

        # Find internal candidates from the pre-smooth boundary list.
        internal = [b for b in raw_set if seg.start_sec < b < seg.end_sec]
        if not internal:
            out.append(seg)
            continue

        # Score each candidate by hist_diff, falling back to proximity to midpoint.
        mid = (seg.start_sec + seg.end_sec) / 2.0
        if hist is not None:
            best = max(internal, key=lambda b: float(hist[min(int(b), T - 1)]))
        else:
            best = min(internal, key=lambda b: abs(b - mid))

        left = _make_split(seg, seg.start_sec, float(best), seg.segment_id, seg.has_hard_cut_before)
        right = _make_split(seg, float(best), seg.end_sec, seg.segment_id + 1, True)
        out.extend([left, right])

    return out


# ── Standard absorb / renumber ────────────────────────────────────────────────

def absorb_short_segments(segments: List[Segment]) -> List[Segment]:
    """Absorb segments shorter than MIN_SEGMENT_DURATION into their stronger neighbor."""
    if len(segments) <= 1:
        return segments

    out = list(segments)
    changed = True
    while changed:
        changed = False
        for i, seg in enumerate(out):
            if seg.end_sec - seg.start_sec < MIN_SEGMENT_DURATION:
                left = out[i - 1] if i > 0 else None
                right = out[i + 1] if i + 1 < len(out) else None

                if left is None and right is None:
                    continue
                elif left is None:
                    out[i + 1] = Segment(
                        segment_id=right.segment_id,
                        start_sec=seg.start_sec,
                        end_sec=right.end_sec,
                        label=right.label,
                        confidence=right.confidence,
                        evidence=right.evidence,
                        summary=right.summary,
                        has_hard_cut_before=seg.has_hard_cut_before,
                        user_corrected=False,
                    )
                    out.pop(i)
                elif right is None:
                    out[i - 1] = Segment(
                        segment_id=left.segment_id,
                        start_sec=left.start_sec,
                        end_sec=seg.end_sec,
                        label=left.label,
                        confidence=left.confidence,
                        evidence=left.evidence,
                        summary=left.summary,
                        has_hard_cut_before=left.has_hard_cut_before,
                        user_corrected=False,
                    )
                    out.pop(i)
                else:
                    target = left if left.confidence >= right.confidence else right
                    if target is left:
                        out[i - 1] = Segment(
                            segment_id=left.segment_id,
                            start_sec=left.start_sec,
                            end_sec=seg.end_sec,
                            label=left.label,
                            confidence=left.confidence,
                            evidence=left.evidence,
                            summary=left.summary,
                            has_hard_cut_before=left.has_hard_cut_before,
                            user_corrected=False,
                        )
                        out.pop(i)
                    else:
                        out[i + 1] = Segment(
                            segment_id=right.segment_id,
                            start_sec=seg.start_sec,
                            end_sec=right.end_sec,
                            label=right.label,
                            confidence=right.confidence,
                            evidence=right.evidence,
                            summary=right.summary,
                            has_hard_cut_before=seg.has_hard_cut_before,
                            user_corrected=False,
                        )
                        out.pop(i)
                changed = True
                break
    return out


# ── Suppress isolated CLIP "ad" false positives ───────────────────────────────

def suppress_isolated_ad_noise(segments: List[Segment]) -> List[Segment]:
    """
    CLIP occasionally mis-labels short content bursts as 'ad'.

    Algorithm:
      1. Group 'ad' segments into clusters: two 'ad' segments belong to the same
         cluster when the gap between them (across any intervening labels) is
         ≤ AD_NOISE_GAP_TOLERANCE seconds.
      2. Compute span = last_end − first_start for each cluster.
      3. If span < AD_NOISE_MIN_SPAN AND every segment in the cluster has an
         empty triggered_rules list (i.e., CLIP-only, no keyword/rule confirmation),
         relabel the entire cluster to 'filler'.

    This removes CLIP noise (e.g. a 12-second burst of 'ad'-looking educational
    content) without touching genuine ad blocks (which span ≥ 20 s) or any ad
    confirmed by a hard rule.
    """
    if not segments:
        return segments

    out = list(segments)
    n = len(out)
    visited = [False] * n
    i = 0
    while i < n:
        if out[i].label != "ad" or visited[i]:
            i += 1
            continue

        # Grow the cluster forward: skip non-"ad" segments while the gap is small.
        cluster: List[int] = [i]
        visited[i] = True
        j = i + 1
        while j < n:
            # Find the next "ad" segment from position j.
            k = j
            while k < n and out[k].label != "ad":
                k += 1
            if k >= n:
                break
            gap = out[k].start_sec - out[cluster[-1]].end_sec
            if gap <= AD_NOISE_GAP_TOLERANCE:
                cluster.append(k)
                visited[k] = True
                j = k + 1
            else:
                break

        span = out[cluster[-1]].end_sec - out[cluster[0]].start_sec
        rule_free = all(len(out[ci].evidence.triggered_rules) == 0 for ci in cluster)

        if span < AD_NOISE_MIN_SPAN and rule_free:
            for ci in cluster:
                s = out[ci]
                out[ci] = Segment(
                    segment_id=s.segment_id,
                    start_sec=s.start_sec,
                    end_sec=s.end_sec,
                    label="filler",
                    confidence=s.confidence,
                    evidence=s.evidence,
                    summary=s.summary,
                    has_hard_cut_before=s.has_hard_cut_before,
                    user_corrected=False,
                )

        i = cluster[-1] + 1

    return out


# ── Fix 4: context-propagation pass ──────────────────────────────────────────

def propagate_context(segments: List[Segment]) -> List[Segment]:
    """
    Re-label short filler segments that are fully surrounded by core_content on
    both sides. These are pauses or b-roll within an ongoing content block, not
    genuine filler/padding.
    """
    if len(segments) < 3:
        return segments

    out = list(segments)
    changed = True
    while changed:
        changed = False
        for i in range(1, len(out) - 1):
            seg = out[i]
            if seg.label != "filler":
                continue
            duration = seg.end_sec - seg.start_sec
            if duration > CONTEXT_FILLER_MAX_SEC:
                continue
            left_label = out[i - 1].label
            right_label = out[i + 1].label
            if left_label == "core_content" and right_label == "core_content":
                out[i] = Segment(
                    segment_id=seg.segment_id,
                    start_sec=seg.start_sec,
                    end_sec=seg.end_sec,
                    label="core_content",
                    confidence=round((out[i - 1].confidence + out[i + 1].confidence) / 2, 3),
                    evidence=seg.evidence,
                    summary=seg.summary,
                    has_hard_cut_before=seg.has_hard_cut_before,
                    user_corrected=False,
                )
                changed = True
    return out


def renumber(segments: List[Segment]) -> List[Segment]:
    """Reset segment_id to 0..N-1 in order."""
    return [
        Segment(
            segment_id=i,
            start_sec=s.start_sec,
            end_sec=s.end_sec,
            label=s.label,
            confidence=s.confidence,
            evidence=s.evidence,
            summary=s.summary,
            has_hard_cut_before=s.has_hard_cut_before,
            user_corrected=s.user_corrected,
        )
        for i, s in enumerate(segments)
    ]


# ── Fix 5: promote stranded break candidates ──────────────────────────────────

PROMOTE_THRESHOLD = 28.0  # seconds — promote a candidate if no seg boundary is closer


def promote_stranded_candidates(
    segments: List[Segment],
    raw_boundaries: List[int],
) -> List[Segment]:
    """
    For each raw boundary that has no segment boundary within PROMOTE_THRESHOLD
    seconds, split the containing segment at that timestamp.

    Converts speech-transition and other non-hard-cut break candidates that
    survive in natural_break_candidates but were erased from segment boundaries
    by same-label merging.

    Safe guarantees:
    - Does NOT modify raw_boundaries, so natural_break_candidates is unchanged.
    - Only fires when nearest seg boundary > PROMOTE_THRESHOLD (28 s). All
      currently-passing insertions have candidates within 20 s of a seg boundary.
    - Split halves inherit the parent label — no reclassification occurs.
    """
    if not raw_boundaries:
        return segments

    out = list(segments)
    # Keep a live set of boundary positions so each split is visible to later candidates.
    seg_bounds = sorted({s.start_sec for s in out} | {s.end_sec for s in out})

    for cand in sorted(raw_boundaries):
        cand_f = float(cand)
        if min(abs(b - cand_f) for b in seg_bounds) <= PROMOTE_THRESHOLD:
            continue  # already close enough to a segment boundary

        # Find the segment that contains this candidate.
        for i, seg in enumerate(out):
            if seg.start_sec < cand_f < seg.end_sec:
                left = _make_split(seg, seg.start_sec, cand_f,
                                   seg.segment_id, seg.has_hard_cut_before)
                right = _make_split(seg, cand_f, seg.end_sec,
                                    seg.segment_id + 1, False)
                out = out[:i] + [left, right] + out[i + 1:]
                seg_bounds = sorted(set(seg_bounds) | {cand_f})
                break

    return out


# ── Pipeline entry point ───────────────────────────────────────────────────────

def smooth_pipeline(
    segments: List[Segment],
    raw_boundaries: Optional[List[int]] = None,
    grid: Optional[Dict[str, np.ndarray]] = None,
) -> List[Segment]:
    """
    Full smooth pass:
      1. merge_adjacent_same_label       (Fix 1: respects hard-cut flags)
      2. split_oversized_segments        (Fix 3: belt-and-suspenders for long segs)
      3. absorb_short_segments
      4. propagate_context               (Fix 4: filler inside core_content → core)
      5. merge_adjacent_same_label       (clean up after relabeling)
      6. suppress_isolated_ad_noise      (remove short CLIP-only "ad" clusters < 20 s)
      7. merge_adjacent_same_label       (clean up after suppression)
      8. promote_stranded_candidates     (Fix 5: speech-flip boundaries > 28 s from
                                          nearest seg boundary become seg boundaries)
      9. renumber
    """
    s = merge_adjacent_same_label(segments)
    if raw_boundaries and grid is not None:
        s = split_oversized_segments(s, raw_boundaries, grid)
    s = absorb_short_segments(s)
    s = propagate_context(s)
    s = merge_adjacent_same_label(s)
    s = suppress_isolated_ad_noise(s)
    s = merge_adjacent_same_label(s)
    if raw_boundaries:
        s = promote_stranded_candidates(s, raw_boundaries)
    s = renumber(s)
    return s
