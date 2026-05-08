"""
Phase 3 Step 5: Temporal smoothing.

Original passes:
  - Merge adjacent segments with the same label.
  - Absorb very short (< MIN_SEGMENT_DURATION) isolated segments into their neighbors.
  - Bridge two sponsorship blocks across a short non-sponsorship gap.

Path B (Roger F1/F3/F4 cherry-picked, 2026-05-06):
  - Fix 1 — merge_adjacent_same_label respects Segment.has_hard_cut_before.
    Adjacent same-label segments are NOT merged when the right segment's left
    boundary was a visual hard cut. Preserves real scene boundaries inside
    long same-label runs.
  - Fix 3 — split_oversized_segments. Any segment longer than MAX_SEG_DURATION
    is split at the strongest internal hard-cut from the pre-smooth boundary
    list. Belt-and-suspenders for the漫灌 case.
  - Fix 4 — propagate_context. Short filler segments fully surrounded by
    core_content are re-labeled core_content (lecture pauses / b-roll, not
    genuine padding).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from backend.pipeline.schemas import Segment, SegmentEvidence


# ── Tunable constants ────────────────────────────────────────────────────────
MIN_SEGMENT_DURATION = 2.0          # seconds — absorb shorter into stronger neighbour
MAX_SEG_DURATION = 300.0            # F3 — force-split segments longer than this
CONTEXT_FILLER_MAX_SEC = 30         # F4 — max filler width eligible for relabel
SPONSORSHIP_BRIDGE_MAX_GAP = 6.0    # bridge two sponsorship blocks across a gap
SPONSORSHIP_BRIDGE_CORE_AUDIO_VETO = 0.5  # mid==core_content with audio_score >= this blocks bridging

# Sponsorship FP suppression (test_009 vlog-mode tightening, 2026-05-08).
#
# Method 1 — clear_sponsorship_islands: short, sub-0.85-conf sponsorship blocks
# fully surrounded by long core_content runs are vlog-mode rule mis-fires
# (rule_ad_block / vlog_speech_burst on dialogue-heavy B-roll). 10s ceiling
# protects real GT ads (test_009 ground-truth ads are 30/31/55s).
SPONSORSHIP_ISLAND_MAX_SEC = 10.0
SPONSORSHIP_ISLAND_MIN_NEIGHBOR_SEC = 30.0
SPONSORSHIP_ISLAND_MAX_CONFIDENCE = 0.85

# Method 2 — filter_short_low_conf_sponsorship: catch-all for tiny low-conf
# sponsorship that island-clear missed (e.g. neighbour is intro/holding_screen
# instead of core_content). 0.50 floor is below every TP block in test_009
# (min TP conf = 0.75) but above the typical FP cluster (0.24–0.48).
MIN_SPONSORSHIP_DURATION = 8.0
MIN_SPONSORSHIP_CONFIDENCE = 0.50


# ── Internal helpers ─────────────────────────────────────────────────────────

def _merge_two(a: Segment, b: Segment, new_id: int) -> Segment:
    """Merge b into a (a comes first in time). Left boundary stays at a's left."""
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
        # Preserve a's hard-cut flag — the merged segment's left boundary IS a's left.
        has_hard_cut_before=a.has_hard_cut_before,
        user_corrected=False,
    )


def _make_split(orig: Segment, start: float, end: float, new_id: int,
                hard_cut_before: bool) -> Segment:
    """Slice `orig` to [start, end) preserving label / confidence / evidence."""
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


# ── Fix 1: merge respects hard-cut boundaries ────────────────────────────────

def merge_adjacent_same_label(segments: List[Segment]) -> List[Segment]:
    """Collapse consecutive segments that share a label, except when the right
    segment's left boundary was a visual hard cut (preserve scene boundary)."""
    if not segments:
        return []
    out = [segments[0]]
    for seg in segments[1:]:
        same_label = seg.label == out[-1].label
        protected = seg.has_hard_cut_before
        if same_label and not protected:
            out[-1] = _merge_two(out[-1], seg, out[-1].segment_id)
        else:
            out.append(seg)
    return out


# ── Fix 3: split segments that are too long ──────────────────────────────────

def split_oversized_segments(
    segments: List[Segment],
    raw_boundaries: List[int],
    grid: Dict[str, np.ndarray],
) -> List[Segment]:
    """Split any segment > MAX_SEG_DURATION at its strongest internal raw boundary.

    'Strongest' = highest hist_diff at that second. Falls back to the boundary
    nearest the midpoint when hist_diff is unavailable.
    """
    if not raw_boundaries or not grid:
        return segments

    hist = grid.get("hist_diff", None)
    T = len(grid["is_speech"])
    raw_set = set(raw_boundaries)
    out: List[Segment] = []

    for seg in segments:
        if seg.end_sec - seg.start_sec <= MAX_SEG_DURATION:
            out.append(seg)
            continue

        internal = [b for b in raw_set if seg.start_sec < b < seg.end_sec]
        if not internal:
            out.append(seg)
            continue

        if hist is not None:
            best = max(internal, key=lambda b: float(hist[min(int(b), T - 1)]))
        else:
            mid = (seg.start_sec + seg.end_sec) / 2.0
            best = min(internal, key=lambda b: abs(b - mid))

        left = _make_split(seg, seg.start_sec, float(best),
                           seg.segment_id, seg.has_hard_cut_before)
        right = _make_split(seg, float(best), seg.end_sec,
                            seg.segment_id + 1, True)  # the split point is a forced boundary
        out.extend([left, right])

    return out


# ── Standard absorb ──────────────────────────────────────────────────────────

def absorb_short_segments(segments: List[Segment]) -> List[Segment]:
    """Absorb segments shorter than MIN_SEGMENT_DURATION into their stronger neighbour."""
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
                    # Extend right's left edge backward to seg.start_sec.
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


# ── Fix 4: filler-in-core context propagation ────────────────────────────────

def propagate_context(segments: List[Segment]) -> List[Segment]:
    """Re-label short filler segments that are flanked on BOTH sides by
    core_content. These are pauses or b-roll inside ongoing content, not
    genuine filler."""
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
            if seg.end_sec - seg.start_sec > CONTEXT_FILLER_MAX_SEC:
                continue
            if out[i - 1].label == "core_content" and out[i + 1].label == "core_content":
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


# ── Sponsorship FP suppression (test_009 vlog-mode tightening) ───────────────

def _relabel_to_core(seg: Segment) -> Segment:
    return Segment(
        segment_id=seg.segment_id,
        start_sec=seg.start_sec,
        end_sec=seg.end_sec,
        label="core_content",
        confidence=seg.confidence,
        evidence=seg.evidence,
        summary=seg.summary,
        has_hard_cut_before=seg.has_hard_cut_before,
        user_corrected=seg.user_corrected,
    )


def clear_sponsorship_islands(segments: List[Segment]) -> List[Segment]:
    """Method 1: relabel short low-conf sponsorship surrounded by long core_content.

    Mirrors propagate_context but for sponsorship. A vlog-mode rule_ad_block /
    vlog_speech_burst hit on dialogue-heavy B-roll produces a 3-10s sponsorship
    island in the middle of a core_content run. test_009 diagnosis (2026-05-08)
    showed every such island is FP; real ad blocks are 26+s.
    """
    if len(segments) < 3:
        return list(segments)

    out = list(segments)
    for i in range(1, len(out) - 1):
        seg = out[i]
        if seg.label != "sponsorship":
            continue
        if seg.end_sec - seg.start_sec > SPONSORSHIP_ISLAND_MAX_SEC:
            continue
        if seg.confidence > SPONSORSHIP_ISLAND_MAX_CONFIDENCE:
            continue
        L, R = out[i - 1], out[i + 1]
        if L.label != "core_content" or R.label != "core_content":
            continue
        if (L.end_sec - L.start_sec) < SPONSORSHIP_ISLAND_MIN_NEIGHBOR_SEC:
            continue
        if (R.end_sec - R.start_sec) < SPONSORSHIP_ISLAND_MIN_NEIGHBOR_SEC:
            continue
        out[i] = _relabel_to_core(seg)
    return out


def filter_short_low_conf_sponsorship(segments: List[Segment]) -> List[Segment]:
    """Method 2: relabel sponsorship segments shorter than MIN_SPONSORSHIP_DURATION
    AND with confidence below MIN_SPONSORSHIP_CONFIDENCE.

    Catch-all for FP that island-clear missed (e.g. neighbour is intro or
    holding_screen). Both gates must trip — 7s @ 0.78 conf is kept, but 3s @
    0.30 is dropped.
    """
    out = []
    for seg in segments:
        if (
            seg.label == "sponsorship"
            and (seg.end_sec - seg.start_sec) < MIN_SPONSORSHIP_DURATION
            and seg.confidence < MIN_SPONSORSHIP_CONFIDENCE
        ):
            out.append(_relabel_to_core(seg))
        else:
            out.append(seg)
    return out


# ── Sponsorship bridging (existing) ──────────────────────────────────────────

def bridge_sponsorship_gaps(
    segments: List[Segment],
    max_gap_sec: float = SPONSORSHIP_BRIDGE_MAX_GAP,
) -> List[Segment]:
    """Fuse [sponsorship, X, sponsorship] when X is non-sponsorship and short.

    Real ad inserts often contain a mid-roll voiceover or stinger that breaks
    the quiet/music pattern. Without bridging these, ad blocks fragment.
    """
    if len(segments) < 3:
        return list(segments)

    out: List[Segment] = []
    i = 0
    while i < len(segments):
        mid = segments[i + 1] if i + 2 < len(segments) else None
        # Veto bridging when the mid segment shows continuous speech. A mid
        # segment with audio_score >= veto threshold is acoustically core
        # content even if a partial-coverage rule painted it sponsorship —
        # bridging across it would re-absorb a real return-to-content gap.
        mid_is_real_content = (
            mid is not None
            and mid.evidence.audio_score >= SPONSORSHIP_BRIDGE_CORE_AUDIO_VETO
        )
        if (
            i + 2 < len(segments)
            and segments[i].label == "sponsorship"
            and segments[i + 2].label == "sponsorship"
            and segments[i + 1].label != "sponsorship"
            and (segments[i + 1].end_sec - segments[i + 1].start_sec) <= max_gap_sec
            and not mid_is_real_content
        ):
            a, mid, b = segments[i], segments[i + 1], segments[i + 2]
            bridged_conf = min(a.confidence, b.confidence)
            triggered = list(set(
                a.evidence.triggered_rules
                + mid.evidence.triggered_rules
                + b.evidence.triggered_rules
                + ["sponsorship_bridge"]
            ))
            merged = Segment(
                segment_id=a.segment_id,
                start_sec=a.start_sec,
                end_sec=b.end_sec,
                label="sponsorship",
                confidence=bridged_conf,
                evidence=SegmentEvidence(
                    visual_score=max(a.evidence.visual_score, mid.evidence.visual_score, b.evidence.visual_score),
                    audio_score=max(a.evidence.audio_score, mid.evidence.audio_score, b.evidence.audio_score),
                    text_score=max(a.evidence.text_score, mid.evidence.text_score, b.evidence.text_score),
                    triggered_rules=triggered,
                ),
                summary=max([a.summary, mid.summary, b.summary], key=len),
                has_hard_cut_before=a.has_hard_cut_before,
                user_corrected=False,
            )
            out.append(merged)
            i += 3
        else:
            out.append(segments[i])
            i += 1
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


# ── Pipeline entry point ─────────────────────────────────────────────────────

def smooth_pipeline(
    segments: List[Segment],
    raw_boundaries: Optional[List[int]] = None,
    grid: Optional[Dict[str, np.ndarray]] = None,
) -> List[Segment]:
    """Run merge → split-oversized → absorb → propagate-context → merge →
    bridge sponsorship gaps → merge again → renumber.

    raw_boundaries / grid are required for Fix 3 (split_oversized_segments).
    Backwards-compatible: when they are None, F3 is skipped.
    """
    s = merge_adjacent_same_label(segments)
    if raw_boundaries and grid is not None:
        s = split_oversized_segments(s, raw_boundaries, grid)
    s = absorb_short_segments(s)
    s = propagate_context(s)
    s = merge_adjacent_same_label(s)
    # Skip sponsorship bridging in vlog mode: vlog content has rapid
    # alternation between speech bursts and silent B-roll, so any "short
    # non-sponsorship gap between two sponsorship segments" is more likely a
    # vlog return-to-content than a single ad block to be fused.
    from backend.pipeline.fusion.rules import _is_vlog_mode
    skip_bridge = grid is not None and _is_vlog_mode(grid)
    if not skip_bridge:
        while True:
            bridged = bridge_sponsorship_gaps(s)
            if len(bridged) == len(s):
                break
            s = merge_adjacent_same_label(bridged)
    # Sponsorship FP suppression (test_009): clear short low-conf islands first,
    # then catch any remaining tiny low-conf sponsorship, then re-merge.
    s = clear_sponsorship_islands(s)
    s = filter_short_low_conf_sponsorship(s)
    s = merge_adjacent_same_label(s)
    s = renumber(s)
    return s
