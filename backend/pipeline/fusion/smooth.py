"""
Phase 3 Step 5: Temporal smoothing.

- Merge adjacent segments with the same label.
- Absorb very short (< MIN_SEGMENT_DURATION) isolated segments into their neighbors.
"""
from __future__ import annotations
from typing import List
from backend.pipeline.schemas import Segment, SegmentEvidence


MIN_SEGMENT_DURATION = 2.0  # seconds
SPONSORSHIP_BRIDGE_MAX_GAP = 12.0  # max seconds of non-sponsorship between two sponsorship blocks to bridge


def _merge_two(a: Segment, b: Segment, new_id: int) -> Segment:
    """Merge segment b into a (a comes first)."""
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
        user_corrected=False,
    )


def merge_adjacent_same_label(segments: List[Segment]) -> List[Segment]:
    if not segments:
        return []
    out = [segments[0]]
    for seg in segments[1:]:
        if seg.label == out[-1].label:
            out[-1] = _merge_two(out[-1], seg, out[-1].segment_id)
        else:
            out.append(seg)
    return out


def absorb_short_segments(segments: List[Segment]) -> List[Segment]:
    """Absorb segments shorter than MIN_SEGMENT_DURATION into their neighbor with higher confidence."""
    if len(segments) <= 1:
        return segments
    
    out = list(segments)
    changed = True
    while changed:
        changed = False
        for i, seg in enumerate(out):
            if seg.end_sec - seg.start_sec < MIN_SEGMENT_DURATION:
                # Decide which neighbor to merge into
                left = out[i - 1] if i > 0 else None
                right = out[i + 1] if i + 1 < len(out) else None
                
                if left is None and right is None:
                    continue  # only segment, leave it
                elif left is None:
                    # Merge with right: extend right's start backward
                    out[i + 1] = Segment(
                        segment_id=right.segment_id,
                        start_sec=seg.start_sec,
                        end_sec=right.end_sec,
                        label=right.label,
                        confidence=right.confidence,
                        evidence=right.evidence,
                        summary=right.summary,
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
                        user_corrected=False,
                    )
                    out.pop(i)
                else:
                    # Merge with whichever neighbor has higher confidence
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
                            user_corrected=False,
                        )
                        out.pop(i)
                changed = True
                break
    return out


def bridge_sponsorship_gaps(
    segments: List[Segment],
    max_gap_sec: float = SPONSORSHIP_BRIDGE_MAX_GAP,
) -> List[Segment]:
    """Fuse [sponsorship, X, sponsorship] when X is non-sponsorship and short.

    Real ad inserts often contain a mid-roll voiceover or stinger that breaks the
    quiet/music pattern. Without bridging these, ad blocks fragment into 3+ pieces.
    The middle segment is rewritten as sponsorship (taking the lower of the two
    flanking confidences) and all three are merged.
    """
    if len(segments) < 3:
        return list(segments)

    out: List[Segment] = []
    i = 0
    while i < len(segments):
        if (
            i + 2 < len(segments)
            and segments[i].label == "sponsorship"
            and segments[i + 2].label == "sponsorship"
            and segments[i + 1].label != "sponsorship"
            and (segments[i + 1].end_sec - segments[i + 1].start_sec) <= max_gap_sec
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
                user_corrected=False,
            )
            out.append(merged)
            i += 3
        else:
            out.append(segments[i])
            i += 1
    return out


def renumber(segments: List[Segment]) -> List[Segment]:
    """Reset segment_id to be 0..N-1 in order."""
    return [
        Segment(
            segment_id=i,
            start_sec=s.start_sec,
            end_sec=s.end_sec,
            label=s.label,
            confidence=s.confidence,
            evidence=s.evidence,
            summary=s.summary,
            user_corrected=s.user_corrected,
        )
        for i, s in enumerate(segments)
    ]


def smooth_pipeline(segments: List[Segment]) -> List[Segment]:
    """Run merge → absorb → merge → bridge sponsorship gaps → merge again → renumber."""
    s = merge_adjacent_same_label(segments)
    s = absorb_short_segments(s)
    s = merge_adjacent_same_label(s)
    # Bridge can iterate: collapsing one gap may reveal another to collapse.
    while True:
        bridged = bridge_sponsorship_gaps(s)
        if len(bridged) == len(s):
            break
        s = merge_adjacent_same_label(bridged)
    s = renumber(s)
    return s