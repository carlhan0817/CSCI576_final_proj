"""
Phase 3 Step 4: Per-segment classification.

For each [start, end) interval between adjacent boundaries, aggregate the
features and assign one of the 10 SegmentLabels.

Strategy:
1. Apply hard rules first — if a rule covers this segment with high confidence,
   take that label.
2. Otherwise, use CLIP scene probabilities + audio activity to pick between
   core_content / filler / transition / etc. via a simple weighted scoring.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from backend.pipeline.schemas import (
    Segment, SegmentEvidence, TextFeatures, TranscriptSegment,
)
from backend.pipeline.fusion.rules import RuleHit


# Map CLIP scene label substrings → SegmentLabels (covers all 10 V2.1 prompts).
# Substring matching against the raw label string (SCENE_LABELS in visual.py).
CLIP_LABEL_TO_SEGMENT: Dict[str, str] = {
    # Core content
    "presentation slide": "core_content",
    "person talking":     "core_content",
    "screen recording":   "core_content",
    "video game":         "core_content",
    "animated scene":     "core_content",
    # Structural / transition
    "blank screen":       "transition",
    "end credits":        "outro",
    "title card":         "intro",
    # Sponsor / promo
    "advertisement":      "ad",    # matches "an advertisement slide" (legacy) and
    "commercial":         "ad",    # matches "a television commercial" (v2.2)
    "sponsor logo":       "ad",
}


def _interval_overlap_seconds(s1: int, e1: int, s2: int, e2: int) -> int:
    """Overlap in seconds between [s1,e1) and [s2,e2)."""
    return max(0, min(e1, e2) - max(s1, s2))


def _aggregate_segment_features(
    grid: Dict[str, np.ndarray],
    start_sec: int,
    end_sec: int,
) -> Dict[str, float]:
    """Compute mean values over [start, end) for all numeric grid columns."""
    agg = {}
    for k, arr in grid.items():
        if k == "text_sentence_id":
            continue
        slice_ = arr[start_sec:end_sec]
        if len(slice_) == 0:
            agg[k] = 0.0
            continue
        if k == "text_sim_to_next":
            # NaN-aware mean
            mask = ~np.isnan(slice_)
            agg[k] = float(slice_[mask].mean()) if mask.any() else 0.0
        else:
            agg[k] = float(slice_.mean())
    return agg


def _resolve_rule_label_for_segment(
    rule_hits: List[RuleHit],
    start_sec: int,
    end_sec: int,
    coverage_threshold: float = 0.5,
) -> Tuple[str, float, List[str]]:
    """
    Find the strongest rule covering this segment.
    Returns (label, confidence, list_of_rule_names) or ("", 0.0, []) if none.
    """
    best = None
    triggered = []
    for hit in rule_hits:
        overlap = _interval_overlap_seconds(start_sec, end_sec, hit.start_sec, hit.end_sec)
        seg_len = max(1, end_sec - start_sec)
        coverage = overlap / seg_len
        if coverage >= coverage_threshold:
            triggered.append(hit.rule_name)
            if best is None or hit.confidence > best.confidence:
                best = hit
    if best is None:
        return "", 0.0, []
    return best.label, best.confidence, triggered


def _classify_by_clip_and_audio(agg: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
    """
    No rule matched → use CLIP + audio heuristic.
    Returns (label, confidence, {visual_score, audio_score, text_score}).
    """
    # Visual score: take the strongest CLIP label and map it
    clip_keys = [k for k in agg.keys() if k.startswith("clip_")]
    if not clip_keys:
        return "core_content", 0.3, {"visual": 0.0, "audio": 0.0, "text": 0.0}
    
    best_clip_key = max(clip_keys, key=lambda k: agg[k])
    best_clip_prob = agg[best_clip_key]
    raw_label = best_clip_key.replace("clip_", "")
    
    # Try fuzzy match against the heuristic table
    mapped_label = "core_content"  # default
    for substring, target in CLIP_LABEL_TO_SEGMENT.items():
        if substring in raw_label:
            mapped_label = target
            break
    
    # Audio modifier: if very little speech AND no transcript text, lean toward
    # transition/filler.  Two exceptions:
    #   1. Animation content ("an animated scene") is core content regardless of speech —
    #      action/music scenes have no dialogue but are genuine content.
    #   2. If the transcript has words in this segment (has_text > 0), the host is
    #      speaking even if the VAD missed it (common in music-heavy or animation audio).
    speech_ratio = agg.get("is_speech", 0.0)
    text_coverage = agg.get("has_text", 0.0)
    is_animation_content = "animated" in raw_label
    has_spoken_content = speech_ratio >= 0.1 or text_coverage > 0.0
    if not has_spoken_content and mapped_label == "core_content" and not is_animation_content:
        mapped_label = "filler"

    visual_score = best_clip_prob
    audio_score = speech_ratio  # higher speech = more likely "real" content
    text_score = float(agg.get("has_text", 0.0))
    
    # Confidence = average of three modality scores
    confidence = (visual_score + audio_score + text_score) / 3.0
    
    return mapped_label, confidence, {
        "visual": visual_score,
        "audio": audio_score,
        "text": text_score,
    }


def _extract_segment_summary(
    transcript_segments: List[TranscriptSegment],
    start_sec: float,
    end_sec: float,
) -> str:
    """Pick the longest sentence inside [start, end) as the summary."""
    candidates = [
        s for s in transcript_segments
        if s.start >= start_sec and s.end <= end_sec
    ]
    if not candidates:
        # Fallback: any overlapping sentence
        candidates = [
            s for s in transcript_segments
            if s.end > start_sec and s.start < end_sec
        ]
    if not candidates:
        return ""
    longest = max(candidates, key=lambda s: len(s.text))
    return longest.text.strip()[:200]


def classify_segments(
    grid: Dict[str, np.ndarray],
    boundaries: List[int],
    rule_hits: List[RuleHit],
    transcript_segments: List[TranscriptSegment],
    hard_cut_set: Optional[Set[int]] = None,
) -> List[Segment]:
    """
    Build the list of Segment objects from candidate boundaries + features + rules.

    hard_cut_set: seconds that originated from a visual hard cut in find_boundaries.
                  When provided, each segment whose left boundary falls in the set gets
                  has_hard_cut_before=True, which prevents smooth.py from erasing it.
    """
    if hard_cut_set is None:
        hard_cut_set = set()

    segments: List[Segment] = []
    for i in range(len(boundaries) - 1):
        s = boundaries[i]
        e = boundaries[i + 1]
        if e <= s:
            continue

        agg = _aggregate_segment_features(grid, s, e)
        rule_label, rule_conf, triggered_rules = _resolve_rule_label_for_segment(rule_hits, s, e)

        if rule_label:
            label = rule_label
            confidence = rule_conf
            clip_keys_in_agg = [k for k in agg if k.startswith("clip_")]
            if clip_keys_in_agg:
                visual_score = max(agg[k] for k in clip_keys_in_agg)
            else:
                visual_score = 0.0
            audio_score = agg.get("is_speech", 0.0)
            text_score = float(agg.get("has_text", 0.0))
        else:
            label, confidence, scores = _classify_by_clip_and_audio(agg)
            visual_score = scores["visual"]
            audio_score = scores["audio"]
            text_score = scores["text"]

        summary = _extract_segment_summary(transcript_segments, s, e)
        # Flag segments whose left boundary came from a hard cut so smooth.py
        # can preserve the boundary even when the label matches the left neighbor.
        has_hard_cut = s in hard_cut_set

        segments.append(Segment(
            segment_id=i,
            start_sec=float(s),
            end_sec=float(e),
            label=label,  # type: ignore[arg-type]
            confidence=round(confidence, 3),
            evidence=SegmentEvidence(
                visual_score=round(visual_score, 3),
                audio_score=round(audio_score, 3),
                text_score=round(text_score, 3),
                triggered_rules=triggered_rules,
            ),
            summary=summary,
            user_corrected=False,
            has_hard_cut_before=has_hard_cut,
        ))

    return segments