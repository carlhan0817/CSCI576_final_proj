"""
Phase 3 Step 2: Hard rule triggers.

High-confidence rules that don't need ML. Each rule outputs a per-second 
"candidate label" plus a reason string. Multiple rules can fire on the same
second; we collect all and let classify.py weigh them.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

from backend.pipeline.schemas import TextFeatures


@dataclass
class RuleHit:
    """A hard rule firing at a specific time range."""
    start_sec: int
    end_sec: int
    label: str            # candidate SegmentLabel
    rule_name: str        # e.g. "dead_air", "sponsor_keyword"
    confidence: float     # 0.0-1.0


# Keyword tables — tweak as you see fit
SPONSOR_KEYWORDS = [
    "sponsored by", "today's video is brought to you by", "brought to you by",
    "thanks to our sponsor", "this video is sponsored", "use code", "promo code",
    "go to", "check out the link", "the link in the description",
]
INTRO_KEYWORDS = [
    "welcome back", "welcome to", "in today's video", "in this video",
    "what's up everyone", "hey guys", "hello and welcome",
]
OUTRO_KEYWORDS = [
    "thanks for watching", "see you next time", "see you in the next video",
    "don't forget to subscribe", "hit the like button", "smash that like",
    "leave a comment below", "like and subscribe",
]
SELF_PROMO_KEYWORDS = [
    "subscribe to my channel", "follow me on", "join my discord",
    "patreon", "merchandise", "my new course",
]
RECAP_KEYWORDS = [
    "last time", "previously on", "in the previous video",
    "as we saw before", "to recap",
]


# Tunable thresholds
DEAD_AIR_MIN_DURATION = 5        # seconds of silence to count as dead_air
DEAD_AIR_RMS_THRESHOLD = 0.01    # RMS below this AND no speech → dead_air
HOLDING_SCREEN_MIN_DURATION = 8  # seconds of low-motion + silence
INTRO_WINDOW_SEC = 90            # search "intro" only in first N seconds
OUTRO_WINDOW_SEC = 90            # search "outro" only in last N seconds


def _find_runs(arr: np.ndarray, min_length: int) -> List[Tuple[int, int]]:
    """Return [(start, end), ...] of consecutive 1-runs in a binary array, length >= min_length."""
    runs = []
    n = len(arr)
    i = 0
    while i < n:
        if arr[i]:
            j = i
            while j < n and arr[j]:
                j += 1
            if j - i >= min_length:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def rule_dead_air(grid: Dict[str, np.ndarray]) -> List[RuleHit]:
    """Long low-energy silence (no speech AND rms below threshold) → dead_air."""
    is_speech = grid["is_speech"]
    rms = grid.get("rms_energy", np.zeros(len(is_speech), dtype=np.float32))
    silent = ((is_speech == 0) & (rms < DEAD_AIR_RMS_THRESHOLD)).astype(np.int8)
    hits = []
    for s, e in _find_runs(silent, DEAD_AIR_MIN_DURATION):
        hits.append(RuleHit(s, e, "dead_air", "dead_air", confidence=0.9))
    return hits


def rule_holding_screen(grid: Dict[str, np.ndarray]) -> List[RuleHit]:
    """Low visual motion + no speech for a long time → likely holding/title screen."""
    hist_diff = grid["hist_diff"]
    is_speech = grid["is_speech"]
    static = ((hist_diff < 0.05) & (is_speech == 0)).astype(np.int8)
    hits = []
    for s, e in _find_runs(static, HOLDING_SCREEN_MIN_DURATION):
        hits.append(RuleHit(s, e, "holding_screen", "holding_screen", confidence=0.7))
    return hits


def rule_sponsor_keyword(text_features: TextFeatures, T: int) -> List[RuleHit]:
    """Sentences with sponsor matched_keywords → ad candidates (extends ±15s)."""
    hits = []
    for seg in text_features.segments:
        sponsor_matches = [kw for kw in seg.matched_keywords if kw.startswith("sponsor:")]
        if sponsor_matches:
            s = max(0, int(np.floor(seg.start - 15)))
            e = min(T, int(np.ceil(seg.end + 15)))
            hits.append(RuleHit(s, e, "ad", sponsor_matches[0], confidence=0.85))
    return hits


def rule_self_promo_keyword(text_features: TextFeatures, T: int) -> List[RuleHit]:
    hits = []
    for seg in text_features.segments:
        promo_matches = [kw for kw in seg.matched_keywords if kw.startswith("self_promo:")]
        if promo_matches:
            s = max(0, int(np.floor(seg.start - 5)))
            e = min(T, int(np.ceil(seg.end + 5)))
            hits.append(RuleHit(s, e, "self_promotion", promo_matches[0], confidence=0.75))
    return hits


def rule_recap_keyword(text_features: TextFeatures, T: int) -> List[RuleHit]:
    hits = []
    for seg in text_features.segments:
        recap_matches = [kw for kw in seg.matched_keywords if kw.startswith("recap:")]
        if recap_matches:
            s = max(0, int(np.floor(seg.start - 5)))
            e = min(T, int(np.ceil(seg.end + 30)))  # recaps can be longer
            hits.append(RuleHit(s, e, "recap", recap_matches[0], confidence=0.7))
    return hits


def rule_intro_window(text_features: TextFeatures, grid: Dict[str, np.ndarray]) -> List[RuleHit]:
    """In the first 90s, if any intro matched_keyword fires, flag intro."""
    T = len(grid["is_speech"])
    for seg in text_features.segments:
        if seg.start > INTRO_WINDOW_SEC:
            break
        intro_matches = [kw for kw in seg.matched_keywords if kw.startswith("intro:")]
        if intro_matches:
            e = min(T, int(np.ceil(seg.end + 10)))
            return [RuleHit(0, e, "intro", intro_matches[0], confidence=0.7)]
    return []


def rule_outro_window(text_features: TextFeatures, grid: Dict[str, np.ndarray]) -> List[RuleHit]:
    """In the last 90s, if any outro matched_keyword fires, flag outro."""
    T = len(grid["is_speech"])
    cutoff = max(0, T - OUTRO_WINDOW_SEC)
    for seg in text_features.segments:
        if seg.end < cutoff:
            continue
        outro_matches = [kw for kw in seg.matched_keywords if kw.startswith("outro:")]
        if outro_matches:
            s = max(0, int(np.floor(seg.start - 5)))
            return [RuleHit(s, T, "outro", outro_matches[0], confidence=0.8)]
    return []


def run_all_rules(
    grid: Dict[str, np.ndarray],
    text_features: TextFeatures,
) -> List[RuleHit]:
    """Run every rule, return all hits (overlaps OK; classify.py resolves them)."""
    T = len(grid["is_speech"])
    all_hits: List[RuleHit] = []
    all_hits.extend(rule_dead_air(grid))
    all_hits.extend(rule_holding_screen(grid))
    all_hits.extend(rule_sponsor_keyword(text_features, T))
    all_hits.extend(rule_self_promo_keyword(text_features, T))
    all_hits.extend(rule_recap_keyword(text_features, T))
    all_hits.extend(rule_intro_window(text_features, grid))
    all_hits.extend(rule_outro_window(text_features, grid))
    return all_hits