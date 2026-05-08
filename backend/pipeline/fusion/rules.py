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
    "the link in the description",
]
# Phrases removed from SPONSOR_KEYWORDS that may still appear in cached text_features.json
# from older pipeline runs.  Explicitly block them in rule_sponsor_keyword so cached
# matches don't trigger false ad labels.
_REMOVED_SPONSOR_KEYWORDS: frozenset = frozenset({"go to", "check out the link"})
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
    """Low visual motion + no speech + near-silence → holding/title screen.

    Requires rms_energy < DEAD_AIR_RMS_THRESHOLD so that animation scenes with
    background music (low hist_diff but audible soundtrack) are not mistaken for
    holding screens.  A real holding screen has both visual stillness AND audio silence.
    """
    hist_diff = grid["hist_diff"]
    is_speech = grid["is_speech"]
    rms = grid.get("rms_energy", np.zeros(len(is_speech), dtype=np.float32))
    static = (
        (hist_diff < 0.05) & (is_speech == 0) & (rms < DEAD_AIR_RMS_THRESHOLD)
    ).astype(np.int8)
    hits = []
    for s, e in _find_runs(static, HOLDING_SCREEN_MIN_DURATION):
        hits.append(RuleHit(s, e, "holding_screen", "holding_screen", confidence=0.7))
    return hits


def rule_sponsor_keyword(text_features: TextFeatures, T: int) -> List[RuleHit]:
    """Sentences with sponsor matched_keywords → ad candidates (extends ±15s)."""
    hits = []
    for seg in text_features.segments:
        sponsor_matches = [
            kw for kw in seg.matched_keywords
            if kw.startswith("sponsor:")
            and kw.split(":", 1)[1] not in _REMOVED_SPONSOR_KEYWORDS
        ]
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


# Window sizes for animation-specific rules (wider than keyword-based windows)
ANIMATION_INTRO_WINDOW_SEC = 120
ANIMATION_OUTRO_WINDOW_SEC = 120
_ANIM_INTRO_CLIP_THRESHOLD  = 0.25   # mean CLIP prob over window
_ANIM_OUTRO_CLIP_THRESHOLD  = 0.20
_ANIM_INTRO_ANIM_THRESHOLD  = 0.40   # mean "animated scene" prob for theme-song fallback
_ANIM_INTRO_SPEECH_MAX      = 0.10   # intro must have very little speech
_ANIM_INTRO_MIN_SEC         = 10     # require at least this many pre-speech seconds


def rule_animation_intro(grid: Dict[str, np.ndarray]) -> List[RuleHit]:
    """
    Detect animation-style intros without speech keywords.

    Two signals are tried, in priority order:
    1. CLIP 'a title card' probability is high in the first ANIMATION_INTRO_WINDOW_SEC →
       the video opens with an explicit title-card screen.
    2. CLIP 'an animated scene' dominates the opening AND speech is nearly absent →
       the video opens with a theme song / non-dialogue sequence (anime OP, cartoon title).

    Complements rule_intro_window which requires verbal cues ("welcome back", etc.).
    """
    T = len(grid["is_speech"])
    end = min(T, ANIMATION_INTRO_WINDOW_SEC)

    title_card  = grid.get("clip_a title card",       np.zeros(T, dtype=np.float32))
    animated    = grid.get("clip_an animated scene",  np.zeros(T, dtype=np.float32))
    is_speech   = grid["is_speech"]

    # Signal 1: title card CLIP
    if title_card[:end].mean() > _ANIM_INTRO_CLIP_THRESHOLD:
        return [RuleHit(0, end, "intro", "animation_title_card", confidence=0.75)]

    # Signal 2: animated scene + no speech at the start
    if animated[:end].mean() > _ANIM_INTRO_ANIM_THRESHOLD and is_speech[:end].mean() < _ANIM_INTRO_SPEECH_MAX:
        # Find first sustained speech — that is where intro ends
        for t in range(end):
            if is_speech[t]:
                if t >= _ANIM_INTRO_MIN_SEC:
                    return [RuleHit(0, t, "intro", "animation_theme_song", confidence=0.65)]
                break
    return []


def rule_animation_outro(grid: Dict[str, np.ndarray]) -> List[RuleHit]:
    """
    Detect animation-style outros without speech keywords.

    CLIP 'an end credits screen' probability is high in the last
    ANIMATION_OUTRO_WINDOW_SEC → label it as outro.
    """
    T = len(grid["is_speech"])
    cutoff = max(0, T - ANIMATION_OUTRO_WINDOW_SEC)

    end_credits = grid.get("clip_an end credits screen", np.zeros(T, dtype=np.float32))

    if end_credits[cutoff:].mean() > _ANIM_OUTRO_CLIP_THRESHOLD:
        return [RuleHit(cutoff, T, "outro", "animation_end_credits", confidence=0.75)]
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
    all_hits.extend(rule_animation_intro(grid))
    all_hits.extend(rule_animation_outro(grid))
    return all_hits