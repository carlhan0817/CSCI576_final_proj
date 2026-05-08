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
INTRO_KEYWORDS = [
    "welcome back", "welcome to", "in today's video", "in this video",
    "what's up everyone", "hey guys", "hello and welcome",
]
OUTRO_KEYWORDS = [
    "thanks for watching", "see you next time", "see you in the next video",
    "don't forget to subscribe", "hit the like button", "smash that like",
    "leave a comment below", "like and subscribe",
    "see you in the next", "hope you enjoyed", "i hope you enjoyed",
    "had a lot of fun making", "thanks for joining", "until next time",
    "bye for now", "see you soon",
]
SELF_PROMO_KEYWORDS = [
    "subscribe to my channel", "follow me on", "join my discord",
    "patreon", "merchandise", "my new course",
]
RECAP_KEYWORDS = [
    "last time", "previously on", "in the previous video",
    "as we saw before", "to recap",
]
# CTA phrases — high-precision, ad-specific. Used by both OCR (ocr_signals.py)
# and transcript matching (text.py). "subscribe" deliberately omitted (YouTube
# self-promo, not a clear ad signal).
CTA_KEYWORDS = [
    "shop now", "buy now", "order now", "order today", "available at",
    "available now", "limited time", "for a limited time", "offer ends",
    "introducing the", "new from", "use promo code", "use code",
    "download the app", "visit our store", "in stores now", "in stores today",
]


# Tunable thresholds
DEAD_AIR_MIN_DURATION = 5        # seconds of silence to count as dead_air
DEAD_AIR_RMS_THRESHOLD = 0.01    # RMS below this AND no speech → dead_air
HOLDING_SCREEN_MIN_DURATION = 8  # seconds of low-motion + silence
INTRO_WINDOW_SEC = 90            # search "intro" only in first N seconds
OUTRO_WINDOW_SEC = 90            # search "outro" only in last N seconds
AD_BREAK_MIN_DURATION = 20       # min seconds of (mostly) no speech to flag as sponsorship
AD_BREAK_MAX_GAP = 3             # tolerate brief speech bursts up to this many seconds
# Vlog/lifestyle content often has background music during b-roll: VAD says is_speech=False
# but RMS is high (0.05+). Only treat a section as a "break" when audio is genuinely quiet.
AD_BREAK_MAX_RMS = 0.05          # segments above this RMS have active audio → not a break

# rule_low_energy_audio: lecture audio is recorded close-mic with high RMS
# and wide spectral bandwidth. Inserted ads (rap, song, voiceover) come from
# remote / mixed sources with lower RMS and narrower bandwidth.
#
# Z-4 fix (2026-05-06): the previous fixed thresholds (0.18 / 1900 Hz) were
# tuned on test_004 and over-fired on test_001/002/003 whose lecture audio
# itself sits below those values, painting most of the video as sponsorship.
# We now derive RMS/BW cutoffs as per-video percentiles of the *non-silent*
# distribution — silence (rms < LOW_ENERGY_RMS_MIN) would otherwise pull
# both percentiles toward zero and disable the rule entirely.
LOW_ENERGY_RMS_PERCENTILE = 15
LOW_ENERGY_BW_PERCENTILE = 15
LOW_ENERGY_RMS_MIN = 0.005       # exclude true dead_air (handled by rule_dead_air)
LOW_ENERGY_MIN_DURATION = 10
LOW_ENERGY_MAX_GAP = 3


def _close_short_gaps(arr: np.ndarray, max_gap: int) -> np.ndarray:
    """Morphological closing on a binary array: fill 0-runs of length <= max_gap
    that are bordered by 1s on both sides. Used to merge nearby quiet stretches
    across brief speech bursts."""
    out = arr.copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i] == 0:
            j = i
            while j < n and out[j] == 0:
                j += 1
            if (j - i) <= max_gap and i > 0 and j < n and arr[i - 1] == 1 and arr[j] == 1:
                out[i:j] = 1
            i = j
        else:
            i += 1
    return out


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


def rule_low_energy_audio(grid: Dict[str, np.ndarray]) -> List[RuleHit]:
    """Sustained low-RMS + narrow-bandwidth audio → likely inserted ad.

    Targets ad inserts that VAD classifies as speech (rap, song with vocals)
    so rule_ad_break misses them. Distinguishes from dead_air by requiring
    RMS above LOW_ENERGY_RMS_MIN.

    Z-4 (2026-05-06): RMS and BW cutoffs are now per-video percentiles of the
    *non-silent* distribution. Computing the percentile across the full track
    including silence (rms < LOW_ENERGY_RMS_MIN) would collapse both
    percentiles toward zero and the rule would never fire.

    Path A (2026-05-06): require at least one second of audio_block_drift
    crossing AUDIO_BLOCK_DRIFT_THRESHOLD inside the run before firing.
    Lecture's natural quiet pauses are stylistically continuous with the
    surrounding lecture (drift low) and should NOT be flagged as
    sponsorship; inserted ad audio always has at least transient drift
    against the lecture baseline.
    """
    rms = grid.get("rms_energy")
    bw = grid.get("spectral_bandwidth")
    audio_drift = grid.get("audio_block_drift")
    if rms is None or bw is None:
        return []

    active = rms >= LOW_ENERGY_RMS_MIN
    if not active.any():
        return []
    rms_thresh = float(np.percentile(rms[active], LOW_ENERGY_RMS_PERCENTILE))
    bw_thresh = float(np.percentile(bw[active], LOW_ENERGY_BW_PERCENTILE))

    cond = (
        (rms < rms_thresh)
        & (rms >= LOW_ENERGY_RMS_MIN)
        & (bw < bw_thresh)
    ).astype(np.int8)
    closed = _close_short_gaps(cond, LOW_ENERGY_MAX_GAP)
    hits = []
    for s, e in _find_runs(closed, LOW_ENERGY_MIN_DURATION):
        # Path A: gate on block_drift confirmation. Skip silently when the
        # signal is unavailable (treat the legacy behaviour as default).
        if audio_drift is not None:
            run_drift = audio_drift[s:e]
            if not (run_drift >= AUDIO_BLOCK_DRIFT_THRESHOLD).any():
                continue
        hits.append(RuleHit(s, e, "sponsorship", "low_energy_audio", confidence=0.75))
    return hits


def rule_ad_break(grid: Dict[str, np.ndarray]) -> List[RuleHit]:
    """Long stretch of genuine silence (no speech + low RMS) → ADVISORY sponsorship.

    Originally a strong signal, demoted because is_speech-only blocks generate
    too many false positives on animation / sports / news / vlog content. Use as a
    tiebreaker; rule_ad_block carries the high-confidence detection.

    Path A (2026-05-06): require at least one second of audio_block_drift
    crossing AUDIO_BLOCK_DRIFT_THRESHOLD inside the run.

    RMS gate (2026-05-08): vlogs with background music have is_speech=False but
    non-zero RMS during b-roll (music, ambient sound). Those sections are NOT ad
    breaks. Require rms < AD_BREAK_MAX_RMS to define "quiet" — this filters out
    content sections with background music/audio that only VAD-silence looks like.
    """
    is_speech = grid["is_speech"]
    rms = grid.get("rms_energy", np.zeros(len(is_speech), dtype=np.float32))
    audio_drift = grid.get("audio_block_drift")
    quiet = ((is_speech == 0) & (rms < AD_BREAK_MAX_RMS)).astype(np.int8)
    quiet_closed = _close_short_gaps(quiet, AD_BREAK_MAX_GAP)
    hits = []
    for s, e in _find_runs(quiet_closed, AD_BREAK_MIN_DURATION):
        if audio_drift is not None:
            if not (audio_drift[s:e] >= AUDIO_BLOCK_DRIFT_THRESHOLD).any():
                continue
        hits.append(RuleHit(s, e, "sponsorship", "ad_break", confidence=0.5))
    return hits


# Tunable thresholds for rule_ad_block.
#
# Path X rewrite (2026-05-06): the old (style_drift, audio_drift) pair is a
# *boundary* detector — drift spikes at splice points and decays inside an ad
# body, so the original "sustained ≥10s OR" gate was mathematically unable to
# fire. The new (style_block_drift, audio_block_drift) pair uses asymmetric
# far-only context (and per-dim z-scored MFCC for audio) so drift stays high
# throughout the ad body. See docs/walkthrough/path-z-deferred.md for context.
AD_BLOCK_MIN_DURATION = 10           # min length of a sustained-drift run (seconds)
AD_BLOCK_BOUNDARY_TOLERANCE = 3      # how close a hard cut must be to count as "bounding"
# At the very start of a video the MFCC baseline hasn't stabilised yet, so the
# first few seconds often show spurious drift against the rest of the lecture.
# Skip any run that ends before this guard to suppress false-positive ad_block
# hits on the opening frames.
AD_BLOCK_START_GUARD = 30            # ignore runs that end within the first N seconds

# Block-drift thresholds (empirical: cover both test_001 and test_004 with
# clean baseline separation; see scripts/diagnose_ad_block.py).
VISUAL_BLOCK_DRIFT_THRESHOLD = 0.40
AUDIO_BLOCK_DRIFT_THRESHOLD = 1.25

# Asymmetric trust: audio z-score is the trusted primary signal; visual alone
# requires hard-cut boundaries on both sides because lecture-style content can
# generate sustained visual drift without being an ad (test_001 baseline).
AD_BLOCK_AUDIO_CONFIDENCE = 0.80           # audio run alone (above holding_screen 0.70)
AD_BLOCK_BOTH_CONFIDENCE = 0.90            # audio AND visual runs overlap
AD_BLOCK_VISUAL_BOUNDED_CONFIDENCE = 0.60  # visual run alone, both ends cut
# Require at least this CLIP combined ad score within the run to fire.
# Suppresses false positives from MFCC drift contamination in vlog/mixed content.
AD_BLOCK_CLIP_GATE = 0.72


def _has_bounding_cut(grid: Dict[str, np.ndarray], t: int, tolerance: int) -> bool:
    cuts = grid.get("is_hard_cut")
    if cuts is None:
        return False
    lo = max(0, t - tolerance)
    hi = min(len(cuts), t + tolerance + 1)
    return bool(cuts[lo:hi].any())


def _runs_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or a[0] >= b[1])


def rule_ad_block(grid: Dict[str, np.ndarray]) -> List[RuleHit]:
    """Multi-signal ad detector using block-drift signals (Path X).

    Asymmetric-trust design:
      - audio_block_drift fires standalone — z-scored MFCC over an asymmetric
        ±30/±90s context cleanly separates inserted ads from lecture content
        in our corpus.
      - style_block_drift fires standalone only when bounded by hard cuts on
        BOTH ends. Lecture videos with frequent slide cuts can sustain visual
        drift without being ads (test_001 baseline), so visual alone needs
        corroboration.

    Confidence:
      - audio AND visual runs overlap            : 0.90
      - audio run alone                          : 0.80   (above holding_screen)
      - visual run alone, both ends bounded by   : 0.60   (loses to holding_screen
        a hard cut                                          until the argmax fix)
    """
    audio_drift = grid.get("audio_block_drift")
    visual_drift = grid.get("style_block_drift")
    if audio_drift is None and visual_drift is None:
        return []

    T = len(audio_drift if audio_drift is not None else visual_drift)
    a = audio_drift if audio_drift is not None else np.zeros(T, dtype=np.float32)
    v = visual_drift if visual_drift is not None else np.zeros(T, dtype=np.float32)

    audio_runs = _find_runs(
        (a >= AUDIO_BLOCK_DRIFT_THRESHOLD).astype(np.int8),
        AD_BLOCK_MIN_DURATION,
    )
    visual_runs_bounded = [
        (s, e) for (s, e) in _find_runs(
            (v >= VISUAL_BLOCK_DRIFT_THRESHOLD).astype(np.int8),
            AD_BLOCK_MIN_DURATION,
        )
        if _has_bounding_cut(grid, s, AD_BLOCK_BOUNDARY_TOLERANCE)
        and _has_bounding_cut(grid, e, AD_BLOCK_BOUNDARY_TOLERANCE)
    ]

    # CLIP gate: require at least some visual ad evidence in the drift run to
    # suppress false positives from MFCC drift contamination near ad boundaries
    # (especially in vlog content where audio style varies naturally).
    ad_slide_arr = grid.get("clip_an advertisement slide", np.zeros(T, dtype=np.float32))
    sponsor_logo_arr = grid.get("clip_a sponsor logo or product", np.zeros(T, dtype=np.float32))
    clip_combined = np.maximum(ad_slide_arr, sponsor_logo_arr)

    hits: List[RuleHit] = []

    # Audio runs are trusted standalone; promote to BOTH confidence when a
    # bounded visual run agrees.
    for run in audio_runs:
        if run[1] <= AD_BLOCK_START_GUARD:
            continue  # opening-frames false positive guard
        if float(clip_combined[run[0]:run[1]].max()) < AD_BLOCK_CLIP_GATE:
            continue  # suppress content regions with no visual ad evidence
        has_visual_support = any(_runs_overlap(run, vr) for vr in visual_runs_bounded)
        conf = AD_BLOCK_BOTH_CONFIDENCE if has_visual_support else AD_BLOCK_AUDIO_CONFIDENCE
        hits.append(RuleHit(run[0], run[1], "sponsorship", "ad_block", confidence=conf))

    # Visual-only path: bounded run with no audio overlap.
    for vrun in visual_runs_bounded:
        if vrun[1] <= AD_BLOCK_START_GUARD:
            continue  # opening-frames false positive guard
        if any(_runs_overlap(vrun, ar) for ar in audio_runs):
            continue
        if float(clip_combined[vrun[0]:vrun[1]].max()) < AD_BLOCK_CLIP_GATE:
            continue
        hits.append(RuleHit(vrun[0], vrun[1], "sponsorship", "ad_block",
                            confidence=AD_BLOCK_VISUAL_BOUNDED_CONFIDENCE))

    return hits


CLIP_AD_HIGH_THRESHOLD = 0.75     # high-confidence CLIP ad label threshold
CLIP_AD_MED_THRESHOLD = 0.35      # medium-confidence used to grow the hit region
CLIP_AD_EXPAND_BACK_SEC = 55      # expand N seconds BACKWARD from first hit (cover long ads)
CLIP_AD_EXPAND_FWRD_SEC = 2       # expand N seconds FORWARD (minimal: stop content bleed)
CLIP_AD_BRIDGE_GAP_SEC = 28       # tolerate up to this many consecutive non-medium-conf
                                   # seconds while walking backward (bridges internal ad gaps)


def rule_clip_high_confidence_ad(grid: Dict[str, np.ndarray]) -> List[RuleHit]:
    """Detect inserted ads via high-confidence CLIP ad labels (≥0.75).

    At threshold ≥0.75, 'an advertisement slide' and 'a sponsor logo or product'
    are empirically high-precision for actual TV ad inserts (tested on test_009:
    hits at t=87/357/592-593 — all inside real ads; content false positives at
    t=538/624 are 0.711/0.705, safely below the threshold).

    Strategy:
    1. Find high-confidence (≥0.75) anchor clusters.
    2. Walk backward from the anchor through medium-confidence (≥0.30) seconds,
       tolerating gaps up to CLIP_AD_BRIDGE_GAP_SEC consecutive non-medium-conf
       seconds (bridges scene-change gaps inside long ads).
    3. Expand only CLIP_AD_EXPAND_FWRD_SEC forward to prevent content bleed
       (high CLIP signals often appear near the end of an ad).
    """
    T = len(grid["is_speech"])
    ad_slide = grid.get("clip_an advertisement slide", np.zeros(T, dtype=np.float32))
    sponsor_logo = grid.get("clip_a sponsor logo or product", np.zeros(T, dtype=np.float32))
    combined = np.maximum(ad_slide, sponsor_logo)

    high_conf = (combined >= CLIP_AD_HIGH_THRESHOLD).astype(np.int8)
    if not high_conf.any():
        return []

    med_conf = (combined >= CLIP_AD_MED_THRESHOLD).astype(np.int8)
    high_closed = _close_short_gaps(high_conf, 5)
    hits = []
    for s, e in _find_runs(high_closed, 1):
        # Walk backward from s, tolerating gaps ≤ CLIP_AD_BRIDGE_GAP_SEC.
        # Stops when we accumulate more consecutive non-medium-conf seconds
        # than the bridge tolerance (this prevents walking into content regions).
        back_limit = max(0, s - CLIP_AD_EXPAND_BACK_SEC)
        left = s
        consecutive_false = 0
        for t in range(s - 1, back_limit - 1, -1):
            if med_conf[t]:
                left = t
                consecutive_false = 0
            else:
                consecutive_false += 1
                if consecutive_false > CLIP_AD_BRIDGE_GAP_SEC:
                    break

        seg_start = max(0, left)
        seg_end = min(T, e + CLIP_AD_EXPAND_FWRD_SEC)
        hits.append(RuleHit(seg_start, seg_end, "sponsorship", "clip_high_conf_ad",
                            confidence=0.88))

    # Discard merged hit regions wider than MAX_CLIP_AD_HIT_SEC.
    # Lecture videos with many consecutive title slides (all CLIP >= 0.75) produce
    # overlapping anchors that merge into a huge false positive; TV ads in our
    # dataset are at most 60-90s so anything wider is spurious.
    MIN_CLIP_AD_HIT_SEC = 15   # shorter than shortest known TV ad (28s); filters title-card FPs
    MAX_CLIP_AD_HIT_SEC = 90   # wider than widest known TV ad (55s); filters lecture FPs
    if hits:
        coverage = np.zeros(T, dtype=np.int8)
        for hit in hits:
            coverage[hit.start_sec:hit.end_sec] = 1
        merged_hits = []
        for s, e in _find_runs(coverage, 1):
            if MIN_CLIP_AD_HIT_SEC <= e - s <= MAX_CLIP_AD_HIT_SEC:
                best_conf = max(h.confidence for h in hits
                                if h.start_sec < e and h.end_sec > s)
                merged_hits.append(RuleHit(s, e, "sponsorship", "clip_high_conf_ad",
                                           confidence=best_conf))
        return merged_hits
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
    """Sentences with sponsor matched_keywords → sponsorship candidates (extends ±15s)."""
    hits = []
    for seg in text_features.segments:
        sponsor_matches = [kw for kw in seg.matched_keywords if kw.startswith("sponsor:")]
        if sponsor_matches:
            s = max(0, int(np.floor(seg.start - 15)))
            e = min(T, int(np.ceil(seg.end + 15)))
            hits.append(RuleHit(s, e, "sponsorship", sponsor_matches[0], confidence=0.85))
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
    """In the last 90s, if any outro matched_keyword fires, flag outro.

    Falls back to raw-text matching against OUTRO_KEYWORDS when matched_keywords
    are stale (pre-cached before new keywords were added).
    """
    T = len(grid["is_speech"])
    cutoff = max(0, T - OUTRO_WINDOW_SEC)
    for seg in text_features.segments:
        if seg.end < cutoff:
            continue
        outro_matches = [kw for kw in seg.matched_keywords if kw.startswith("outro:")]
        # Fallback: raw-text scan covers newly added keywords not yet cached.
        if not outro_matches and hasattr(seg, "text") and seg.text:
            text_lower = seg.text.lower()
            for kw in OUTRO_KEYWORDS:
                if kw in text_lower:
                    outro_matches = [f"outro:{kw}"]
                    break
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
    all_hits.extend(rule_ad_block(grid))
    all_hits.extend(rule_ad_break(grid))
    all_hits.extend(rule_low_energy_audio(grid))
    all_hits.extend(rule_clip_high_confidence_ad(grid))
    return all_hits