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
AD_BREAK_MIN_DURATION = 15       # min seconds of (mostly) no speech to flag as sponsorship
AD_BREAK_MAX_GAP = 3             # tolerate brief speech bursts up to this many seconds

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
    if _is_vlog_mode(grid):
        return []
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
        # Hard-cut gate: a real ad insertion has a splice on at least one side.
        # Without this, animated / cinematic content with quiet musical stretches
        # gets painted as sponsorship.
        if not (_has_bounding_cut(grid, s, AD_BLOCK_BOUNDARY_TOLERANCE)
                or _has_bounding_cut(grid, e, AD_BLOCK_BOUNDARY_TOLERANCE)):
            continue
        hits.append(RuleHit(s, e, "sponsorship", "low_energy_audio", confidence=0.75))
    return hits


def rule_ad_break(grid: Dict[str, np.ndarray]) -> List[RuleHit]:
    """Long stretch of no speech → ADVISORY-confidence sponsorship.

    Originally a strong signal, demoted because is_speech-only blocks generate
    too many false positives on animation / sports / news content. Use as a
    tiebreaker; rule_ad_block carries the high-confidence detection.

    Path A (2026-05-06): require at least one second of audio_block_drift
    crossing AUDIO_BLOCK_DRIFT_THRESHOLD inside the run. A lecture's natural
    long pause is acoustically continuous with the surrounding lecture; an
    inserted ad break has at least transient drift against the baseline.
    Diagnostic: 630/791 s of test_003 sponsorship FP came from ad_break alone
    before this gate.
    """
    if _is_vlog_mode(grid):
        return []
    is_speech = grid["is_speech"]
    audio_drift = grid.get("audio_block_drift")
    quiet = (is_speech == 0).astype(np.int8)
    quiet_closed = _close_short_gaps(quiet, AD_BREAK_MAX_GAP)
    hits = []
    for s, e in _find_runs(quiet_closed, AD_BREAK_MIN_DURATION):
        if audio_drift is not None:
            if not (audio_drift[s:e] >= AUDIO_BLOCK_DRIFT_THRESHOLD).any():
                continue
        # Hard-cut gate: real ad inserts have a splice at one boundary or the
        # other. A lecture pause or an animated quiet stretch typically does
        # not. Without this, ad_break paints long stretches of test_010
        # (animation) and IDE-pause regions of test_008 as sponsorship.
        if not (_has_bounding_cut(grid, s, AD_BLOCK_BOUNDARY_TOLERANCE)
                or _has_bounding_cut(grid, e, AD_BLOCK_BOUNDARY_TOLERANCE)):
            continue
        hits.append(RuleHit(s, e, "sponsorship", "ad_break", confidence=0.5))
    return hits


# --------------------------------------------------------------------------- #
# Vlog mode (test_009-class content)                                          #
#                                                                             #
# Lecture-style videos satisfy "content = continuous speech, ad = quiet
# insertion". In vlog content the polarity inverts: B-roll has almost no
# speech, while inserted ads (kelloggs, ramp, ubereats) carry dense dialogue.
# rule_ad_break and rule_low_energy_audio therefore generate massive FP on
# vlog content. We detect vlog material from the joint signature
# (low global speech + dense hard cuts) and:
#   - disable rule_ad_break and rule_low_energy_audio (their assumption
#     is structurally inverted in vlog),
#   - relax rule_ad_block's min-duration and visual threshold so it can
#     catch the short drift bursts visible inside vlog ads (kelloggs has
#     a 6s visual-drift run at >=0.40, a 7s audio-drift run at >=1.25).
#
# Sign-off thresholds came from the 2026-05-07 diagnostic across
# test_003/008/009/010:
#   test_003 (lecture-ish): speech 0.39, cut 0.24 → not vlog
#   test_008 (lecture):     speech 0.80, cut 0.04 → not vlog
#   test_009 (vlog):        speech 0.38, cut 0.36 → vlog
#   test_010 (animation):   speech 0.24, cut 0.04 → not vlog
# --------------------------------------------------------------------------- #
VLOG_SPEECH_FRAC_MAX = 0.5
VLOG_CUT_DENSITY_MIN = 0.30
VLOG_AD_BLOCK_MIN_DURATION = 6
VLOG_VISUAL_BLOCK_DRIFT_THRESHOLD = 0.35
# Speech-burst rule (vlog-only): real ads in vlog content carry far denser
# dialogue than the surrounding B-roll (kelloggs 0.87, ramp 0.74, ubereats
# 0.91 vs typical vlog content 0.30-0.56). A sustained speech-frac window
# above this floor is the strongest available signal for vlog ads — visual
# and audio drifts are too noisy to fire reliably (ubereats has visual
# drift mean 0.31, below VISUAL_BLOCK_DRIFT_THRESHOLD).
VLOG_SPEECH_BURST_FRAC_MIN = 0.70
VLOG_SPEECH_BURST_WINDOW = 10
VLOG_SPEECH_BURST_MIN_DURATION = 12
# Cut-density band for vlog speech-burst confirmation. Real ads in vlogs are
# fast-cut commercials (kelloggs 0.43, ramp 0.45, ubereats 0.38 cuts/sec),
# while FP regions split into two failure modes:
#   - vlogger speaking direct-to-camera: cut <= 0.13 (long single takes)
#   - vlog B-roll montage with VO: cut >= 0.55 (rapid-fire scene changes)
# The 0.20-0.55 band keeps real ads while rejecting both FP modes.
VLOG_BURST_CUT_DENSITY_MIN = 0.20
VLOG_BURST_CUT_DENSITY_MAX = 0.55


def _is_vlog_mode(grid: Dict[str, np.ndarray]) -> bool:
    """Heuristic: low global speech + dense hard cuts → vlog content."""
    is_speech = grid.get("is_speech")
    cuts = grid.get("is_hard_cut")
    if is_speech is None or cuts is None or len(is_speech) == 0:
        return False
    speech_frac = float(is_speech.mean())
    cut_density = float(cuts.mean())
    return speech_frac < VLOG_SPEECH_FRAC_MAX and cut_density > VLOG_CUT_DENSITY_MIN


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

    vlog = _is_vlog_mode(grid)
    min_dur = VLOG_AD_BLOCK_MIN_DURATION if vlog else AD_BLOCK_MIN_DURATION
    visual_thresh = VLOG_VISUAL_BLOCK_DRIFT_THRESHOLD if vlog else VISUAL_BLOCK_DRIFT_THRESHOLD

    is_speech = grid.get("is_speech")

    def _vlog_speech_ok(run: Tuple[int, int]) -> bool:
        # vlog content discriminator: real ads in vlogs carry dense dialogue
        # (kelloggs speech_frac=0.87, ramp=0.74) while vlog B-roll FP regions
        # are mostly silent (post_ramp=0.16, post_kelloggs=0.30).
        if not vlog or is_speech is None:
            return True
        sf = float(is_speech[run[0]:run[1]].mean()) if run[1] > run[0] else 0.0
        return sf >= 0.5

    audio_runs = [
        r for r in _find_runs(
            (a >= AUDIO_BLOCK_DRIFT_THRESHOLD).astype(np.int8),
            min_dur,
        ) if _vlog_speech_ok(r)
    ]
    raw_visual_runs = _find_runs(
        (v >= visual_thresh).astype(np.int8),
        min_dur,
    )
    if vlog:
        # Vlog content has dense hard cuts everywhere, so requiring cuts on
        # both ends adds no signal. Accept any visual run that has a cut on
        # at least one side AND passes the speech-density gate.
        visual_runs_bounded = [
            (s, e) for (s, e) in raw_visual_runs
            if (_has_bounding_cut(grid, s, AD_BLOCK_BOUNDARY_TOLERANCE)
                or _has_bounding_cut(grid, e, AD_BLOCK_BOUNDARY_TOLERANCE))
            and _vlog_speech_ok((s, e))
        ]
    else:
        visual_runs_bounded = [
            (s, e) for (s, e) in raw_visual_runs
            if _has_bounding_cut(grid, s, AD_BLOCK_BOUNDARY_TOLERANCE)
            and _has_bounding_cut(grid, e, AD_BLOCK_BOUNDARY_TOLERANCE)
        ]

    hits: List[RuleHit] = []

    # Audio runs are trusted standalone; promote to BOTH confidence when a
    # bounded visual run agrees.
    for run in audio_runs:
        has_visual_support = any(_runs_overlap(run, vr) for vr in visual_runs_bounded)
        conf = AD_BLOCK_BOTH_CONFIDENCE if has_visual_support else AD_BLOCK_AUDIO_CONFIDENCE
        hits.append(RuleHit(run[0], run[1], "sponsorship", "ad_block", confidence=conf))

    # Visual-only path: bounded run with no audio overlap.
    for vrun in visual_runs_bounded:
        if any(_runs_overlap(vrun, ar) for ar in audio_runs):
            continue
        hits.append(RuleHit(vrun[0], vrun[1], "sponsorship", "ad_block",
                            confidence=AD_BLOCK_VISUAL_BOUNDED_CONFIDENCE))

    return hits


def rule_vlog_speech_burst(grid: Dict[str, np.ndarray]) -> List[RuleHit]:
    """Vlog-only: dense dialogue inside otherwise low-speech vlog content.

    In vlog material, B-roll is mostly silent (speech_frac 0.16-0.30) while
    inserted ads carry a constant voiceover (kelloggs 0.87, ramp 0.74,
    ubereats 0.91). We slide a window of length VLOG_SPEECH_BURST_WINDOW
    and mark seconds whose window-mean speech is >= VLOG_SPEECH_BURST_FRAC_MIN,
    then keep runs of length >= VLOG_SPEECH_BURST_MIN_DURATION.

    Skipped outside vlog mode (lecture videos have speech everywhere).
    """
    if not _is_vlog_mode(grid):
        return []
    is_speech = grid.get("is_speech")
    if is_speech is None:
        return []
    T = len(is_speech)
    half = VLOG_SPEECH_BURST_WINDOW // 2
    # Sliding-window speech fraction.
    speech_f = is_speech.astype(np.float32)
    cum = np.concatenate([[0.0], np.cumsum(speech_f)])
    window_mean = np.zeros(T, dtype=np.float32)
    for t in range(T):
        s = max(0, t - half)
        e = min(T, t + half + 1)
        window_mean[t] = (cum[e] - cum[s]) / max(1, e - s)
    mask = (window_mean >= VLOG_SPEECH_BURST_FRAC_MIN).astype(np.int8)
    closed = _close_short_gaps(mask, 3)
    cuts = grid.get("is_hard_cut")
    hits: List[RuleHit] = []
    for s, e in _find_runs(closed, VLOG_SPEECH_BURST_MIN_DURATION):
        # Confirm cut density is in the "ad" band — rejects both vlogger
        # direct-to-camera takes and B-roll-montage VOs.
        if cuts is not None:
            local_cut = float(cuts[s:e].mean()) if e > s else 0.0
            if local_cut < VLOG_BURST_CUT_DENSITY_MIN or local_cut > VLOG_BURST_CUT_DENSITY_MAX:
                continue
        hits.append(RuleHit(s, e, "sponsorship", "vlog_speech_burst", confidence=0.75))
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
    all_hits.extend(rule_ad_block(grid))
    all_hits.extend(rule_ad_break(grid))
    all_hits.extend(rule_low_energy_audio(grid))
    all_hits.extend(rule_vlog_speech_burst(grid))
    return all_hits