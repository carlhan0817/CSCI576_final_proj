# Ad End Time Detection — Improvement Plan (2026-05-07)

## Problem Statement

`_relabel_ad_blocks` in `export.py` correctly identifies **ad start times** but terminates
too early when scanning for the ad **end**. All three reference ads in test_009 (Kellogg's
60–90s, Ramp 330–361s, UberEats 541–596s) contain voiceover narration, which triggers the
current primary end signal ("speech at a hard cut") 10–25 seconds before the actual ad ends.

Current results vs. reference:

| Ad | Reference | Detected |
|----|-----------|----------|
| Kellogg's | 60–90s | 60–75s (−15s) |
| Ramp | 330–361s | 337–352s (−9s) |
| UberEats | 541–596s | fragmented |

---

## Root Cause

The two termination signals in `_relabel_ad_blocks` (`export.py:75–88`):

```python
# Primary: speech at a hard cut
if hc_t and sp_t:
    t_end = t + 1; break

# Secondary: luminance flips sign with magnitude >= 50
if hc_t and (ld_t * start_sign) < 0 and abs(ld_t) >= _LUM_END_THRESHOLD:
    t_end = t + 1; break
```

- **Primary** fires on ad voiceover narration (single-second speech bursts at product-shot cuts)
  because it cannot distinguish "host resumed speaking" from "ad narrator is speaking."
- **Secondary** misses gradual luminance returns and requires a very large (50-unit) reversal.

---

## Available Grid Signals

| Column | Useful for |
|--------|-----------|
| `is_hard_cut` | frame-accurate scene cuts |
| `lum_context_delta` | brightness jump at each second |
| `is_speech` | VAD per second |
| `hc_burst_density` | cut density in a 60s sliding window |
| `clip_*` (10 labels) | CLIP scene probabilities |
| `rms_energy` | audio energy |
| `hist_diff` | HSV histogram diff per second |

---

## Fix A — Require Sustained Speech (Minimal, Highest Confidence)

**File**: `export.py` — `_relabel_ad_blocks`

**Change**: Replace the 1-second speech check with N ≥ 3 **consecutive** speech seconds.

```python
# Current (fires on single-second ad narration bursts):
if hc_t and sp_t:
    t_end = t + 1
    break

# Proposed:
SUSTAINED_SPEECH_SEC = 3
if hc_t and sp_t:
    run = sum(bool(is_speech[min(t + k, T - 1)]) for k in range(SUSTAINED_SPEECH_SEC))
    if run >= SUSTAINED_SPEECH_SEC:
        t_end = t + 1
        break
```

**Why it works**: Ad voiceover bursts are typically 1–2 seconds per product shot before the
next cut. A content host resuming after an ad typically speaks for 5–20 consecutive seconds.
`SUSTAINED_SPEECH_SEC = 3` separates these cases cleanly.

**Risk**: Very low. The only failure mode is a host who begins with a very short utterance
at a hard cut immediately after a long ad — extremely rare and would shift end time by ≤ 2s.

---

## Fix B — Burst Density Drop Signal (For Burst-Detected Ads)

**File**: `export.py` — `_relabel_ad_blocks`

**Change**: Add a new termination signal when `hc_burst_density` clearly exits the burst
zone at a hard cut.

```python
_BURST_EXIT_THRESHOLD = 0.04  # clearly below BURST_DENSITY_THRESHOLD (0.10)

# New third signal in the scan loop:
if hc_t and burst_density is not None:
    bd_t = float(burst_density[min(t, T - 1)])
    if bd_t < _BURST_EXIT_THRESHOLD:
        t_end = t + 1
        break
```

Also pull `burst_density` from the grid at the top of the function alongside `is_hard_cut`:

```python
burst_density = grid.get("hc_burst_density")
```

**Why it works**: Burst-detected ad starts use the STEP-UP in `hc_burst_density` from
sparse to dense. The step-down is equally distinctive: leaving the ad block causes cut
density to drop sharply back to content levels. This signal is immune to narration because
it reads the visual edit rhythm, not the audio track.

**Limitation**: Only applies when the ad was detected via `_find_burst_starts`. For ads
found via silence/CLIP only, burst density may not be elevated. The UberEats ad (test_009
Ad 3) is burst-detected, so Fix B directly targets that case.

---

## Fix C — CLIP Content-Label Rollup (Semantic Visual Signal)

**File**: `export.py` — `_relabel_ad_blocks`

**Change**: At each candidate hard cut, compute a 3-second rolling mean of content-facing
CLIP labels. If the combined content score crosses a threshold, the ad has ended.

```python
_CONTENT_CLIP_KEYS = [
    "clip_a person talking to a camera",
    "clip_a presentation slide",
    "clip_a screen recording of software",
]
_CLIP_CONTENT_THRESHOLD = 0.40  # combined score across 3 labels to signal end


def _clip_content_score(t: int, grid: Dict, T: int, window: int = 3) -> float:
    total = 0.0
    for k in _CONTENT_CLIP_KEYS:
        arr = grid.get(k)
        if arr is not None:
            total += float(arr[min(t, T - 1) : min(t + window, T)].mean())
    return total


# In the scan loop:
if hc_t and _clip_content_score(t, grid, T) >= _CLIP_CONTENT_THRESHOLD:
    t_end = t + 1
    break
```

**Why it works**: Inside a Kellogg's/Ramp/UberEats ad, CLIP assigns high probability to
"a television commercial." When the video cuts back to the host, CLIP shifts to "a person
talking to a camera" or "a presentation slide." The shift is visually unambiguous and
audio-independent.

**Limitation**: CLIP is probabilistic. An ad showing a person explaining a product may
score high on "person talking," causing a false early exit. The 3-second rolling window
and the 0.40 combined threshold reduce this risk but may need per-video tuning.

**Risk**: Medium. Tune `_CLIP_CONTENT_THRESHOLD` empirically on test_009 and test_010
before committing a final value.

---

## Fix D — Lower Luminance Threshold

**File**: `export.py` — constant `_LUM_END_THRESHOLD`

**Change**:

```python
# Current:
_LUM_END_THRESHOLD = 50.0

# Proposed:
_LUM_END_THRESHOLD = 30.0
```

**Why**: The 50-unit threshold was calibrated for dramatic content→ad brightness swings.
Ad→content returns are often softer (gradual fade back to talking-head lighting). Lowering
to 30 catches more of these. The sign-flip requirement already filters random brightness
variation, so false positives remain low.

**Risk**: Low. One constant change, no logic modification.

---

## Recommended Implementation Order

Apply in sequence — each fix is independently useful and can be tested in isolation:

1. **Fix A** — sustained speech check. Three lines in `_relabel_ad_blocks`. Test immediately.
2. **Fix D** — lower `_LUM_END_THRESHOLD` from 50 → 30. One constant.
3. **Fix B** — burst density drop signal. One new constant + one grid lookup + 3-line branch.
4. **Fix C** — CLIP content rollup. Add helper function + 2-line call. Tune threshold.

---

## Expected Impact on test_009

| Ad | Reference | Current | After A+D | After A+D+B | After all fixes |
|----|-----------|---------|-----------|-------------|-----------------|
| Kellogg's | 60–90s | 60–75s | ~60–88s | ~60–90s | ~60–90s |
| Ramp | 330–361s | 337–352s | ~337–361s | ~330–361s | ~330–361s |
| UberEats | 541–596s | fragmented | ~541–580s | ~541–596s | ~541–596s |

Fix A targets Ads 1 and 2 (narration-inside-ad problem).
Fix B targets Ad 3 (burst zone exit).
Fix D is a general improvement across all three.
Fix C provides a visual fallback for any ad the other signals miss.

---

## Files to Modify

| File | Fixes |
|------|-------|
| `backend/pipeline/fusion/export.py` | A, B, C, D — all changes are isolated to `_relabel_ad_blocks` and its constants |

No other pipeline files need to change. The grid schema, segment schema, and `ad_slots.py`
scoring are unaffected.
