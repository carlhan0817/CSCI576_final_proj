# Phase 3: Fusion Pipeline — Fixes & Implementation

**Addresses:** Phase 3 gaps found after Phase 2 (V2.1) was completed  
**Date:** 2026-04-30  
**Implemented:** 2026-04-30  
**Demo deadline:** 2026-05-06  
**Status:** ✅ All changes implemented into source files  

---

## Table of Contents

1. [Fix 1 — `classify.py`: Stale CLIP label map (critical)](#1-fix-1--classifypy-stale-clip-label-map-critical)
2. [Fix 2 — `rules.py`: `rule_dead_air` ignores RMS energy](#2-fix-2--rulespy-rule_dead_air-ignores-rms-energy)
3. [Fix 3 — `boundaries.py`: Missing music transition signal](#3-fix-3--boundariespy-missing-music-transition-signal)
4. [Fix 4 — `rules.py`: Keyword rules re-scan raw text](#4-fix-4--rulespy-keyword-rules-re-scan-raw-text)
5. [Summary Table](#5-summary-table)

---

## 1. Fix 1 — `classify.py`: Stale CLIP label map (critical)

### Problem

`CLIP_LABEL_TO_SEGMENT` only covered the original 4 CLIP prompts. Phase 2 (V2.1) expanded `SCENE_LABELS` to 10 prompts, but 6 of the new labels had no mapping and fell through to the `"core_content"` default. This meant CLIP-detected sponsors, outros, title cards, and end-credits screens were all **mislabeled as `core_content`** — a visible failure at demo time.

**Old (4 entries, broken):**
```python
CLIP_LABEL_TO_SEGMENT: Dict[str, str] = {
    "blank screen":       "transition",
    "presentation slide": "core_content",
    "person talking":     "core_content",
    "screen recording":   "core_content",
}
# "end credits screen", "title card", "advertisement slide",
# "sponsor logo", "video game", "animated scene" → all fell through to core_content
```

### Fix

Updated to cover all 10 V2.1 prompts:

```python
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
    "advertisement":      "sponsorship",
    "sponsor logo":       "sponsorship",
}
```

**Why substring matching still works:** `raw_label = best_clip_key.replace("clip_", "")` gives the full prompt string (e.g. `"an end credits screen"`). Each substring above uniquely identifies its prompt.

---

## 2. Fix 2 — `rules.py`: `rule_dead_air` ignores RMS energy

### Problem

`rule_dead_air` only checked `is_speech == 0`. Loud music and crowd noise also have `is_speech == 0` (VAD correctly says "no speech") but are not dead air — they are music or noise segments. This caused the rule to over-fire, mislabeling background-music seconds as `dead_air`.

The PDD §4.3 explicitly states: "连续 ≥ N 秒 RMS 低于阈值**且** VAD = 0 → `dead_air`". Phase 2 (V2.1) added `rms_energy` to the per-second grid specifically to unblock this fix.

**Old:**
```python
def rule_dead_air(grid):
    is_speech = grid["is_speech"]
    silent = (is_speech == 0).astype(np.int8)
    ...
```

### Fix

Added `DEAD_AIR_RMS_THRESHOLD = 0.01` and require both conditions:

```python
DEAD_AIR_RMS_THRESHOLD = 0.01    # RMS below this AND no speech → dead_air

def rule_dead_air(grid: Dict[str, np.ndarray]) -> List[RuleHit]:
    """Long low-energy silence (no speech AND rms below threshold) → dead_air."""
    is_speech = grid["is_speech"]
    rms = grid.get("rms_energy", np.zeros(len(is_speech), dtype=np.float32))
    silent = ((is_speech == 0) & (rms < DEAD_AIR_RMS_THRESHOLD)).astype(np.int8)
    hits = []
    for s, e in _find_runs(silent, DEAD_AIR_MIN_DURATION):
        hits.append(RuleHit(s, e, "dead_air", "dead_air", confidence=0.9))
    return hits
```

**`grid.get("rms_energy", zeros)`** — safe fallback so the rule still works if an old workspace only has `is_speech` (no RMS). The zeros would make every second pass the RMS check, reproducing old behavior.

**Threshold rationale:** `0.01` is 10× above the `SILENCE_RMS_THRESHOLD` used in `audio.py` (`0.001`). A pure tone or music track sits at 0.05–0.5 RMS; an actual silent gap is < 0.001. The `0.01` midpoint keeps background hiss out of `dead_air` while catching true silence.

---

## 3. Fix 3 — `boundaries.py`: Missing music transition signal

### Problem

`find_boundaries` detected only 4 boundary signals. The PDD §4.3 lists "音频 'speech→music' 或 'music→silence' 的状态切换" as a required boundary signal. These transitions are strong indicators of:
- Speech → music: start of a sponsorship or intro
- Music → silence: end of an intro or sponsorship

Phase 2 added `is_music` to the per-second grid, but `boundaries.py` was never updated to use it.

### Fix

Added Signal 5 after the existing `is_speech` transition check:

```python
# Signal 5: speech ↔ music and music ↔ silence transitions
# Strong indicator of sponsorship/intro/outro boundaries.
if "is_music" in grid:
    is_music = grid["is_music"]
    for t in range(1, T):
        if is_music[t] != is_music[t - 1]:
            candidate_seconds.add(t)
```

**`if "is_music" in grid`** — guarded so old workspaces without the column don't crash.

The new boundary candidates feed into the existing `MIN_BOUNDARY_GAP_SEC = 3` dedup filter, so a music track that flickers won't generate dozens of micro-boundaries.

---

## 4. Fix 4 — `rules.py`: Keyword rules re-scan raw text

### Problem

All 5 keyword rules (`rule_sponsor_keyword`, `rule_self_promo_keyword`, `rule_recap_keyword`, `rule_intro_window`, `rule_outro_window`) iterated over `seg.text.lower()` and checked it against the keyword lists. Phase 2 (V2.1) added `matched_keywords: List[str]` to `TextFeatureSegment` specifically so that downstream consumers don't need to re-do the scan. The rules were never updated to use it.

Consequences:
- Duplicated logic: keyword lists maintained in two places
- `rules.py` and `text.py` could silently diverge if one list is updated and the other isn't

### Fix

Each rule now filters `seg.matched_keywords` by prefix instead of scanning `seg.text`:

**Old pattern (all 5 rules):**
```python
for seg in text_features.segments:
    text_lower = seg.text.lower()
    for kw in SPONSOR_KEYWORDS:
        if kw in text_lower:
            hits.append(...)
            break
```

**New pattern:**
```python
# sponsor
for seg in text_features.segments:
    sponsor_matches = [kw for kw in seg.matched_keywords if kw.startswith("sponsor:")]
    if sponsor_matches:
        s = max(0, int(np.floor(seg.start - 15)))
        e = min(T, int(np.ceil(seg.end + 15)))
        hits.append(RuleHit(s, e, "sponsorship", sponsor_matches[0], confidence=0.85))

# self_promo
for seg in text_features.segments:
    promo_matches = [kw for kw in seg.matched_keywords if kw.startswith("self_promo:")]
    if promo_matches:
        ...

# recap
for seg in text_features.segments:
    recap_matches = [kw for kw in seg.matched_keywords if kw.startswith("recap:")]
    if recap_matches:
        ...

# intro (first 90s, returns on first hit)
for seg in text_features.segments:
    if seg.start > INTRO_WINDOW_SEC:
        break
    intro_matches = [kw for kw in seg.matched_keywords if kw.startswith("intro:")]
    if intro_matches:
        e = min(T, int(np.ceil(seg.end + 10)))
        return [RuleHit(0, e, "intro", intro_matches[0], confidence=0.7)]

# outro (last 90s, returns on first hit)
for seg in text_features.segments:
    if seg.end < cutoff:
        continue
    outro_matches = [kw for kw in seg.matched_keywords if kw.startswith("outro:")]
    if outro_matches:
        s = max(0, int(np.floor(seg.start - 5)))
        return [RuleHit(s, T, "outro", outro_matches[0], confidence=0.8)]
```

The `matched_keywords` format is `"<group>:<keyword>"` (e.g. `"sponsor:sponsored by"`), set by `_match_keywords()` in `features/text.py`. The prefix filter maps exactly to each rule's group.

**Note:** The keyword constant lists (`SPONSOR_KEYWORDS`, `INTRO_KEYWORDS`, etc.) are kept in `rules.py` as the single source of truth. `text.py` imports them from here. This means adding a keyword to `rules.py` automatically propagates to both the text feature stage and the rule trigger — no more divergence risk.

---

## 5. Summary Table

| File | Fix | Priority |
|---|---|---|
| `backend/pipeline/fusion/classify.py` | Expand `CLIP_LABEL_TO_SEGMENT` from 4 → 10 entries | Critical — wrong labels without this |
| `backend/pipeline/fusion/rules.py` | `rule_dead_air` uses `rms_energy AND is_speech==0` | Quality — prevents over-firing on music |
| `backend/pipeline/fusion/boundaries.py` | Signal 5: `is_music` flip → boundary candidate | Quality — catches sponsor/intro transitions |
| `backend/pipeline/fusion/rules.py` | All 5 keyword rules read `matched_keywords` | Cleanup — eliminates duplicated scan logic |
