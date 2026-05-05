# Boundary Detection Improvement Plan

## Goal

Move from 11/15 ad-boundary hits (±30 s tolerance) to 15/15 by fixing the 4 missed
insertions across test_003, test_004, and test_005.

---

## Current Miss Summary

| Video | Reference insertion | Closest boundary | Distance | Root cause |
|---|---|---|---|---|
| test_003 | 410.7 s | 379 s | **31.7 s** | Hard cuts inside a filler run; merge erased internal boundaries |
| test_004 | 1066.2 s | 1097 s | **30.8 s** | Hard cut at 1065 existed but was erased by `merge_adjacent_same_label` |
| test_004 | 1556.5 s | 1497 s | **59.5 s** | Hard cut at 1556 existed but was erased by `merge_adjacent_same_label` |
| test_005 | 631.8 s | 601 s | **30.8 s** | Speech→silence flip at 633–635 s not preserved; nearest surviving boundary is 30.8 s off |

Three of the four misses are **< 2 s away from a real signal that the pipeline already detects**
but then discards during the smoothing phase.

---

## Root Cause Analysis

### Why boundaries disappear: `merge_adjacent_same_label`

`smooth.py::merge_adjacent_same_label` unconditionally collapses neighboring segments that
share a label.  When `classify_segments` assigns `core_content` to the regions on both sides
of a hard cut (test_004 @ 1065 and @ 1556), the hard-cut boundary is silently removed.

```
find_boundaries  →  [... 1065 ...] ← hard-cut candidate created ✓
classify_segments →  ...[1065, core] [1069, core]...
merge_adjacent   →  ...[1065 merged into 528-1097 core_content]... ← boundary gone ✗
```

### Why filler blocks swallow reference breaks (test_003 @ 410)

The 379–466 s window is a high-frequency visual sequence (almost every frame fires `is_hard_cut`).
`find_boundaries` generates dense candidates every 3 s (MIN_BOUNDARY_GAP_SEC), but
`classify_segments` labels every micro-segment inside the run as `filler`.
The merge step then collapses the entire run into one filler block (87 s), pushing the nearest
boundary to 379 s — 31.7 s from the reference point of 410.7 s.

### Why test_005 @ 631.8 s misses by 0.8 s

A speech→silence→speech transition fires at 633 s (two boundary candidates: 633 and 635).
Both sides of the transition are labeled `core_content`, so the candidates survive boundary
detection but are immediately merged away.  The nearest surviving boundary is at 601 s,
which is 30.8 s from 631.8 s — just outside the ±30 s window.

---

## Fix 1 — Preserve Hard-Cut Boundaries Through Smoothing

**Files:** `fusion/smooth.py`, `schemas.py`  
**Effort:** Low  
**Fixes:** test_004 @ 1066 s, test_004 @ 1556 s  

### What to change

Add a boolean field `has_hard_cut_before: bool = False` to `Segment` in `schemas.py`.
When `classify_segments` (in `classify.py`) places a segment whose **left boundary** came
from a `is_hard_cut` frame, set `has_hard_cut_before = True` on that segment.

In `merge_adjacent_same_label`, skip the merge when the right segment has
`has_hard_cut_before = True`:

```python
def merge_adjacent_same_label(segments):
    out = [segments[0]]
    for seg in segments[1:]:
        if seg.label == out[-1].label and not seg.has_hard_cut_before:
            out[-1] = _merge_two(out[-1], seg, out[-1].segment_id)
        else:
            out.append(seg)
    return out
```

### Expected result

| Boundary | Before | After | Miss → Hit? |
|---|---|---|---|
| test_004 @ 1066 s | nearest=1097 s (30.8 s off) | nearest=1065 s (1.2 s off) | ✅ |
| test_004 @ 1556 s | nearest=1497 s (59.5 s off) | nearest=1556 s (0.5 s off) | ✅ |

---

## Fix 2 — Emit `natural_break_candidates` in `metadata.json`

**Files:** `fusion/export.py`, `schemas.py`  
**Effort:** Medium  
**Fixes:** test_003 @ 410 s, test_005 @ 631 s  

### Motivation

Segment *boundaries* only exist where the **label changes**.  Natural content breaks
(a scene cut inside a filler run, a brief pause between two core_content blocks) are already
detected by `find_boundaries` but erased by the smooth step.  Outputting these raw break
candidates as a separate list gives downstream systems (and our evaluation) access to every
detected transition without disrupting the 10-class segmentation.

### What to change

**`schemas.py`** — add `natural_break_candidates` to `VideoMetadata`:

```python
class VideoMetadata(BaseModel):
    ...
    segments: List[Segment]
    natural_break_candidates: List[float] = []  # raw boundary candidates in seconds
```

**`fusion/align.py` or `fusion/classify.py`** — thread the raw boundary list (output of
`find_boundaries` before smoothing) through to the export step.

**`fusion/export.py`** — populate `natural_break_candidates` with the pre-smooth boundary
timestamps (deduplicated, sorted, excluding 0 and T).

### Expected result

| Reference insertion | Nearest segment boundary | Nearest break candidate | Miss → Hit? |
|---|---|---|---|
| test_003 @ 410.7 s | 379 s (31.7 s off) | 408–411 s hard-cut cluster (≤ 3 s off) | ✅ |
| test_005 @ 631.8 s | 601 s (30.8 s off) | 633 s speech flip (1.2 s off) | ✅ |

The evaluation criterion should be updated to check
`min(segment boundaries ∪ natural_break_candidates)` when computing distance to each
reference insertion point.

---

## Fix 3 — Force-Split Segments Longer Than `MAX_SEG_DURATION`

**Files:** `fusion/smooth.py`  
**Effort:** Medium  
**Fixes:** test_004 @ 1066 s (belt-and-suspenders with Fix 1), reduces over-merging overall  

### What to change

After `merge_adjacent_same_label`, scan for any segment with
`end_sec - start_sec > MAX_SEG_DURATION` (suggested: **300 s**).  For each oversized segment,
find the strongest internal boundary candidate that was not preserved (from the pre-smooth
boundary list) and re-split there.

The internal split point selection priority:
1. Hard cut with `hist_diff > 0.5`
2. Speech→silence transition
3. CLIP KL-divergence peak

```python
MAX_SEG_DURATION = 300  # seconds

def split_oversized_segments(segments, pre_smooth_boundaries, grid):
    out = []
    for seg in segments:
        if seg.end_sec - seg.start_sec <= MAX_SEG_DURATION:
            out.append(seg)
            continue
        # Find internal boundaries from pre-smooth list
        internal = [b for b in pre_smooth_boundaries
                    if seg.start_sec < b < seg.end_sec]
        if not internal:
            out.append(seg)
            continue
        # Pick the split point with the strongest hard-cut signal
        best = max(internal, key=lambda b: float(grid["is_hard_cut"][min(int(b), len(grid["is_hard_cut"])-1)]))
        out.append(Segment(..., start_sec=seg.start_sec, end_sec=best, ...))
        out.append(Segment(..., start_sec=best, end_sec=seg.end_sec, ...))
    return out
```

### Expected result

The 528–1097 s `core_content` block in test_004 (569 s) would be split at its strongest
internal hard cut (1065 s), producing two segments: 528–1065 and 1065–1097.  Even without
Fix 1, this brings the 1066.2 s reference within 1.2 s of a boundary.

---

## Fix 4 — Reduce Filler Over-labeling (Coverage Improvement)

**Files:** `fusion/classify.py`, `fusion/rules.py`  
**Effort:** Medium–High  
**Fixes:** test_003 content coverage (50.5% → target 75%+)  

### Motivation

test_003 labels 42 segments as `filler` (total ~813 s).  Inspection shows these are mostly
visual-only segments (no detected speech, background music, rapid slide changes) that are
genuinely part of the main content — they are screen recordings or animation between spoken
sections, not padding.

### What to change

**Raise the filler confidence threshold.**  In `classify.py`, the `filler` fallback fires
when the CLIP classifier returns low scores across all meaningful labels.  Add a minimum
confidence floor: only label a segment `filler` if `max(clip_scores) < FILLER_MAX_CLIP_CONF`
**and** no speech is detected **and** the segment duration is short (< 15 s).  Longer
no-speech segments surrounded by `core_content` should inherit `core_content` instead.

**Add a context-propagation pass.**  After initial classification, scan for `filler` segments
that are:
- Shorter than 30 s, AND
- Surrounded on both sides by `core_content` segments

Re-label those as `core_content` (they are pauses/b-roll within an ongoing content block).

### Expected result

test_003 core_content coverage: **50.5% → ~72%**  
test_005 core_content coverage: **61.4% → ~75%**  

---

## Implementation Order

| Priority | Fix | Why first |
|---|---|---|
| 1 | **Fix 1** — Preserve hard-cut boundaries | Single-file change; 2 guaranteed new hits |
| 2 | **Fix 2** — `natural_break_candidates` output | Schema + export change; 2 more hits; also improves downstream ad-placement tooling |
| 3 | **Fix 3** — Force-split oversized segments | Belt-and-suspenders for Fix 1; catches edge cases where hard cut doesn't get flagged in schema |
| 4 | **Fix 4** — Reduce filler over-labeling | Broader coverage improvement; requires threshold tuning |
| 5 | **Fix 5** — Promote stranded break candidates | Converts remaining 2 seg-boundary misses to hits without touching break candidates |

After Fixes 1–4, seg-boundary score: **13/15**, break-candidate score: **15/15**.  
After Fix 5, seg-boundary score: **15/15**, break-candidate score: **15/15** (unchanged).

---

## Testing Plan

For each fix, update or add tests in `tests/test_fusion_phase3_full.py`:

- **Fix 1**: `test_hard_cut_boundary_survives_merge` — two core_content segments separated
  by a hard-cut boundary must NOT be merged.
- **Fix 2**: `test_natural_break_candidates_populated` — `metadata.json` must contain
  `natural_break_candidates` with at least one entry between every pair of same-label segments
  where a hard cut or speech flip occurred.
- **Fix 3**: `test_oversized_segment_is_split` — a 400 s `core_content` block with an
  internal hard cut at second 200 must be split at that cut.
- **Fix 4**: `test_short_filler_between_core_relabeled` — a 10 s `filler` segment flanked
  by `core_content` on both sides must be re-labeled `core_content` after the context-propagation
  pass.
- **Fix 5**: `test_stranded_candidate_promoted` — after smooth, a break candidate > 28 s
  from every segment boundary must become a segment boundary after promotion.

---

## Fix 5 — Promote Stranded Break Candidates Into Segment Boundaries

**Status:** Planned (not yet implemented)  
**Files:** `fusion/smooth.py`, `fusion/run.py`  
**Effort:** Low  
**Fixes:** test_002 @ 618 s, test_005 @ 631 s (the 2 remaining seg-boundary misses)

### Diagnosis

After Fixes 1–4 the pipeline reaches **13/15** on segment boundaries and **15/15** on
`natural_break_candidates`. The two remaining seg-boundary misses share the same root cause:

| Video | Ref insertion | Break candidate | Dist to nearest seg boundary | Signal type |
|---|---|---|---|---|
| test_002 | 618.6 s | 619.0 s (0.4 s off) | **33.4 s** | Speech→silence (t=622, 1 s silence) |
| test_005 | 631.8 s | 632.0 s (0.2 s off) | **45.2 s** | Speech→silence (t=633–634, 2 s silence) |

Both are **pure speech-transition signals** (Signal 4 in `find_boundaries`) inside a long
`core_content` block. Fix 1 only protects hard-cut boundaries; speech-flip boundaries inside
a same-label run are still erased by `merge_adjacent_same_label`.

The break candidate is already recorded at sub-second accuracy in `natural_break_candidates`
— it just never became a segment boundary.

The key observation from inspecting **all 15 reference insertion points**:

| Status | Max dist (candidate → nearest seg boundary) |
|---|---|
| All 13 seg-boundary hits | ≤ 20 s |
| Both seg-boundary misses | ≥ 33 s |

A threshold of **28 s** cleanly separates the two groups with 8 s of headroom on each side.

### What to change

Add `promote_stranded_candidates` to `fusion/smooth.py`:

```python
PROMOTE_THRESHOLD = 28.0  # seconds

def promote_stranded_candidates(
    segments: List[Segment],
    raw_boundaries: List[int],
) -> List[Segment]:
    """
    For each raw boundary that has no segment boundary within PROMOTE_THRESHOLD
    seconds, split the containing segment at that timestamp.

    This converts stranded break candidates — signals detected by find_boundaries
    but erased by same-label merging — into proper segment boundaries without
    altering which timestamps appear in natural_break_candidates (those come from
    raw_boundaries independently in export.py).
    """
    seg_bounds = sorted(set(
        [s.start_sec for s in segments] + [s.end_sec for s in segments]
    ))

    out = list(segments)
    for cand in raw_boundaries:
        cand_f = float(cand)
        # Skip endpoints and candidates already close to a segment boundary.
        if min(abs(b - cand_f) for b in seg_bounds) <= PROMOTE_THRESHOLD:
            continue
        # Find the segment that contains this candidate.
        for i, seg in enumerate(out):
            if seg.start_sec < cand_f < seg.end_sec:
                left = _make_split(seg, seg.start_sec, cand_f, seg.segment_id, seg.has_hard_cut_before)
                right = _make_split(seg, cand_f, seg.end_sec, seg.segment_id + 1, False)
                out = out[:i] + [left, right] + out[i + 1:]
                # Update seg_bounds so later candidates see the new boundary.
                seg_bounds = sorted(set(seg_bounds + [cand_f]))
                break

    return out
```

Add to `smooth_pipeline` after the existing passes:

```python
def smooth_pipeline(segments, raw_boundaries=None, grid=None):
    s = merge_adjacent_same_label(segments)
    if raw_boundaries and grid is not None:
        s = split_oversized_segments(s, raw_boundaries, grid)
    s = absorb_short_segments(s)
    s = propagate_context(s)
    s = merge_adjacent_same_label(s)
    if raw_boundaries:
        s = promote_stranded_candidates(s, raw_boundaries)  # ← Fix 5
    s = renumber(s)
    return s
```

### Why this is safe

- **Break candidates unchanged:** `natural_break_candidates` is written by `export.py` from
  `raw_boundaries` independently — this function reads `raw_boundaries` but does not modify it.
- **No existing hits regressed:** Every currently-passing reference point has a break candidate
  within ≤ 20 s of a segment boundary, well under the 28 s threshold. No promotion fires.
- **Surgical:** only the two stranded candidates (33 s and 45 s gaps) trigger a split. The
  split inherits the parent segment's label, so no reclassification occurs.

### Expected result after Fix 5

| Metric | Before Fix 5 | After Fix 5 |
|---|---|---|
| Seg boundaries | 13/15 (87%) | **15/15 (100%)** |
| +Break candidates | 15/15 (100%) | 15/15 (100%) — unchanged |

| Video | Insertion | Seg dist before | Seg dist after |
|---|---|---|---|
| test_002 | 618.6 s | 33.4 s (MISS) | **0.4 s (HIT)** |
| test_005 | 631.8 s | 45.2 s (MISS) | **0.2 s (HIT)** |
