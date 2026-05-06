# Ad Insertion Detection Plan

## What the Reference Output Actually Is

Our `sponsorship` label detects **in-video host sponsor reads** — segments where the presenter
turns to the camera and reads an ad script. These are a production artifact of the *video itself*.

The reference `output/*.json` files describe something different: **external ad clips** (e.g.
`ads_009.mp4`) inserted at **natural seams** between large content blocks. The reference system
found natural pause points in the video and inserted a separate ad video at each one.

These are two different problems:

| Concept | Our current output | Reference JSON |
|---|---|---|
| In-video sponsor read | `sponsorship` segment | Not captured |
| External ad insertion slot | Not captured | `inserted_ads[].original_video_insert_at_seconds` |

Our pipeline already **detects the seams** — after Fixes 1–5 we hit 15/15 reference insertion
points within ±30 s using `natural_break_candidates`. What is missing is a module that **scores
and ranks** those candidates to output a short list of recommended ad slots.

---

## Reference Data Analysis

Every test video has exactly **3 ad insertions** that divide it into **4 content blocks**.

| Video | Duration | Ad insertions (s) | Flanking blocks (s, s) |
|---|---|---|---|
| test_001 | 1280 s | 106, 510, 938 | (106,404), (404,428), (428,342) |
| test_002 | 1200 s | 271, 619, 1025 | (271,348), (348,407), (407,175) |
| test_003 | 1641 s | 411, 787, 1292 | (411,377), (377,504), (504,349) |
| test_004 | 1800 s | 260, 1066, 1557 | (260,806), (806,490), (490,244) |
| test_005 | 1315 s | 151, 632, 979 | (151,481), (481,347), (347,336) |

Key observations:
- Smallest flanking block across all 15 insertions: **106 s** (test_001 left of first ad)
- Minimum inter-ad gap across all videos: **347 s** (~5.8 min)
- Each insertion aligns with a boundary already in `natural_break_candidates` (≤ 30 s)
- No insertion is adjacent to an in-video `sponsorship` segment (test_004 @ 260 s is the
  closest: our sponsorship ends at 263 s, which is 3 s *after* the ad insertion point)

---

## Algorithm Design

### Input

- `segments`: smoothed segment list from `smooth_pipeline`
- `natural_break_candidates`: list of pre-smooth boundary timestamps (seconds, floats)
- `meta_raw.duration_sec`: total video duration
- `grid`: per-second feature grid (for signal strength scoring)
- `N_ads`: number of ad slots to select (default = 3)

### Step 1 — Filter Eligible Candidates

A candidate `t` is eligible if:
1. The content block to its **left** spans ≥ `MIN_BLOCK_SEC` (= 90 s).
2. The content block to its **right** spans ≥ `MIN_BLOCK_SEC` (= 90 s).
3. It is **not** within `SPONSOR_EXCLUSION_SEC` (= 60 s) of an in-video `sponsorship` segment.

"Content block" = contiguous run of non-ad segments on that side; the block boundary is the
nearest other ad candidate that has already been selected (or the video start/end).

### Step 2 — Score Each Eligible Candidate

```
score(t) = w_signal  * signal_strength(t)
         + w_balance * balance_score(t)
         - w_sponsor * sponsor_proximity_penalty(t)
```

**signal_strength(t)** — how strong the detected break signal is at `t`:

| Signal | Value |
|---|---|
| `is_hard_cut[t]` == True | 1.0 |
| `is_speech[t-1]` == 1 and `is_speech[t]` == 0 (speech→silence) | 0.7 |
| `is_speech[t-1]` == 0 and `is_speech[t]` == 1 (silence→speech) | 0.5 |
| `hist_diff[t]` > 0.3 (scene change without hard cut) | 0.4 |
| Segment label changes at `t` | +0.2 bonus |

**balance_score(t)** — rewards candidates that split remaining content evenly:

```python
left_dur  = t - last_ad_before_t   # seconds of content to the left
right_dur = next_ad_after_t - t    # seconds of content to the right
balance_score = 1.0 - abs(left_dur - right_dur) / (left_dur + right_dur)
```

A perfect 50/50 split → 1.0; completely lopsided → 0.0.

**sponsor_proximity_penalty(t)** — reduces score if close to an in-video sponsor read:

```python
min_dist = min distance to any sponsorship segment boundary
sponsor_proximity_penalty = max(0, 1 - min_dist / SPONSOR_EXCLUSION_SEC)
```

Suggested weights (tunable): `w_signal=0.4, w_balance=0.5, w_sponsor=0.1`.

### Step 3 — Greedy Selection with Spacing Constraint

```python
MIN_AD_SPACING_SEC = 300  # ≥ 5 min between ads

selected = []
for cand in sorted(eligible_candidates, key=lambda t: -score(t)):
    if all(abs(t - s) >= MIN_AD_SPACING_SEC for s in selected):
        selected.append(cand)
    if len(selected) == N_ads:
        break
return sorted(selected)
```

This is a greedy top-score-first selection. A dynamic-programming variant (maximise minimum
block duration) is more principled but produces nearly identical results on these videos.

---

## Implementation Plan

### New File: `backend/pipeline/fusion/ad_slots.py`

```python
MIN_BLOCK_SEC       = 90.0
MIN_AD_SPACING_SEC  = 300.0
SPONSOR_EXCLUSION_SEC = 60.0
W_SIGNAL  = 0.4
W_BALANCE = 0.5
W_SPONSOR = 0.1

class AdInsertionCandidate(BaseModel):
    timestamp_sec: float
    score: float
    signal_strength: float
    balance_score: float
    left_block_sec: float
    right_block_sec: float

def score_candidates(
    segments: List[Segment],
    natural_break_candidates: List[float],
    grid: Dict[str, np.ndarray],
    duration_sec: float,
    n_ads: int = 3,
) -> List[AdInsertionCandidate]:
    ...
```

`score_candidates` returns the top-N candidates sorted by timestamp, ready for export.

### Schema Addition: `schemas.py`

```python
class AdInsertionCandidate(BaseModel):
    timestamp_sec: float = Field(..., description="Recommended ad insertion point in seconds")
    score: float          = Field(..., description="Composite score 0–1")
    signal_strength: float
    balance_score: float
    left_block_sec: float
    right_block_sec: float

class Metadata(BaseModel):
    ...
    ad_insertion_candidates: List[AdInsertionCandidate] = Field(
        default_factory=list,
        description="Top-N recommended external ad insertion points, scored and ranked.",
    )
```

### Export: `fusion/export.py`

```python
from backend.pipeline.fusion.ad_slots import score_candidates

def export_metadata(workspace, meta_raw, segments,
                    raw_boundaries=None, grid=None, n_ads=3):
    ...
    ad_candidates = score_candidates(
        segments, break_candidates, grid, meta_raw.duration_sec, n_ads
    ) if grid else []

    metadata = Metadata(
        ...,
        natural_break_candidates=break_candidates,
        ad_insertion_candidates=ad_candidates,
    )
```

### Pipeline: `fusion/run.py`

Pass `grid` to `export_metadata`:

```python
out_path = export_metadata(workspace, meta_raw, segments,
                           raw_boundaries=raw_boundaries, grid=grid)
```

### Frontend

In the timeline player, render ad slot markers as vertical orange dashed lines at each
`ad_insertion_candidates[i].timestamp_sec`. Show a tooltip with score and flanking block
durations on hover.

In the segment table add an **"Ad Slots"** section below the segment list showing the ranked
candidates, their scores, and left/right content block durations.

---

## Testing Plan

**`tests/test_ad_slots.py`**

| Test | What it checks |
|---|---|
| `test_candidate_below_min_block_excluded` | Candidate whose left content block < 90 s is filtered out |
| `test_sponsor_adjacent_candidate_penalized` | Candidate within 60 s of sponsorship segment has reduced score |
| `test_spacing_constraint_enforced` | Two top-scoring candidates < 300 s apart — only the higher-scored one is selected |
| `test_hard_cut_scores_higher_than_speech_flip` | Hard-cut boundary beats speech-flip boundary at same balance score |
| `test_balance_score_prefers_equal_split` | 50/50 split beats 80/20 split when signal strengths are equal |
| `test_n_ads_respected` | Returns exactly N candidates (or fewer if insufficient eligible points) |
| `test_empty_candidates_returns_empty` | `score_candidates([])` returns `[]` without error |
| `test_integration_metadata_json_has_field` | End-to-end: exported `metadata.json` contains `ad_insertion_candidates` list |

---

## Expected Results

Given that our pipeline already hits all 15 reference insertions within ±30 s via
`natural_break_candidates`, the ad-slot scorer just needs to pick the right 3 from among ~15
candidates per video. Targeting:

| Metric | Baseline | After this plan |
|---|---|---|
| Ad slots matching reference (±30 s) | 0/15 | **≥ 12/15** |
| Ad slots matching reference (±60 s) | 0/15 | **15/15** |
| Exported field in `metadata.json` | absent | `ad_insertion_candidates` always present |

The balance-score term is the dominant driver: in every reference video the ads are placed
roughly where they divide the content into equal-duration blocks. Signal strength breaks ties
between candidates at similar balance positions.

---

## Files to Create / Change

| File | Change |
|---|---|
| `backend/pipeline/fusion/ad_slots.py` | **New** — `score_candidates`, `AdInsertionCandidate` |
| `backend/pipeline/schemas.py` | Add `AdInsertionCandidate` model; add `ad_insertion_candidates` to `Metadata` |
| `backend/pipeline/fusion/export.py` | Accept `grid` parameter; call `score_candidates`; populate field |
| `backend/pipeline/fusion/run.py` | Pass `grid` to `export_metadata` |
| `tests/test_ad_slots.py` | **New** — 8 unit tests |
| `frontend/` (player JS) | Render dashed orange markers + tooltip at candidate timestamps |

---

## Implementation Order

1. `schemas.py` — add `AdInsertionCandidate` and the `Metadata` field (no logic, safe start)
2. `ad_slots.py` — implement `score_candidates` with the three score components
3. `export.py` + `run.py` — wire it into the pipeline
4. Tests — confirm filtering, scoring, spacing, integration
5. Frontend — add timeline markers and table section
