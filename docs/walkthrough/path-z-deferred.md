# Path Z — deferred items from the Path X (block_drift) rewrite

Path X rewrote `rule_ad_block` to use sustained `block_drift` signals
(`style_block_drift`, `audio_block_drift`) instead of the boundary-spike
`drift` pair. See `backend/pipeline/fusion/style_drift.py::compute_block_drift`
for the math and `scripts/diagnose_ad_block.py` for the empirical evidence.

This file lists the four issues that surfaced during Path X but were
**deliberately not fixed** in the same change, so future work has a clear
starting point.

---

## Z-1. Ads ≥ 120 seconds

**Symptom.** `test_001 ad #1` (`ads_009.mp4`, 119 s, final-video window
`[106, 225)`) is not detected by any block_drift configuration.

**Why.** `compute_block_drift` compares each second against context in
`[t-far_window, t-near_skip) ∪ (t+near_skip, t+far_window]`. Inside an ad
that is longer than `2 * far_window` seconds, both halves of that context
are *also inside the ad*, so the cosine distance collapses toward zero and
the second is no longer flagged as anomalous. Path X uses
`audio: near=30 far=90`, which can tolerate ads up to ≈120 s; beyond that
the signal stops being sustained.

**Why we didn't push `far_window` higher.** Empirically, growing
`far_window` past 90 s degrades baseline separation on `test_001` because
60–90 s lecture segments have non-trivial intra-style variation. The
`near_skip` / `far_window` knobs cannot be made arbitrarily large without
trading recall on long ads for precision elsewhere.

**Plausible directions when this becomes blocking.**
- Adaptive context: detect candidate ad windows first (e.g., from
  `low_energy_audio` or `ad_break`), then re-compute drift with
  `near_skip = candidate_length / 2`.
- Use a *global* lecture baseline (mean of all "non-candidate" seconds)
  rather than a local context window. This was rejected for Path X
  because the baseline depends on first identifying which seconds are
  non-ad — circular without iteration.
- Train a classifier on (audio_block_drift, hard_cut_density,
  rms_energy, …) features rather than thresholding a single signal.

---

## Z-2. Argmax仲裁: structural rules outrank weak ad rules

**Symptom (疑点 1).** Inside `classify._resolve_rule_label_for_segment`,
the segment label is the rule with the highest `confidence` covering it
(coverage ≥ 0.25). When `holding_screen` (0.70) and `ad_block`
visual-only path (0.60) both cover a segment, `holding_screen` wins. The
ad ends up labeled `holding_screen` rather than `sponsorship`, even
though both rules' triggers are recorded in `evidence.triggered_rules`.

**Why we didn't fix it in Path X.** The audio-only path of the new
`rule_ad_block` fires at confidence 0.80 — already above
`holding_screen` 0.70 — so test_004 Ad #2 (the example case in the
diagnostic) is correctly merged into a single `sponsorship` segment
without touching the argmax. The visual-only path at 0.60 *does* still
lose to `holding_screen`, but that path is intended as a corroborating
signal, not a primary detection, and over-bumping its confidence would
make test_001 lecture-content visual variability fire FPs.

**The clean fix.** Replace the argmax with a label-class hierarchy:
sponsorship-class rules (ad_block, low_energy_audio, ad_break) should
preempt structural rules (holding_screen, dead_air) when both fire on
the same segment, regardless of confidence. Structural labels then
become a fallback rather than a competitor.

---

## Z-3. CLIP-only short sponsorship segments + isolated short segments

**Symptom (疑点 2 + 疑点 4).** Step 4b in `classify.py` falls back to
"strongest CLIP probability" when no rule covers a segment. For 3-second
slide-change moments where CLIP recognises a corporate logo, this paints
the segment `sponsorship` with no `triggered_rules`. The cut-density
demotion in `classify.py:133-136` blocks the obvious lecture-slide case
(<0.25 cuts/s) but lets through 3-second segments containing one cut
(density 0.5 ≥ 0.25). After smoothing, these survive because
`MIN_SEGMENT_DURATION = 2.0` s.

Concrete examples in `workspace/test_004/metadata.json` after Path X:
seg 3 (218–221), seg 11 (1169–1172), seg 13 (1199–1202).

**Why we didn't fix it.** Out of scope for Path X (which is about ad
detection signal, not CLIP-fallback behaviour). The right fix is one
small extra guard in either `classify.py` or `smooth.py`:

```
if segment.label in {"sponsorship", "self_promotion"}
   and segment.evidence.triggered_rules == []
   and (segment.end_sec - segment.start_sec) < 10:
       absorb into the higher-confidence neighbour
```

This handles isolation (疑点 4) and the short-no-rule case (疑点 2)
together, without touching duration thresholds in a way that hurts
real short ad stingers (which carry `triggered_rules`).

---

## Z-4. `low_energy_audio` paints most of `test_001` as sponsorship

**Symptom.** Re-running fusion on `videos/test_001.mp4` produces 5
top-level segments where four of them are `sponsorship`. The cause is
`rule_low_energy_audio` firing across the entire lecture (≈70 % of
seconds) because `test_001`'s recording has lower RMS / narrower
spectral bandwidth than the test_004 talk this rule was originally
tuned on.

**Why this was not Path X's job.** Path X did not change
`rule_low_energy_audio`. The ad_block additions in Path X correctly
appear as `triggered_rules` on the segments containing the GT ads, so
downstream "find segments where ad_block is in triggered_rules" tooling
works. The over-broad `low_energy_audio` trigger is a separate
precision problem on `test_001` audio.

**Direction.** Make `LOW_ENERGY_RMS_MAX` and `LOW_ENERGY_BW_MAX` either
adaptive (e.g., percentiles of the per-video distribution) or revisit
whether `low_energy_audio` should use `audio_block_drift` directly
instead of raw RMS/bandwidth thresholds.

---

## Quick map: original four "疑点" → which Z item resolves them

| 疑点 | Z item | Notes |
|---|---|---|
| 1. Argmax仲裁 | Z-2 | Visual-only path still loses to holding_screen at 0.60 vs 0.70; audio path safely wins at 0.80 |
| 2. CLIP-only short sponsorship | Z-3 | Fix together with Z-3 |
| 3. ad_block召回外包给 OCR | **resolved in Path X** | `_has_any_commercial_signal` removed from the rule |
| 4. absorb 阈值 | Z-3 | Replace size-based absorb with isolation-based absorb |

| Path X validation finding | Z item |
|---|---|
| 119 s ad detection | Z-1 |
| `low_energy_audio` over-trigger on `test_001` | Z-4 |
