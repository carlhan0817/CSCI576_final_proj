# Multi-Signal Ad Detection — test_004_2 Results

Source video: `workspace/test_004_2/test_004_2.mp4` (1936s ≈ 32 min)
Ground-truth ad regions: Ad 1 = 260–290s, Ad 2 = 1096–1142s, Ad 3 = 1632–1692s.

## Sponsorship coverage by region

| Region | GT range | Old (rule_ad_break + bridges) | New (rule_ad_block + advisory ad_break) |
|---|---|---|---|
| Ad 1 | 260–290 | 3s (seg #5, conf 0.45) | _pending re-run_ |
| Ad 2 | 1096–1142 | 47s (seg #11, conf 0.85) | _pending re-run_ |
| Ad 3 | 1632–1692 | 33s combined (#17 + #23) | _pending re-run_ |
| FP count outside GT | — | 4 (seg #3, #9, #13, #26) | _pending re-run_ |

## BEFORE snapshot

Captured pre-rerun from `metadata.json` produced under the old single-signal pipeline:

| seg # | start–end (s) | confidence | triggered rules |
|---|---|---|---|
| 3   | 218–227   | 0.85 | (smoothing-only)            |
| 5   | 260–263   | 0.45 | (smoothing-only)            |
| 9   | 503–528   | 0.83 | sponsorship_bridge          |
| 11  | 1095–1142 | 0.85 | ad_break, holding_screen    |
| 13  | 1169–1202 | 0.81 | sponsorship_bridge          |
| 17  | 1633–1654 | 0.49 | sponsorship_bridge          |
| 23  | 1676–1688 | 0.53 | sponsorship_bridge          |
| 26  | 1806–1863 | 0.84 | (smoothing-only)            |

True positives: #5 (Ad 1, partial 3s), #11 (Ad 2, full 47s), #17+#23 (Ad 3, 33s combined).
False positives: #3 (218–227, lecture region), #9 (503–528, lecture region), #13 (1169–1202, post-Ad-2 lecture), #26 (1806–1863, lecture region).

## AFTER snapshot

_To be filled in after `scripts/rerun_test_004_2.py` completes._

## OCR signals fired during ads

_To be filled in after the re-run by inspecting `visual_features.json` per-frame OCR flags._

## Notes

- New pipeline relies on `rule_ad_block` (style_drift OR audio_drift, ≥10s sustained, with one commercial signal) plus the demoted `rule_ad_break` (now confidence 0.5, advisory only).
- New signals: pooled CLIP embedding (Phase 2 visual), per-second 20-dim MFCC (Phase 2 audio), per-frame RapidOCR text + commercial-pattern flags (Phase 2 visual), transcript-side `cta:*` keyword matches (Phase 2 text).
- See `docs/superpowers/plans/2026-05-04-multimodal-ad-detection.md` for the implementation plan.
