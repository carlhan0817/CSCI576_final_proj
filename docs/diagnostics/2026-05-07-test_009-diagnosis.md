# test_009 fusion diagnosis (2026-05-07)

Read-only diagnostic. Determines whether the high error rate on test_009 (vlog with music B-roll) stems from one dominant signal or several rules collectively misfiring. No backend code modified; grid recomputed from phase-2 features via `align.build_per_second_grid`.

## Headline

- GT ad seconds: **116**
- Predicted sponsorship seconds: **181**
- TP=93  FP=88  FN=23
- Precision=0.514  Recall=0.802

## Q1 — Rule attribution

| rule | fired_total | TP | FP | top FP runs |
|---|---:|---:|---:|---|
| ad_block | 78 | 30 | 48 | 384–408(24s), 182–200(18s), 705–711(6s) |
| vlog_speech_burst | 125 | 92 | 33 | 182–215(33s) |
| holding_screen | 25 | 0 | 25 | 662–675(13s), 149–161(12s) |
| dead_air | 8 | 0 | 8 | 42–50(8s) |

FN attribution (GT=ad but predicted!=sponsorship — 23 sec total):
- *no rule fired on any GT-ad second that we missed* — i.e. the missed ads (kelloggs) were silent at the rule layer.

**Verdict (mechanical):** top rule `ad_block` accounts for 42% of rule-attributed FP-seconds → **multi-factor**.

## Q2 — Signal distribution across populations

Per-second percentiles of drift / energy / speech over four mutually exclusive groups.

| group | n_sec | audio_drift p15/p50/p85 | visual_drift p15/p50/p85 | rms p50 | speech_frac |
|---|---:|---|---|---:|---:|
| A_TP_ad_correct | 93 | 0.86/1.09/1.44 | 0.29/0.36/0.44 | 0.099 | 0.94 |
| B_FN_ad_missed | 23 | 1.03/1.16/1.50 | 0.24/0.36/0.39 | 0.084 | 0.52 |
| C_FP_content_painted_sp | 88 | 0.99/1.33/1.51 | 0.19/0.28/0.35 | 0.093 | 0.73 |
| D_TN_content_correct | 189 | 0.84/1.17/1.40 | 0.21/0.28/0.35 | 0.093 | 0.62 |

## Q3 — Kelloggs ad per-second snapshot (58–92s)

GT ad is 60.0–90.016s. Looking for the signals that should but don't fire.

| t | gt | pred | rules | spk | rms | audio_drift | vis_drift | hard_cut | OCR | transcript |
|---:|---|---|---|---:|---:|---:|---:|---:|---|---|
| 58 | content | filler | — | 0 | 0.010 | 1.01 | 0.25 | 0 | — | We'll shed here, bringing fiber to the masses with Kellogg's |
| 59 | content | core_content | — | 0 | 0.093 | 0.79 | 0.23 | 0 | — | We'll shed here, bringing fiber to the masses with Kellogg's |
| 60 | ad | core_content | — | 0 | 0.055 | 1.15 | 0.24 | 1 | — | We'll shed here, bringing fiber to the masses with Kellogg's |
| 61 | ad | core_content | — | 1 | 0.065 | 0.80 | 0.43 | 1 | — | We'll shed here, bringing fiber to the masses with Kellogg's |
| 62 | ad | core_content | — | 1 | 0.072 | 1.10 | 0.39 | 0 | — | We'll shed here, bringing fiber to the masses with Kellogg's |
| 63 | ad | core_content | — | 1 | 0.057 | 1.18 | 0.36 | 0 | — | We'll shed here, bringing fiber to the masses with Kellogg's |
| 64 | ad | core_content | — | 1 | 0.032 | 1.10 | 0.37 | 0 | — | We'll shed here, bringing fiber to the masses with Kellogg's |
| 65 | ad | sponsorship | vlog_speech_burst | 1 | 0.032 | 1.15 | 0.42 | 1 | — | We'll shed here, bringing fiber to the masses with Kellogg's |
| 66 | ad | sponsorship | vlog_speech_burst | 0 | 0.044 | 0.96 | 0.42 | 0 | — | Duty calls. |
| 67 | ad | sponsorship | vlog_speech_burst | 1 | 0.044 | 0.88 | 0.46 | 0 | — | Duty calls. |
| 68 | ad | sponsorship | vlog_speech_burst,ad_block | 1 | 0.031 | 0.85 | 0.46 | 1 | — | We'll shed? |
| 69 | ad | sponsorship | vlog_speech_burst,ad_block | 0 | 0.024 | 0.86 | 0.38 | 1 | — | Every darn day. |
| 70 | ad | sponsorship | vlog_speech_burst,ad_block | 1 | 0.033 | 1.47 | 0.43 | 0 | — | Every darn day. |
| 71 | ad | sponsorship | vlog_speech_burst,ad_block | 1 | 0.028 | 1.58 | 0.42 | 0 | — | Whoa. |
| 72 | ad | sponsorship | vlog_speech_burst,ad_block | 1 | 0.018 | 1.64 | 0.43 | 0 | — | We'll shed in the house. |
| 73 | ad | sponsorship | vlog_speech_burst,ad_block | 1 | 0.018 | 1.56 | 0.43 | 0 | — | It's fiber time. |
| 74 | ad | sponsorship | vlog_speech_burst,ad_block | 0 | 0.019 | 1.55 | 0.45 | 1 | — | Is that dog a shit son? |
| 75 | ad | sponsorship | vlog_speech_burst,ad_block | 1 | 0.055 | 1.56 | 0.39 | 1 | — | Perfect. |
| 76 | ad | sponsorship | vlog_speech_burst,ad_block | 1 | 0.033 | 1.52 | 0.47 | 0 | — | We'll shed on the car. |
| 77 | ad | sponsorship | vlog_speech_burst,ad_block | 1 | 0.015 | 1.17 | 0.44 | 0 | — | We'll too old for this shit. |
| 78 | ad | sponsorship | vlog_speech_burst,ad_block | 1 | 0.007 | 1.25 | 0.42 | 0 | — | Kellogg's raisin bread. |
| 79 | ad | sponsorship | vlog_speech_burst,ad_block | 1 | 0.046 | 1.54 | 0.46 | 0 | — | High fiber, happy go. |
| 80 | ad | sponsorship | vlog_speech_burst | 1 | 0.010 | 1.17 | 0.48 | 1 | — | — |
| 81 | ad | sponsorship | vlog_speech_burst | 1 | 0.011 | 1.08 | 0.45 | 0 | — | — |
| 82 | ad | sponsorship | vlog_speech_burst | 1 | 0.045 | 1.11 | 0.40 | 1 | — | — |
| 83 | ad | sponsorship | vlog_speech_burst | 1 | 0.047 | 1.19 | 0.45 | 1 | — | — |
| 84 | ad | sponsorship | vlog_speech_burst | 1 | 0.053 | 1.47 | 0.42 | 1 | — | — |
| 85 | ad | sponsorship | vlog_speech_burst | 1 | 0.077 | 1.23 | 0.35 | 0 | — | — |
| 86 | ad | sponsorship | vlog_speech_burst | 1 | 0.084 | 1.24 | 0.50 | 1 | — | — |
| 87 | ad | sponsorship | vlog_speech_burst | 1 | 0.056 | 1.09 | 0.54 | 0 | — | — |
| 88 | ad | sponsorship | vlog_speech_burst | 1 | 0.091 | 0.73 | 0.36 | 1 | — | — |
| 89 | ad | sponsorship | vlog_speech_burst | 1 | 0.105 | 0.78 | 0.30 | 0 | — | — |
| 90 | content | filler | — | 0 | 0.111 | 0.57 | 0.35 | 1 | — | — |
| 91 | content | filler | — | 0 | 0.114 | 0.61 | 0.32 | 0 | — | — |
| 92 | content | filler | — | 0 | 0.083 | 0.38 | 0.33 | 0 | — | — |

## Q4 — Paired region compare

True ad regions vs adjacent FP regions. Same column = same statistic; compare row-pairs to see whether GT-ad and FP-content separate in any feature dimension.

| region | len | audio_drift mean | visual_drift mean | rms p50 | speech_frac | hard_cut/sec | OCR_comm hits |
|---|---:|---:|---:|---:|---:|---:|---:|
| ramp_ad_TP_330_361 | 31 | 1.17 | 0.39 | 0.125 | 0.74 | 0.45 | 0 |
| post_ramp_FP_363_456 | 93 | 1.31 | 0.25 | 0.128 | 0.16 | 0.39 | 0 |
| kelloggs_ad_FN_60_90 | 30 | 1.20 | 0.42 | 0.044 | 0.87 | 0.43 | 0 |
| post_kelloggs_FP_90_200 | 110 | 1.06 | 0.29 | 0.119 | 0.30 | 0.39 | 0 |

## Q5 — "Music" transcript placeholder vs sponsor_keyword

- Segments whose text starts with "Music": **3** (34 sec total)
- Of those, segments with a `sponsor:*` matched keyword: **0**
- Total `sponsor:*` keyword hits across the entire transcript: **0**

**Verdict:** the "Music" placeholder does NOT feed `rule_sponsor_keyword` in this video. The high text_score on FP segments must come from another channel (probably the keyword-free 'has_text' axis or a different rule).
