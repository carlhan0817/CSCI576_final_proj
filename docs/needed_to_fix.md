# Phase 2 & Phase 3 — Needed Fixes

Gaps between the current implementation (commits `d85011c phase2` and `2fec079 phase 3`) and PDD §4.2 / §4.3. Ordered by impact.

---

## Repo hygiene (fix first — blocks everything else)

- [ ] **Rename `backend/pipeline/features ` → `backend/pipeline/features`.** The directory currently has a literal trailing space, breaking `from backend.pipeline.features.visual import …` on most setups.
- [ ] **Remove `get-pip.py`** (27,506 lines accidentally committed in the phase 3 commit). Add to `.gitignore`.
- [ ] **Fix the empty test file** `tests/test_audio_phase2.py` (currently 0 bytes) and replace the print-only stubs in `tests/test_visual.py` and `tests/test_text.py` with real assertions.
- [ ] **Update `docs/walkthrough/phase2.md`** — example uses `Workspace.for_video(mp4_path)` but the actual signature requires a `root=` argument.

---

## §4.2 — Multimodal Feature Extraction

### 4.2.1 Visual ([backend/pipeline/features /visual.py](../backend/pipeline/features%20/visual.py))

- [ ] **Expand the CLIP prompt set.** Currently only 4 prompts: `"a presentation slide"`, `"a person talking to a camera"`, `"a screen recording of software"`, `"a blank screen"`. PDD §4.2.1 requires more — at minimum add: `"an advertisement slide"`, `"an end credits screen"`, `"a title card"`, `"a sponsor logo"`, `"a video game UI"`. Without these, 6 of the 10 taxonomy labels become unreachable through the CLIP fallback classifier.
- [ ] **Add black-frame / pure-color frame detection.** Per-frame mean luminance + variance. Currently `rules.rule_holding_screen` approximates this with `hist_diff < 0.05`, which is not the same thing.
- [ ] **Add motion intensity feature.** Either optical flow magnitude or absolute frame difference. Currently only `hist_diff_to_previous` (HSV histogram correlation distance) is computed — PDD §4.2.1 lists both.
- [ ] **[V2.1 add-on] Color subsampling (4:2:0 / 4:2:2)** for optimized signal processing-style diff. Course-content requirement.
- [ ] **[V2.1 add-on] DCT high-frequency band analysis** on sampled frames. Course-content requirement.
- [ ] **(Optional) Switch from `transformers.CLIPModel` to `open_clip_torch`** if PDD §3.2 alignment matters; functionally equivalent but a different declared dependency.

### 4.2.2 Audio ([backend/pipeline/features /audio.py](../backend/pipeline/features%20/audio.py)) — **most under-implemented module**

- [ ] **Compute RMS energy envelope (per-second).** Required for proper silence/dead-air detection — currently `rule_dead_air` only checks `is_speech == 0`, so a music-only intro registers as continuous dead_air.
- [ ] **Compute spectral features** — spectral centroid, bandwidth, zero-crossing rate. Required for speech vs. music vs. noise discrimination per PDD §4.2.2.
- [ ] **Add music vs. speech classification.** PDD lists this as a separate output. Intros and sponsorships are often music-only; without this signal `boundaries.py` cannot detect the speech↔music transitions the PDD specifies.
- [ ] **Use `librosa`** for the spectral computations (PDD §4.2.2). Not currently imported.
- [ ] **Extend `AudioFeatureSegment` schema** to carry `rms_energy`, `spectral_centroid`, `spectral_bandwidth`, `zero_crossing_rate`, and an `audio_class` ∈ {speech, music, silence, noise} field. Currently only `start`/`end`/`is_speech` exist.
- [ ] **[V2.1 add-on] Signal entropy** on the audio spectrum. Course-content requirement.

### 4.2.3 Text ([backend/pipeline/features /text.py](../backend/pipeline/features%20/text.py))

- [ ] **Move keyword-dictionary matching into the text feature stage** and emit a `matched_keywords: list[str]` field per `TextFeatureSegment`. Currently keyword detection lives in `fusion/rules.py` and is recomputed at fusion time — PDD §4.2.3 places it in the feature stage.
- [ ] **Persist the 384-d MiniLM embedding vectors** (e.g. as a `.npy` next to `text_features.json`). Currently embeddings are computed then discarded; only cosine-to-next is kept. This blocks the supervised classifier path (§4.3 step 4) and the cross-episode matching (§4.3 V2.1 add-on).

### 4.2.4 Cross-cutting

- [ ] **Add a phase 2 CLI orchestrator** (e.g. `python -m backend.pipeline.features.run`) that runs all three sub-pipelines in one process and unloads each model between stages (`del model; torch.cuda.empty_cache()`) per PDD §4.2.4.
- [ ] **Chain phase 2 + phase 3 into the existing `ingest.py` orchestrator** so `python -m backend.pipeline.ingest <video.mp4>` produces `metadata.json` end-to-end.
- [ ] **Cache invalidation:** features stages currently skip if output JSON exists; should also check that the upstream artifact (transcript / frames / audio) is not newer.

---

## §4.3 — Segmentation & Fusion

### Step 1 — Time-grid alignment ([fusion/align.py](../backend/pipeline/fusion/align.py))
No gaps. ✅

### Step 2 — Hard-rule triggers ([fusion/rules.py](../backend/pipeline/fusion/rules.py))

- [ ] **Add CLIP-based intro/outro rule.** `rule_intro_window` / `rule_outro_window` are keyword-only. PDD: "若 CLIP 'title card' 平均数高 → intro / outro 候选". Requires the CLIP prompt set expansion above.
- [ ] **[V2.1 add-on] Cross-episode comparison rule.** Load an external "feature dictionary" of audio spectra / text embeddings from prior videos and match the current segment to detect repeated intros / recap segments. Currently `recap` is keyword-only. Needs:
  - external feature-store schema
  - matching code (cosine similarity over text embeddings + audio spectral signature)
  - an opt-in `--feature-dict <path>` CLI flag
- [ ] **Refine `dead_air` rule** to require RMS-below-threshold AND VAD == 0 (currently only VAD == 0). Depends on §4.2.2 RMS implementation.
- [ ] **Refine `sponsorship` extension** — currently extends ±15s symmetrically; sponsor reads typically extend further forward than backward. Consider asymmetric window (e.g. -5s / +25s).

### Step 3 — Boundary candidate generation ([fusion/boundaries.py](../backend/pipeline/fusion/boundaries.py))

- [ ] **Add music↔silence and speech↔music transitions** to boundary candidates. Currently only speech↔silence (depends on §4.2.2 music classification).
- [ ] **Smooth the per-second CLIP KL divergence** with a small rolling window before thresholding — second-by-second KL is noisy.

### Step 4 — Segment classification ([fusion/classify.py](../backend/pipeline/fusion/classify.py)) — **biggest weakness**

- [ ] **Replace `CLIP_LABEL_TO_SEGMENT` substring table with full 10-class prompt similarity.** PDD §4.3 step 4: "计算该段的 CLIP 视觉均值嵌入与 10 个类别的 prompt 嵌入的相似度，取最高类". The current 4-entry hardcoded mapping can only ever output `core_content`, `transition`, or `filler` from the CLIP path — **6 of 10 taxonomy labels are unreachable unless a hard rule fires.**
- [ ] **Use text and audio as real weighting signals in the fallback classifier.** Currently `text_score` is just `has_text` (0/1) and audio is just `speech_ratio`. Should incorporate keyword hits, similarity-to-class-prompts, and music/speech ratio.
- [ ] **(Optional) Add the supervised classifier mode** (sklearn LogisticRegression / GBM on segment-level features). PDD §4.3 step 4 explicitly lists this as the second of two modes. Acceptable to defer per PDD MVP note.

### Step 5 — Temporal smoothing ([fusion/smooth.py](../backend/pipeline/fusion/smooth.py))
No gaps. ✅

### Step 6 — Evidence generation

- [ ] **Make `visual_score` / `audio_score` / `text_score` actually explanatory.** They currently store modality activity proxies (max CLIP prob, `is_speech` mean, `has_text` mean) — they don't explain *why this label was chosen*. They should record the contribution each modality made to the chosen label specifically (e.g. for a `sponsorship` segment, `text_score` = max keyword-hit confidence, `visual_score` = CLIP probability for "advertisement slide" / "sponsor logo" prompts, etc.). Acceptance criterion §5 currently passes mechanically but the scores aren't load-bearing for "why".

---

## Testing gaps

- [ ] No tests at all for the phase 3 fusion modules (`align`, `rules`, `boundaries`, `classify`, `smooth`, `export`).
- [ ] Phase 2 tests are 10-line print-to-stdout stubs with no assertions.
- [ ] No end-to-end test that runs ingest → features → fusion on the synthetic MP4 fixture and checks the resulting `metadata.json` validates against the Pydantic schema.

---

## Priority recommendation (highest leverage first)

1. Repo hygiene block (rename `features `, remove `get-pip.py`).
2. Expand CLIP prompt set + replace classify.py mapping with all-10-class similarity. Together these unlock 6 currently-unreachable taxonomy labels.
3. Add RMS energy + music/speech classification to audio. Unblocks dead_air / holding_screen / sponsorship correctness.
4. Add black-frame brightness/variance to visual. Cheap, no model needed.
5. Move keyword detection into text features + persist embeddings. Sets up cross-episode matching later.
6. V2.1 course-content add-ons (4:2:0, DCT, entropy, cross-episode dictionary) — upgrades, not blockers for §4.4.
