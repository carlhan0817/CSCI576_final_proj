# Multi-Signal Ad Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-signal `rule_ad_break` (which relied on `is_speech == 0`) with a multi-modal ad detector that works across host content types (lecture / sports / animation / news), built around four converging signals: **visual style discontinuity** (CLIP embedding drift), **acoustic style discontinuity** (MFCC drift), **commercial-intent screen text** (OCR URL / price / phone / CTA / brand-lockup), and **commercial-intent transcript text** (CTA phrases said in narration). All bounded by hard-cut detection and minimum ad duration.

**Architecture:**
- **Phase 2 additions (visual)**: extract per-frame pooled CLIP image embedding (free — already running CLIP) and per-frame OCR text + commercial-pattern flags (new EasyOCR pass).
- **Phase 2 additions (audio)**: per-second 20-dim MFCC vector (`librosa.feature.mfcc`).
- **Phase 2 additions (text)**: extend the existing keyword-group infrastructure with a `cta:` group; CTA phrase list is shared between OCR (visual) and transcript (text) paths.
- **Phase 3 additions**: a `style_drift` module computes per-second cosine distance between each second's embedding and the mean of its 15-second past+future neighborhood. The same generic function applies to both 512-d CLIP embeddings and 20-d MFCC vectors. A new `rule_ad_block` fires when **either** drift signal is sustained AND at least one commercial signal (OCR-side OR transcript-side) is present, with optional confidence boost from bounding hard cuts.
- **Backwards compatible**: existing `rule_ad_break` and CLIP zero-shot prompts stay untouched. The new rule fires alongside them; classification picks the highest-confidence rule per segment. `rule_ad_break` is demoted to advisory confidence (Task 9).

**Tech Stack:**
- **RapidOCR (ONNX Runtime)** for screen text detection — chosen over EasyOCR for ~2-3× CPU speedup, ~10× smaller model footprint (~10 MB vs ~120 MB), and a lighter Windows install (no torchvision pull). Pure ONNX, CPU-by-default.
- Existing CLIP (`openai/clip-vit-base-patch32`) — re-used via `model.get_image_features()` for pooled embedding
- librosa MFCC (already a project dependency) for acoustic features
- numpy cosine distance (no scikit-learn dependency added)
- pydantic schemas extended for new fields
- pytest with synthetic fixtures

**Performance budget:**
For an unknown-length input video (the demo target tolerates videos up to ~120 min), the OCR pass is the dominant cost. Three optimisations are baked into Tasks 2, 3, and 5:
1. **RapidOCR over EasyOCR** (Task 2/3) — ~2-3× CPU speedup.
2. **Frame stride** (Task 5) — OCR every `OCR_STRIDE_SEC=3` seconds, forward-fill the result to skipped frames. ~3× additional speedup, near-zero recall loss because on-screen text typically persists ≥3 s.
3. **Luminance gate** (Task 5) — skip OCR on black/near-blank frames (no text possible).

Combined expected OCR pass cost (CPU): ~0.5 min for 30 min video, ~1 min for 60 min video, ~3 min for 120 min video.

---

## File Structure

**New files:**
- `backend/pipeline/features/ocr_signals.py` — EasyOCR loader + `extract_text(image)` + `detect_commercial_patterns(texts)`
- `backend/pipeline/fusion/style_drift.py` — `compute_drift(embeddings, window_sec)` per-second neighborhood divergence
- `tests/test_ocr_signals.py`
- `tests/test_style_drift.py`

**Extended files:**
- `backend/pipeline/schemas.py` — add fields to `VisualFrameFeature` (`clip_embedding`, `ocr_text`, `has_url`, `has_price`, `has_phone`, `has_cta`, `has_brand_lockup`) and to `AudioFeatureSegment` (`mfcc`)
- `backend/pipeline/features/visual.py` — extract pooled embedding alongside zero-shot logits; call OCR per frame; populate new fields
- `backend/pipeline/features/audio.py` — compute per-second MFCC; populate the new `mfcc` field
- `backend/pipeline/features/text.py` — register the new `cta` keyword group
- `backend/pipeline/features/ocr_signals.py` — import shared `CTA_KEYWORDS` from `rules.py` (Task 14 refactor; replaces the inline list from Task 2)
- `backend/pipeline/fusion/align.py` — propagate new per-second signals into the grid: `style_drift[t]`, `audio_drift[t]`, `has_url` / `has_price` / `has_phone` / `has_cta` / `has_brand_lockup` (each T×1), `text_has_cta[t]`
- `backend/pipeline/fusion/rules.py` — add `CTA_KEYWORDS` constant, add `rule_ad_block`, demote `rule_ad_break` to advisory; register `rule_ad_block` in `run_all_rules`
- `tests/test_visual.py`, `tests/test_audio_phase2.py`, `tests/test_text.py`, `tests/test_align.py`, `tests/test_rules.py`, `tests/test_schemas.py` — extend with new-signal assertions

**Untouched:** `classify.py`, `smooth.py`, `boundaries.py`, `export.py`. CLIP `SCENE_LABELS` deliberately unchanged (per user: don't rewrite prompts).

---

## Task 1: Extend `VisualFrameFeature` schema with embedding + OCR fields

**Files:**
- Modify: `backend/pipeline/schemas.py:25-36`
- Modify: `tests/test_schemas.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_schemas.py`:

```python
def test_visual_frame_feature_supports_embedding_and_ocr():
    from backend.pipeline.schemas import VisualFrameFeature

    f = VisualFrameFeature(
        frame_index=0,
        timestamp_sec=0.0,
        clip_embedding=[0.1] * 512,
        ocr_text="visit example.com",
        has_url=True,
        has_price=False,
        has_phone=False,
        has_cta=False,
        has_brand_lockup=False,
    )
    assert len(f.clip_embedding) == 512
    assert f.has_url is True
    assert f.ocr_text == "visit example.com"


def test_visual_frame_feature_defaults_keep_backcompat():
    from backend.pipeline.schemas import VisualFrameFeature

    f = VisualFrameFeature(frame_index=0, timestamp_sec=0.0)
    assert f.clip_embedding == []
    assert f.ocr_text == ""
    assert f.has_url is False
    assert f.has_brand_lockup is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schemas.py::test_visual_frame_feature_supports_embedding_and_ocr -v`
Expected: FAIL — `clip_embedding` etc. are unexpected keys.

- [ ] **Step 3: Add fields to schema**

In `backend/pipeline/schemas.py`, modify `VisualFrameFeature`:

```python
class VisualFrameFeature(BaseModel):
    frame_index: int = Field(..., ge=0)
    timestamp_sec: float = Field(..., ge=0.0)
    hist_diff_to_previous: float = Field(0.0, description="HSV histogram correlation distance from the previous frame.")
    is_hard_cut: bool = Field(False, description="True if hist_diff exceeds the cut threshold.")
    clip_labels: Dict[str, float] = Field(default_factory=dict, description="Zero-shot CLIP probabilities per scene prompt.")
    clip_embedding: List[float] = Field(default_factory=list, description="Pooled CLIP image embedding (512-d for ViT-B/32).")
    mean_luminance: float = Field(0.0, ge=0.0, description="Mean pixel brightness in [0, 255] (grayscale).")
    luminance_variance: float = Field(0.0, ge=0.0, description="Variance of pixel brightness.")
    is_black_frame: bool = Field(False, description="True if mean_luminance < 10 and luminance_variance < 50.")
    motion_intensity: float = Field(0.0, ge=0.0, description="Mean absolute pixel difference from the previous frame.")
    chroma_diff: float = Field(0.0, ge=0.0, description="[V2.1] Mean chroma channel difference after 4:2:0 downsampling.")
    dct_hf_energy: float = Field(0.0, ge=0.0, description="[V2.1] Normalized high-frequency DCT energy of the frame.")
    ocr_text: str = Field("", description="Concatenated text detected on screen by OCR (lowercased, deduped).")
    has_url: bool = Field(False, description="OCR text matches a URL pattern.")
    has_price: bool = Field(False, description="OCR text matches a price/discount pattern.")
    has_phone: bool = Field(False, description="OCR text matches a phone-number pattern.")
    has_cta: bool = Field(False, description="OCR text matches a CTA phrase (e.g. 'shop now').")
    has_brand_lockup: bool = Field(False, description="OCR text matches a known brand-lockup pattern (e.g. trademark symbol present).")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_schemas.py -v -k visual_frame`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/schemas.py tests/test_schemas.py
git commit -m "feat(schema): add CLIP embedding and OCR commercial-signal fields to VisualFrameFeature"
```

---

## Task 2: OCR signals module — patterns only (no model yet)

**Files:**
- Create: `backend/pipeline/features/ocr_signals.py`
- Create: `tests/test_ocr_signals.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ocr_signals.py`:

```python
"""Tests for backend/pipeline/features/ocr_signals.py — pattern detection only."""
from __future__ import annotations

import pytest

from backend.pipeline.features.ocr_signals import detect_commercial_patterns


class TestUrl:
    def test_https_url(self):
        out = detect_commercial_patterns(["visit https://shop.example.com today"])
        assert out["has_url"] is True

    def test_bare_domain(self):
        out = detect_commercial_patterns(["go to shop.com"])
        assert out["has_url"] is True

    def test_lecture_text_no_url(self):
        out = detect_commercial_patterns(["the cosine similarity is 0.83"])
        assert out["has_url"] is False


class TestPrice:
    def test_dollar_amount(self):
        assert detect_commercial_patterns(["only $19.99 today"])["has_price"] is True

    def test_percent_off(self):
        assert detect_commercial_patterns(["50% off"])["has_price"] is True

    def test_free_with_purchase(self):
        # Just "free" alone shouldn't trigger; should require commercial context.
        assert detect_commercial_patterns(["the function is free of side effects"])["has_price"] is False


class TestPhone:
    def test_us_phone(self):
        assert detect_commercial_patterns(["call 1-800-555-1234"])["has_phone"] is True

    def test_lecture_number_not_phone(self):
        assert detect_commercial_patterns(["the answer is 12345"])["has_phone"] is False


class TestCta:
    def test_shop_now(self):
        assert detect_commercial_patterns(["shop now"])["has_cta"] is True

    def test_subscribe(self):
        # 'subscribe' alone is too generic (YouTube self-promo) — require an ad-style phrase.
        assert detect_commercial_patterns(["please subscribe"])["has_cta"] is False

    def test_limited_time(self):
        assert detect_commercial_patterns(["limited time offer"])["has_cta"] is True


class TestBrandLockup:
    def test_trademark_symbol(self):
        assert detect_commercial_patterns(["BrandX™"])["has_brand_lockup"] is True

    def test_registered_symbol(self):
        assert detect_commercial_patterns(["BrandX®"])["has_brand_lockup"] is True

    def test_lecture_no_lockup(self):
        assert detect_commercial_patterns(["this is a slide"])["has_brand_lockup"] is False


class TestEmptyAndNoise:
    def test_empty(self):
        out = detect_commercial_patterns([])
        assert out == {
            "has_url": False, "has_price": False, "has_phone": False,
            "has_cta": False, "has_brand_lockup": False,
        }

    def test_only_whitespace(self):
        out = detect_commercial_patterns(["   ", ""])
        assert all(v is False for v in out.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ocr_signals.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement pattern detection**

Create `backend/pipeline/features/ocr_signals.py`:

```python
"""Phase 2: OCR-derived commercial-intent signals.

This module is split into two parts:

    * Pattern detection (`detect_commercial_patterns`) — pure-Python regex over
      strings. Cheap, deterministic, easy to test.
    * Frame-level OCR (`extract_text`) — wraps EasyOCR. Heavy import, lazily
      loaded on first call.

Phase 2 calls both per-frame and stores the resulting flags + raw OCR text on
each VisualFrameFeature.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional


# ── Regex tables ──────────────────────────────────────────────────────────────

# URL: explicit scheme, www., or a "<word>.<tld>" with a known commercial TLD.
_URL_PATTERNS = [
    re.compile(r"\bhttps?://[^\s]+", re.I),
    re.compile(r"\bwww\.[a-z0-9-]+\.[a-z]{2,}\b", re.I),
    re.compile(r"\b[a-z0-9-]+\.(?:com|net|org|io|ai|co|tv|app|store|shop)\b", re.I),
]

# Price: $X, X% off, "for $X".
_PRICE_PATTERNS = [
    re.compile(r"\$\s?\d{1,5}(?:[.,]\d{2})?\b"),
    re.compile(r"\b\d{1,3}\s?%\s*off\b", re.I),
    re.compile(r"\bsave\s+\$?\d", re.I),
]

# Phone: NANP-ish — 1-800, (XXX) XXX-XXXX, XXX.XXX.XXXX, etc.
_PHONE_PATTERNS = [
    re.compile(r"\b1[-. ]?\d{3}[-. ]?\d{3}[-. ]?\d{4}\b"),
    re.compile(r"\b\(\d{3}\)\s*\d{3}[-. ]?\d{4}\b"),
    re.compile(r"\b1[-. ]?(?:800|888|877|866|855|844|833|822)[-. ]?[A-Z]{4,}\b"),
]

# CTA phrases — chosen to NOT collide with lecture/news/animation phrasing.
# "subscribe" alone deliberately omitted (YouTube self-promo, not ad-specific).
_CTA_PHRASES = [
    "shop now", "buy now", "order now", "order today", "available at",
    "available now", "limited time", "for a limited time", "offer ends",
    "introducing the", "new from", "use promo code", "use code",
    "download the app", "visit our store", "in stores now", "in stores today",
]

# Brand lockup: trademark/registered glyphs. Cheap and high-precision.
_LOCKUP_PATTERNS = [
    re.compile(r"[™®©]"),  # ™ ® ©
]


def _scan_any(patterns, text: str) -> bool:
    return any(p.search(text) for p in patterns)


def detect_commercial_patterns(texts: List[str]) -> Dict[str, bool]:
    """Run all regex tables across the joined OCR text. Returns 5 boolean flags."""
    joined = " ".join(t for t in texts if t and t.strip())
    if not joined:
        return {
            "has_url": False, "has_price": False, "has_phone": False,
            "has_cta": False, "has_brand_lockup": False,
        }

    return {
        "has_url": _scan_any(_URL_PATTERNS, joined),
        "has_price": _scan_any(_PRICE_PATTERNS, joined),
        "has_phone": _scan_any(_PHONE_PATTERNS, joined),
        "has_cta": any(phrase in joined.lower() for phrase in _CTA_PHRASES),
        "has_brand_lockup": _scan_any(_LOCKUP_PATTERNS, joined),
    }


# ── RapidOCR loader (lazy) ────────────────────────────────────────────────────

_OCR_READER = None


def _get_reader(device: str = "cpu"):
    """Lazy-load RapidOCR (ONNX Runtime). ~10 MB models, CPU-only by default.

    `device` is accepted for signature compatibility but ignored — RapidOCR runs
    on CPU via onnxruntime. For GPU we'd switch to rapidocr_paddle (deferred).
    """
    global _OCR_READER
    if _OCR_READER is None:
        from rapidocr_onnxruntime import RapidOCR  # heavy import — lazy
        _OCR_READER = RapidOCR()
    return _OCR_READER


def extract_text(image_bgr, device: str = "cpu") -> List[str]:
    """Run RapidOCR on a BGR numpy image. Returns a list of detected text strings.

    RapidOCR returns either `(result, elapse)` where `result` is
    `[[bbox, text, confidence], ...]` or `(None, elapse)` when nothing detected.
    Returns lowercased strings, confidence-filtered (>=0.3), non-empty only.
    """
    reader = _get_reader(device)
    result, _elapse = reader(image_bgr)
    if not result:
        return []
    return [
        str(text).lower()
        for _bbox, text, conf in result
        if conf >= 0.3 and str(text).strip()
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ocr_signals.py -v`
Expected: all tests pass (these only exercise `detect_commercial_patterns`, no EasyOCR import).

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/features/ocr_signals.py tests/test_ocr_signals.py
git commit -m "feat(features): add ocr_signals module with commercial pattern regexes"
```

---

## Task 3: RapidOCR install + smoke test for the loader

**Files:**
- Modify: `pyproject.toml` (or `requirements.txt`, whichever the repo uses)
- Modify: `tests/test_ocr_signals.py`

- [ ] **Step 1: Confirm dependency manifest location**

Run: `Get-ChildItem pyproject.toml, requirements*.txt -ErrorAction SilentlyContinue`
Expected: one or both exist. Note which.

- [ ] **Step 2: Add rapidocr-onnxruntime dependency**

If `pyproject.toml` (PEP 621 style):

```toml
# under [project] dependencies = [...]
"rapidocr-onnxruntime>=1.3,<2.0",
```

If `requirements.txt`:

```
rapidocr-onnxruntime>=1.3,<2.0
```

- [ ] **Step 3: Install**

Run: `pip install "rapidocr-onnxruntime>=1.3,<2.0"`
Expected: success. RapidOCR ships its own ONNX models (~10 MB), bundled with the wheel — no separate download. The first call still has a small one-off cost to load the ONNX session.

- [ ] **Step 4: Add a smoke test gated on opt-in env var**

Append to `tests/test_ocr_signals.py`:

```python
import os

import numpy as np
import cv2
import pytest


@pytest.mark.skipif(
    os.environ.get("RUN_OCR_TESTS") != "1",
    reason="Heavy OCR test — set RUN_OCR_TESTS=1 to enable.",
)
def test_extract_text_reads_white_on_black():
    """Render the word 'SHOP NOW' and verify RapidOCR finds it."""
    from backend.pipeline.features.ocr_signals import extract_text

    img = np.zeros((100, 400, 3), dtype=np.uint8)
    cv2.putText(
        img, "SHOP NOW", (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
        2.0, (255, 255, 255), 4, cv2.LINE_AA,
    )

    texts = extract_text(img)
    joined = " ".join(texts)
    assert "shop" in joined or "now" in joined, f"OCR failed; got {texts!r}"
```

- [ ] **Step 5: Run smoke test once**

Run (PowerShell): `$env:RUN_OCR_TESTS = "1"; pytest tests/test_ocr_signals.py::test_extract_text_reads_white_on_black -v; Remove-Item Env:RUN_OCR_TESTS`
Expected: PASS. The first run takes a few seconds for ONNX session warm-up.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml tests/test_ocr_signals.py
git commit -m "build: add rapidocr-onnxruntime dependency and gated OCR smoke test"
```

---

## Task 4: Extract pooled CLIP image embedding in `visual.py`

**Files:**
- Modify: `backend/pipeline/features/visual.py:208-216` (CLIP block) and `:218-233` (assembly)
- Modify: `tests/test_visual.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_visual.py`:

```python
def test_visual_features_include_clip_embedding(phase1_workspace):
    """Each frame should carry a 512-dim normalised CLIP embedding."""
    import json

    from backend.pipeline.features.visual import extract_visual_features
    from backend.pipeline.schemas import VisualFeatures

    extract_visual_features(phase1_workspace, device="cpu")
    data = json.loads(phase1_workspace.visual_features_path.read_text())
    feats = VisualFeatures.model_validate(data)

    assert len(feats.frames) > 0
    for f in feats.frames:
        assert len(f.clip_embedding) == 512
        # Pooled embedding should be non-zero (not all defaults).
        assert sum(abs(x) for x in f.clip_embedding) > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_visual.py::test_visual_features_include_clip_embedding -v`
Expected: FAIL — `clip_embedding` is empty list because feature extractor doesn't write it yet.

- [ ] **Step 3: Modify the CLIP block**

In `backend/pipeline/features/visual.py`, replace the CLIP scoring block (around lines 208-216):

```python
        # ── CLIP zero-shot scene classification + pooled image embedding ────
        pil_image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        text_inputs = processor(
            text=SCENE_LABELS, return_tensors="pt", padding=True
        ).to(device)
        image_inputs = processor(images=pil_image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = clip_model.get_image_features(**image_inputs)
            text_features = clip_model.get_text_features(**text_inputs)
            # Normalise then dot-product = cosine similarity.
            image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
            text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
            logits = image_norm @ text_norm.T * clip_model.logit_scale.exp()
            probs = logits.softmax(dim=1)[0].cpu().numpy()
            pooled = image_norm[0].cpu().numpy()
        clip_results = {label: float(prob) for label, prob in zip(SCENE_LABELS, probs)}
        clip_embedding = [round(float(x), 6) for x in pooled.tolist()]
```

And in the assembly block (around line 218), add `clip_embedding=clip_embedding,` inside the `VisualFrameFeature(...)` constructor:

```python
        frame_features.append(
            VisualFrameFeature(
                frame_index=frame_idx,
                timestamp_sec=timestamp_sec,
                hist_diff_to_previous=round(hist_diff, 6),
                is_hard_cut=is_cut,
                clip_labels=clip_results,
                clip_embedding=clip_embedding,
                mean_luminance=round(mean_lum, 3),
                luminance_variance=round(var_lum, 3),
                is_black_frame=is_black,
                motion_intensity=round(motion, 4),
                chroma_diff=round(chroma_diff, 6),
                dct_hf_energy=round(dct_energy, 6),
            )
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_visual.py::test_visual_features_include_clip_embedding -v`
Expected: PASS (slow because it boots CLIP).

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/features/visual.py tests/test_visual.py
git commit -m "feat(visual): emit pooled CLIP image embedding alongside zero-shot probs"
```

---

## Task 5: Wire OCR into `visual.py` with stride + black-frame gate

**Files:**
- Modify: `backend/pipeline/features/visual.py` (top-level imports + per-frame loop, add `_should_run_ocr` helper)
- Modify: `tests/test_visual.py`

This task wires OCR into the per-frame extraction with two performance optimisations baked in from the start:
1. **Stride** — only call OCR every `OCR_STRIDE_SEC` seconds; forward-fill the result onto skipped frames.
2. **Luminance gate** — skip OCR on near-black or near-uniform frames (no on-screen text possible).

A small helper `_should_run_ocr()` makes the gating logic unit-testable without booting the model.

- [ ] **Step 1: Write the failing helper test**

Append to `tests/test_visual.py`:

```python
class TestShouldRunOcr:
    """_should_run_ocr decides whether to call OCR on a given frame.

    Logic:
      - Honour stride: only frames where (frame_idx % stride) == 0 qualify.
      - Skip if the frame is black or near-uniform (low luminance variance).
    """

    def test_first_frame_with_normal_content_runs_ocr(self):
        from backend.pipeline.features.visual import _should_run_ocr
        assert _should_run_ocr(frame_idx=0, stride=3, is_black=False, lum_var=500.0) is True

    def test_skipped_by_stride(self):
        from backend.pipeline.features.visual import _should_run_ocr
        assert _should_run_ocr(frame_idx=1, stride=3, is_black=False, lum_var=500.0) is False
        assert _should_run_ocr(frame_idx=2, stride=3, is_black=False, lum_var=500.0) is False
        assert _should_run_ocr(frame_idx=3, stride=3, is_black=False, lum_var=500.0) is True

    def test_black_frame_skipped_even_when_stride_qualifies(self):
        from backend.pipeline.features.visual import _should_run_ocr
        assert _should_run_ocr(frame_idx=0, stride=3, is_black=True, lum_var=500.0) is False

    def test_near_uniform_frame_skipped(self):
        from backend.pipeline.features.visual import _should_run_ocr
        # Variance below threshold => effectively a flat / near-blank frame
        assert _should_run_ocr(frame_idx=0, stride=3, is_black=False, lum_var=50.0) is False

    def test_stride_one_means_every_frame(self):
        from backend.pipeline.features.visual import _should_run_ocr
        for t in range(5):
            assert _should_run_ocr(frame_idx=t, stride=1, is_black=False, lum_var=500.0) is True
```

- [ ] **Step 2: Write the failing forward-fill integration test**

Append to `tests/test_visual.py`:

```python
def test_ocr_strides_and_forward_fills(phase1_workspace, monkeypatch):
    """Patch OCR to a counter; with stride=2 the call count should be ~half the frame count,
    but every frame still ends up with the OCR result from the most recent OCR'd frame."""
    import json
    from backend.pipeline.features import visual as visual_mod
    from backend.pipeline.schemas import VisualFeatures

    calls = {"n": 0}

    def counting_ocr(img, device="cpu"):
        calls["n"] += 1
        return [f"visit shop.com call_{calls['n']}"]

    monkeypatch.setattr(visual_mod, "_ocr_extract_text", counting_ocr)
    monkeypatch.setattr(visual_mod, "OCR_STRIDE_SEC", 2)

    if phase1_workspace.visual_features_path.exists():
        phase1_workspace.visual_features_path.unlink()
    visual_mod.extract_visual_features(phase1_workspace, device="cpu")

    data = json.loads(phase1_workspace.visual_features_path.read_text())
    feats = VisualFeatures.model_validate(data)
    n_frames = len(feats.frames)

    # Stride=2 over 3 frames → OCR runs on frames 0 and 2 → 2 calls.
    assert calls["n"] <= (n_frames // 2) + 1
    assert calls["n"] >= 1

    # Every frame should still be marked has_url because the result was forward-filled.
    assert all(f.has_url for f in feats.frames)
    assert all("shop.com" in f.ocr_text for f in feats.frames)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_visual.py -v -k "should_run_ocr or ocr_strides"`
Expected: FAIL — `_should_run_ocr` and `OCR_STRIDE_SEC` not defined; `_ocr_extract_text` not defined on visual module.

- [ ] **Step 4: Add module-level constants and helper**

Near the top of `backend/pipeline/features/visual.py`, after the existing imports, add:

```python
from backend.pipeline.features.ocr_signals import (
    extract_text as _ocr_extract_text,
    detect_commercial_patterns,
)


# OCR is the most expensive per-frame step. We sample every Nth frame and reuse
# the result on the skipped frames (forward-fill). On-screen text typically
# persists ≥3 s, so stride 3 has negligible recall impact and triples throughput.
OCR_STRIDE_SEC = 3
# Skip OCR on frames whose pixel variance is below this — too uniform to host text.
OCR_MIN_LUMINANCE_VARIANCE = 100.0


def _should_run_ocr(
    frame_idx: int,
    stride: int,
    is_black: bool,
    lum_var: float,
) -> bool:
    """Gating logic for whether to run OCR on a particular frame.

    Returns True only when ALL of:
      - frame_idx is on a stride boundary
      - frame is not black
      - frame variance is high enough to plausibly contain text
    """
    if stride > 1 and (frame_idx % stride) != 0:
        return False
    if is_black:
        return False
    if lum_var < OCR_MIN_LUMINANCE_VARIANCE:
        return False
    return True
```

- [ ] **Step 5: Wire OCR into the per-frame loop with forward-fill**

Inside `extract_visual_features`, before the per-frame loop initialise the carry-forward state:

```python
    # Forward-fill state for OCR results across stride-skipped frames.
    last_ocr_lines: List[str] = []
    last_commercial: Dict[str, bool] = {
        "has_url": False, "has_price": False, "has_phone": False,
        "has_cta": False, "has_brand_lockup": False,
    }
```

In the per-frame loop body, just BEFORE the `VisualFrameFeature(...)` assembly, add:

```python
        # ── OCR + commercial-pattern detection (with stride + luminance gate)
        if _should_run_ocr(frame_idx, OCR_STRIDE_SEC, is_black, var_lum):
            try:
                last_ocr_lines = _ocr_extract_text(img_bgr, device=device)
            except Exception as e:
                log.warning("OCR failed on frame %d: %s", frame_idx, e)
                last_ocr_lines = []
            last_commercial = detect_commercial_patterns(last_ocr_lines)
        # else: keep last_ocr_lines / last_commercial as-is (forward-fill)
        ocr_joined = " ".join(last_ocr_lines)[:500]
```

And extend the `VisualFrameFeature(...)` constructor:

```python
                ocr_text=ocr_joined,
                has_url=last_commercial["has_url"],
                has_price=last_commercial["has_price"],
                has_phone=last_commercial["has_phone"],
                has_cta=last_commercial["has_cta"],
                has_brand_lockup=last_commercial["has_brand_lockup"],
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_visual.py -v -k "should_run_ocr or ocr_strides"`
Expected: 6 passed (5 helper tests + 1 forward-fill integration).

- [ ] **Step 7: Commit**

```bash
git add backend/pipeline/features/visual.py tests/test_visual.py
git commit -m "feat(visual): wire OCR with stride forward-fill and luminance gate"
```

---

## Task 6: `style_drift` module — per-second neighborhood divergence

**Files:**
- Create: `backend/pipeline/fusion/style_drift.py`
- Create: `tests/test_style_drift.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_style_drift.py`:

```python
"""Tests for backend/pipeline/fusion/style_drift.py."""
from __future__ import annotations

import numpy as np

from backend.pipeline.fusion.style_drift import compute_drift


def _normalise(v):
    return v / np.linalg.norm(v)


class TestComputeDrift:
    def test_uniform_embeddings_have_zero_drift(self):
        T, D = 60, 16
        emb = np.tile(_normalise(np.ones(D)), (T, 1)).astype(np.float32)
        drift = compute_drift(emb, window_sec=15)
        assert drift.shape == (T,)
        assert np.allclose(drift, 0.0, atol=1e-5)

    def test_inserted_block_produces_drift_peak(self):
        T, D = 100, 16
        host = _normalise(np.array([1.0] + [0.0] * (D - 1)))
        ad = _normalise(np.array([0.0, 1.0] + [0.0] * (D - 2)))

        emb = np.tile(host, (T, 1)).astype(np.float32)
        emb[40:60] = ad  # 20-second ad block in the middle

        drift = compute_drift(emb, window_sec=15)

        host_mean_drift = drift[5:35].mean()  # quiet host region
        ad_mean_drift = drift[42:58].mean()   # interior of ad block
        assert ad_mean_drift > 0.5, f"ad drift too low: {ad_mean_drift}"
        assert host_mean_drift < 0.1, f"host drift not flat: {host_mean_drift}"

    def test_short_video_does_not_crash(self):
        emb = np.random.RandomState(0).randn(5, 16).astype(np.float32)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        drift = compute_drift(emb, window_sec=15)
        assert drift.shape == (5,)
        assert np.all(np.isfinite(drift))

    def test_handles_zero_vector_safely(self):
        T, D = 30, 16
        emb = np.tile(_normalise(np.ones(D)), (T, 1)).astype(np.float32)
        emb[10] = 0.0
        drift = compute_drift(emb, window_sec=10)
        assert np.all(np.isfinite(drift))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_style_drift.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement compute_drift**

Create `backend/pipeline/fusion/style_drift.py`:

```python
"""Phase 3: per-second visual style discontinuity from CLIP embeddings.

For each second t, drift[t] is the cosine distance between the embedding at
second t and the mean embedding of its surrounding context (past and future
window of `window_sec` seconds, excluding the immediate ±2-second neighbours
to prevent self-leakage at boundaries).

A sustained run of high drift is the visual fingerprint of "this block is
foreign to its neighbourhood" — the most reliable cross-genre ad signal.
"""
from __future__ import annotations
import numpy as np


def _l2_normalise(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < eps:
        return np.zeros_like(v)
    return v / norm


def compute_drift(embeddings: np.ndarray, window_sec: int = 15) -> np.ndarray:
    """Compute per-second neighborhood drift.

    Args:
        embeddings: shape (T, D), one row per second. Need not be pre-normalised.
        window_sec: half-window in seconds. Context = past[t-window..t-2] ∪ future[t+2..t+window].

    Returns:
        drift: shape (T,) float32 array in [0, 2]. Higher = more anomalous.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D (T, D); got {embeddings.shape}")
    T = embeddings.shape[0]
    drift = np.zeros(T, dtype=np.float32)

    # Pre-normalise once.
    normed = np.zeros_like(embeddings, dtype=np.float32)
    for t in range(T):
        normed[t] = _l2_normalise(embeddings[t])

    GAP = 2  # immediate-neighbor exclusion radius
    for t in range(T):
        past_lo, past_hi = max(0, t - window_sec), max(0, t - GAP)
        fut_lo, fut_hi = min(T, t + GAP + 1), min(T, t + window_sec + 1)
        ctx_chunks = []
        if past_hi > past_lo:
            ctx_chunks.append(normed[past_lo:past_hi])
        if fut_hi > fut_lo:
            ctx_chunks.append(normed[fut_lo:fut_hi])
        if not ctx_chunks:
            drift[t] = 0.0
            continue
        ctx = np.concatenate(ctx_chunks, axis=0)
        ctx_mean = _l2_normalise(ctx.mean(axis=0))
        drift[t] = float(1.0 - np.dot(normed[t], ctx_mean))

    return drift
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_style_drift.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/fusion/style_drift.py tests/test_style_drift.py
git commit -m "feat(fusion): add style_drift module computing per-second CLIP-embedding neighborhood divergence"
```

---

## Task 7: Wire embedding + drift + OCR signals into `align.py`

**Files:**
- Modify: `backend/pipeline/fusion/align.py`
- Modify: `tests/test_align.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_align.py` (use existing fixtures or build a minimal one):

```python
def test_grid_includes_clip_embedding_drift_and_ocr_flags():
    import numpy as np
    from backend.pipeline.fusion.align import build_per_second_grid
    from backend.pipeline.schemas import (
        VisualFeatures, VisualFrameFeature,
        AudioFeatures, AudioFeatureSegment,
        TextFeatures,
    )

    visual = VisualFeatures(frames=[
        VisualFrameFeature(
            frame_index=t, timestamp_sec=float(t),
            clip_embedding=[1.0] + [0.0] * 511 if t < 5 else [0.0, 1.0] + [0.0] * 510,
            has_url=(t == 7),
            has_price=False, has_phone=False, has_cta=False, has_brand_lockup=False,
        )
        for t in range(10)
    ])
    audio = AudioFeatures(segments=[
        AudioFeatureSegment(start=float(t), end=float(t + 1)) for t in range(10)
    ])
    text = TextFeatures(segments=[])

    grid = build_per_second_grid(visual, audio, text, duration_sec=10.0)

    # New per-second arrays exist
    assert "style_drift" in grid
    assert grid["style_drift"].shape == (10,)
    # Clear regime change at t=5: drift around boundary should be > drift in stable regions
    assert grid["style_drift"][5] > grid["style_drift"][1]

    assert "has_url" in grid
    assert grid["has_url"][7] == 1
    assert grid["has_url"][0] == 0

    for k in ("has_price", "has_phone", "has_cta", "has_brand_lockup"):
        assert k in grid
        assert grid[k].shape == (10,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_align.py::test_grid_includes_clip_embedding_drift_and_ocr_flags -v`
Expected: FAIL — `style_drift` / `has_url` not in grid.

- [ ] **Step 3: Modify align.py**

In `backend/pipeline/fusion/align.py`, after the existing visual block (after the line `for lbl, arr in clip_per_label.items(): grid[f"clip_{lbl}"] = arr`), add:

```python
    # ── Visual: pooled CLIP embedding + per-second style drift ────────────────
    embed_dim = (
        len(visual.frames[0].clip_embedding) if visual.frames and visual.frames[0].clip_embedding else 0
    )
    if embed_dim > 0:
        embeddings = np.zeros((T, embed_dim), dtype=np.float32)
        for f in visual.frames:
            t = int(f.timestamp_sec)
            if t >= T or not f.clip_embedding:
                continue
            embeddings[t] = np.array(f.clip_embedding, dtype=np.float32)
        from backend.pipeline.fusion.style_drift import compute_drift
        grid["style_drift"] = compute_drift(embeddings, window_sec=15)
    else:
        grid["style_drift"] = np.zeros(T, dtype=np.float32)

    # ── Visual: OCR commercial signals ───────────────────────────────────────
    has_url = np.zeros(T, dtype=np.int8)
    has_price = np.zeros(T, dtype=np.int8)
    has_phone = np.zeros(T, dtype=np.int8)
    has_cta = np.zeros(T, dtype=np.int8)
    has_brand_lockup = np.zeros(T, dtype=np.int8)

    for f in visual.frames:
        t = int(f.timestamp_sec)
        if t >= T:
            continue
        has_url[t] = int(f.has_url)
        has_price[t] = int(f.has_price)
        has_phone[t] = int(f.has_phone)
        has_cta[t] = int(f.has_cta)
        has_brand_lockup[t] = int(f.has_brand_lockup)

    grid["has_url"] = has_url
    grid["has_price"] = has_price
    grid["has_phone"] = has_phone
    grid["has_cta"] = has_cta
    grid["has_brand_lockup"] = has_brand_lockup
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_align.py::test_grid_includes_clip_embedding_drift_and_ocr_flags -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/fusion/align.py tests/test_align.py
git commit -m "feat(align): expose style_drift and OCR commercial flags in per-second grid"
```

---

## Task 8: New `rule_ad_block` — multi-signal ad detector

**Files:**
- Modify: `backend/pipeline/fusion/rules.py`
- Modify: `tests/test_rules.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_rules.py`:

```python
class TestRuleAdBlock:
    """rule_ad_block requires (a) sustained high style_drift, (b) at least one
    OCR commercial pattern hit, (c) bounding hard-cuts."""

    def _build_grid(self, T, drift_high, ocr_hit_t, hard_cuts):
        import numpy as np
        g = _grid(T)
        g["style_drift"] = np.zeros(T, dtype=np.float32)
        for s, e in drift_high:
            g["style_drift"][s:e] = 0.6
        g["has_url"] = np.zeros(T, dtype=np.int8)
        g["has_price"] = np.zeros(T, dtype=np.int8)
        g["has_phone"] = np.zeros(T, dtype=np.int8)
        g["has_cta"] = np.zeros(T, dtype=np.int8)
        g["has_brand_lockup"] = np.zeros(T, dtype=np.int8)
        if ocr_hit_t is not None:
            g["has_url"][ocr_hit_t] = 1
        for c in hard_cuts:
            g["is_hard_cut"][c] = 1
        return g

    def test_drift_with_ocr_and_cut_fires(self):
        from backend.pipeline.fusion.rules import rule_ad_block
        # 30-second drift block from t=50 to t=80, OCR url at t=60, cuts at 49 and 81
        g = self._build_grid(T=200, drift_high=[(50, 80)], ocr_hit_t=60, hard_cuts=[49, 81])
        hits = rule_ad_block(g)
        assert len(hits) == 1
        h = hits[0]
        assert h.label == "sponsorship"
        assert h.rule_name == "ad_block"
        assert 49 <= h.start_sec <= 51 and 79 <= h.end_sec <= 82
        assert h.confidence >= 0.85

    def test_drift_without_ocr_does_not_fire(self):
        from backend.pipeline.fusion.rules import rule_ad_block
        g = self._build_grid(T=200, drift_high=[(50, 80)], ocr_hit_t=None, hard_cuts=[49, 81])
        hits = rule_ad_block(g)
        assert hits == []

    def test_short_drift_below_min_duration_does_not_fire(self):
        from backend.pipeline.fusion.rules import rule_ad_block
        # 5-second drift is too short
        g = self._build_grid(T=200, drift_high=[(50, 55)], ocr_hit_t=52, hard_cuts=[49, 56])
        hits = rule_ad_block(g)
        assert hits == []

    def test_ocr_outside_drift_block_does_not_count(self):
        from backend.pipeline.fusion.rules import rule_ad_block
        # OCR url 30s away from the drift block — the rule should not pair them.
        g = self._build_grid(T=200, drift_high=[(50, 80)], ocr_hit_t=20, hard_cuts=[49, 81])
        hits = rule_ad_block(g)
        assert hits == []

    def test_no_hard_cut_boundary_still_fires_at_lower_confidence(self):
        from backend.pipeline.fusion.rules import rule_ad_block
        # Drift + OCR but no cut ⇒ still fire (drift+OCR is enough), but confidence lower.
        g = self._build_grid(T=200, drift_high=[(50, 80)], ocr_hit_t=60, hard_cuts=[])
        hits = rule_ad_block(g)
        assert len(hits) == 1
        assert hits[0].confidence < 0.85
        assert hits[0].confidence >= 0.6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rules.py::TestRuleAdBlock -v`
Expected: FAIL — `rule_ad_block` not defined.

- [ ] **Step 3: Implement rule_ad_block**

In `backend/pipeline/fusion/rules.py`, add near the other rule functions:

```python
# Tunable thresholds for rule_ad_block
AD_BLOCK_DRIFT_THRESHOLD = 0.30      # cosine-distance threshold to call a second "anomalous"
AD_BLOCK_MIN_DURATION = 10           # min length of a sustained-drift run (seconds)
AD_BLOCK_BOUNDARY_TOLERANCE = 3      # how close a hard cut must be to count as "bounding"
AD_BLOCK_BASE_CONFIDENCE = 0.7       # confidence when only drift+OCR agree
AD_BLOCK_BOUNDED_CONFIDENCE = 0.9    # confidence when also bounded by hard cuts


def _has_any_commercial_signal(grid: Dict[str, np.ndarray], s: int, e: int) -> bool:
    """True iff at least one OCR commercial flag is set anywhere in [s, e)."""
    for key in ("has_url", "has_price", "has_phone", "has_cta", "has_brand_lockup"):
        arr = grid.get(key)
        if arr is None:
            continue
        if arr[s:e].any():
            return True
    return False


def _has_bounding_cut(grid: Dict[str, np.ndarray], t: int, tolerance: int) -> bool:
    cuts = grid.get("is_hard_cut")
    if cuts is None:
        return False
    lo = max(0, t - tolerance)
    hi = min(len(cuts), t + tolerance + 1)
    return bool(cuts[lo:hi].any())


def rule_ad_block(grid: Dict[str, np.ndarray]) -> List[RuleHit]:
    """Multi-signal ad detector.

    Fires when ALL of the following hold over a contiguous run of length ≥ AD_BLOCK_MIN_DURATION:
      1. style_drift[t] >= AD_BLOCK_DRIFT_THRESHOLD for every t in the run (visual style discontinuity).
      2. At least one OCR commercial pattern (URL / price / phone / CTA / brand-lockup) within the run.
    Confidence is boosted to AD_BLOCK_BOUNDED_CONFIDENCE when both run boundaries are within
    AD_BLOCK_BOUNDARY_TOLERANCE of a hard cut; otherwise AD_BLOCK_BASE_CONFIDENCE.
    """
    drift = grid.get("style_drift")
    if drift is None:
        return []

    hot = (drift >= AD_BLOCK_DRIFT_THRESHOLD).astype(np.int8)
    hits: List[RuleHit] = []
    for s, e in _find_runs(hot, AD_BLOCK_MIN_DURATION):
        if not _has_any_commercial_signal(grid, s, e):
            continue
        bounded = (
            _has_bounding_cut(grid, s, AD_BLOCK_BOUNDARY_TOLERANCE)
            and _has_bounding_cut(grid, e, AD_BLOCK_BOUNDARY_TOLERANCE)
        )
        conf = AD_BLOCK_BOUNDED_CONFIDENCE if bounded else AD_BLOCK_BASE_CONFIDENCE
        hits.append(RuleHit(s, e, "sponsorship", "ad_block", confidence=conf))

    return hits
```

Then register it in `run_all_rules`, before the existing `rule_ad_break` line:

```python
    all_hits.extend(rule_ad_block(grid))
    all_hits.extend(rule_ad_break(grid))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_rules.py::TestRuleAdBlock -v`
Expected: 5 passed.

- [ ] **Step 5: Run the full rules test suite for regressions**

Run: `pytest tests/test_rules.py -v`
Expected: all green (no existing rule tests broken).

- [ ] **Step 6: Commit**

```bash
git add backend/pipeline/fusion/rules.py tests/test_rules.py
git commit -m "feat(rules): add rule_ad_block requiring style drift + OCR commercial signal"
```

---

## Task 9: Demote `rule_ad_break` to advisory (low confidence)

**Files:**
- Modify: `backend/pipeline/fusion/rules.py:rule_ad_break`
- Modify: `tests/test_rules.py`

The current `rule_ad_break` fires at confidence 0.85 on any 15s of no-speech. With the new `rule_ad_block` carrying the strong evidence, demote `rule_ad_break` to confidence 0.5 so it only acts as a tiebreaker when nothing else fires. This avoids the lecture/animation/news false positives while preserving Ad 2's coincidental success path.

- [ ] **Step 1: Update existing test expectations**

In `tests/test_rules.py`, find the existing `rule_ad_break` test cases and update assertions on `confidence` from `0.85` to `0.5`. Search for `rule_ad_break` and `confidence == 0.85` matches. (If there are no such existing tests, skip.)

- [ ] **Step 2: Modify the rule**

In `backend/pipeline/fusion/rules.py:rule_ad_break`, change the confidence value:

```python
def rule_ad_break(grid: Dict[str, np.ndarray]) -> List[RuleHit]:
    """Long stretch of no speech → ADVISORY-confidence sponsorship.

    Originally a strong signal, demoted because is_speech-only blocks generate
    too many false positives on animation / sports / news content. Use as a
    tiebreaker; rule_ad_block carries the high-confidence detection.
    """
    is_speech = grid["is_speech"]
    quiet = (is_speech == 0).astype(np.int8)
    quiet_closed = _close_short_gaps(quiet, AD_BREAK_MAX_GAP)
    hits = []
    for s, e in _find_runs(quiet_closed, AD_BREAK_MIN_DURATION):
        hits.append(RuleHit(s, e, "sponsorship", "ad_break", confidence=0.5))
    return hits
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_rules.py -v`
Expected: all green. If any test still asserts the old 0.85, update it to 0.5.

- [ ] **Step 4: Commit**

```bash
git add backend/pipeline/fusion/rules.py tests/test_rules.py
git commit -m "refactor(rules): demote rule_ad_break to advisory confidence"
```

---

## Task 10: End-to-end verification on test_004_2

**Files:**
- (no source changes — verification only)
- May modify: `tests/test_e2e_pipeline.py` if a regression check is desired

- [ ] **Step 1: Clear cached visual features**

Run (PowerShell): `Remove-Item workspace/test_004_2/features/visual_features.json -ErrorAction SilentlyContinue`

(audio + text features can stay — they're untouched.)

- [ ] **Step 2: Re-run Phase 2 visual extraction (slow — runs OCR per frame)**

Run: `python -m backend.pipeline.features.run --workspace workspace/test_004_2 --stage visual`

Expected: completes without error in ~5-10 minutes for a 32-min video at 1 FPS. Watch the log for OCR exceptions.

- [ ] **Step 3: Re-run Phase 3 fusion**

Run: `python -m backend.pipeline.fusion.run --workspace workspace/test_004_2`

Expected: completes in seconds.

- [ ] **Step 4: Inspect new metadata**

Run: `python -c "import json; m=json.load(open('workspace/test_004_2/metadata.json')); [print(s['segment_id'], s['start_sec'], s['end_sec'], s['label'], s['confidence'], s['evidence']['triggered_rules']) for s in m['segments'] if s['label']=='sponsorship']"`

Expected — at minimum:
- One sponsorship segment overlapping Ad 2 (1096–1142). **Regression check.**
- At least one of {Ad 1 (260–290), Ad 3 (1632–1692)} now produces a sponsorship segment with `ad_block` in `triggered_rules`.
- False-positive sponsorship segments outside the GT ad regions should be ≤ the previous count (currently 4: seg 3, 9, 13, 26).

- [ ] **Step 5: Save the comparison numbers as a verification note**

Create `docs/superpowers/plans/2026-05-04-multimodal-ad-detection-results.md` (one short table — not a long writeup):

```markdown
# Multi-Signal Ad Detection — test_004_2 Results

| Region | GT range | Old (rule_ad_break only) | New (with rule_ad_block) |
|---|---|---|---|
| Ad 1 | 260-290 | 3s | <fill in> |
| Ad 2 | 1096-1142 | 47s | <fill in> |
| Ad 3 | 1632-1692 | 33s | <fill in> |
| FP count | — | 4 | <fill in> |

OCR signals fired during ads: <list ad_index : flag : count>
```

- [ ] **Step 6: Commit results**

```bash
git add docs/superpowers/plans/2026-05-04-multimodal-ad-detection-results.md
git commit -m "docs: record test_004_2 ad-detection results before/after multi-signal rule"
```

---

---

## Task 11: Extend `AudioFeatureSegment` schema with MFCC field

**Files:**
- Modify: `backend/pipeline/schemas.py:6-17`
- Modify: `tests/test_schemas.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_schemas.py`:

```python
def test_audio_feature_segment_supports_mfcc():
    from backend.pipeline.schemas import AudioFeatureSegment

    s = AudioFeatureSegment(start=0.0, end=1.0, mfcc=[0.1] * 20)
    assert len(s.mfcc) == 20


def test_audio_feature_segment_mfcc_default_empty():
    from backend.pipeline.schemas import AudioFeatureSegment

    s = AudioFeatureSegment(start=0.0, end=1.0)
    assert s.mfcc == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schemas.py -v -k mfcc`
Expected: FAIL — `mfcc` is unexpected.

- [ ] **Step 3: Add the field**

In `backend/pipeline/schemas.py`, modify `AudioFeatureSegment`:

```python
class AudioFeatureSegment(BaseModel):
    start: float = Field(..., ge=0.0)
    end: float = Field(..., ge=0.0)
    is_speech: bool = Field(False, description="True if Silero VAD detects speech in this second.")
    rms_energy: float = Field(0.0, ge=0.0, description="Root-mean-square amplitude of the audio chunk.")
    spectral_centroid: float = Field(0.0, ge=0.0, description="Frequency centroid of the power spectrum (Hz).")
    spectral_bandwidth: float = Field(0.0, ge=0.0, description="Spectral bandwidth around the centroid (Hz).")
    zero_crossing_rate: float = Field(0.0, ge=0.0, description="Fraction of samples where the waveform crosses zero.")
    spectral_entropy: float = Field(0.0, ge=0.0, description="[V2.1] Shannon entropy of the normalized power spectrum.")
    audio_class: Literal["speech", "music", "silence", "noise"] = Field(
        "silence", description="Heuristic audio class for this second."
    )
    mfcc: List[float] = Field(
        default_factory=list,
        description="MFCC coefficients (20 by default) for this second. Used for audio style drift.",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_schemas.py -v -k mfcc`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/pipeline/schemas.py tests/test_schemas.py
git commit -m "feat(schema): add mfcc field to AudioFeatureSegment for audio style drift"
```

---

## Task 12: Compute MFCC per second in `audio.py`

**Files:**
- Modify: `backend/pipeline/features/audio.py`
- Modify: `tests/test_audio_phase2.py` (or wherever existing audio tests live)

- [ ] **Step 1: Locate the existing audio test file**

Run (PowerShell): `Get-ChildItem tests | Where-Object { $_.Name -like 'test_audio*' }`
Expected: at least `test_audio_phase2.py`.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_audio_phase2.py`:

```python
def test_audio_features_include_mfcc(synthetic_mp4, tmp_path):
    """Each AudioFeatureSegment should carry a 20-dim MFCC vector."""
    import json
    from backend.pipeline.workspace import Workspace
    from backend.pipeline.audio import extract_audio as phase1_extract_audio
    from backend.pipeline.features.audio import extract_audio_features
    from backend.pipeline.schemas import AudioFeatures

    ws = Workspace.for_video(synthetic_mp4, root=tmp_path / "ws")
    ws.ensure()
    phase1_extract_audio(synthetic_mp4, ws)
    extract_audio_features(ws, device="cpu")

    data = json.loads(ws.audio_features_path.read_text())
    feats = AudioFeatures.model_validate(data)

    assert len(feats.segments) > 0
    for s in feats.segments:
        assert len(s.mfcc) == 20
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_audio_phase2.py::test_audio_features_include_mfcc -v`
Expected: FAIL — mfcc is empty list.

- [ ] **Step 4: Add MFCC computation**

In `backend/pipeline/features/audio.py`, find `_compute_second_features` (around line 48-) and extend it:

```python
def _compute_second_features(chunk: np.ndarray, sr: int = 16000) -> dict:
    """Compute all spectral features for a 1-second audio chunk.
    Returns a dict with keys: rms, centroid, bandwidth, zcr, entropy, mfcc.
    """
    if len(chunk) == 0:
        return dict(rms=0.0, centroid=0.0, bandwidth=0.0, zcr=0.0, entropy=0.0, mfcc=[0.0] * 20)

    # ... existing rms / centroid / bandwidth / zcr / entropy code stays unchanged ...

    # MFCC: 20 coefficients, mean over the second
    mfcc_frames = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=20)  # (20, n_frames)
    mfcc_mean = mfcc_frames.mean(axis=1)
    mfcc = [round(float(x), 4) for x in mfcc_mean.tolist()]

    return dict(
        rms=rms, centroid=centroid, bandwidth=bandwidth,
        zcr=zcr, entropy=entropy, mfcc=mfcc,
    )
```

(Keep the existing rms/centroid/etc. computations — only add the MFCC block and extend the return dict.)

Then in the per-second loop in `extract_audio_features`, pass `mfcc=feat["mfcc"]` to `AudioFeatureSegment(...)`:

```python
        segments.append(
            AudioFeatureSegment(
                start=float(t),
                end=float(t + 1),
                is_speech=bool(is_speech_arr[t]),
                rms_energy=round(feat["rms"], 6),
                spectral_centroid=round(feat["centroid"], 2),
                spectral_bandwidth=round(feat["bandwidth"], 2),
                zero_crossing_rate=round(feat["zcr"], 6),
                spectral_entropy=round(feat["entropy"], 4),
                audio_class=audio_class,
                mfcc=feat["mfcc"],
            )
        )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_audio_phase2.py::test_audio_features_include_mfcc -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/pipeline/features/audio.py tests/test_audio_phase2.py
git commit -m "feat(audio): emit per-second 20-dim MFCC vectors for downstream drift analysis"
```

---

## Task 13: Audio drift in grid + integrate into `rule_ad_block`

**Files:**
- Modify: `backend/pipeline/fusion/align.py`
- Modify: `backend/pipeline/fusion/rules.py:rule_ad_block`
- Modify: `tests/test_align.py`
- Modify: `tests/test_rules.py`

`compute_drift` from Task 6's `style_drift.py` is modality-agnostic — same algorithm on the (T, 20) MFCC matrix gives an audio drift signal. We expose it as `audio_drift` and let `rule_ad_block` fire when EITHER `style_drift` OR `audio_drift` is sustained.

- [ ] **Step 1: Write the failing align test**

Append to `tests/test_align.py`:

```python
def test_grid_includes_audio_drift():
    import numpy as np
    from backend.pipeline.fusion.align import build_per_second_grid
    from backend.pipeline.schemas import (
        VisualFeatures, AudioFeatures, AudioFeatureSegment, TextFeatures,
    )

    # 10 seconds: lecture-like MFCC for first 5s, ad-like MFCC for next 5s.
    lecture_mfcc = [10.0] + [0.0] * 19
    ad_mfcc = [0.0, 10.0] + [0.0] * 18
    audio = AudioFeatures(segments=[
        AudioFeatureSegment(
            start=float(t), end=float(t + 1),
            mfcc=(lecture_mfcc if t < 5 else ad_mfcc),
        )
        for t in range(10)
    ])
    visual = VisualFeatures(frames=[])
    text = TextFeatures(segments=[])

    grid = build_per_second_grid(visual, audio, text, duration_sec=10.0)

    assert "audio_drift" in grid
    assert grid["audio_drift"].shape == (10,)
    # Drift should peak around the regime change at t=5
    assert grid["audio_drift"][5] > grid["audio_drift"][1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_align.py::test_grid_includes_audio_drift -v`
Expected: FAIL — `audio_drift` not in grid.

- [ ] **Step 3: Wire MFCC + audio drift into align.py**

In `backend/pipeline/fusion/align.py`, after the existing audio block (after `grid["is_music"] = is_music`), add:

```python
    # ── Audio: MFCC matrix + per-second audio style drift ─────────────────────
    mfcc_dim = (
        len(audio.segments[0].mfcc) if audio.segments and audio.segments[0].mfcc else 0
    )
    if mfcc_dim > 0:
        mfcc_matrix = np.zeros((T, mfcc_dim), dtype=np.float32)
        for seg in audio.segments:
            t = int(np.floor(seg.start))
            if t >= T or not seg.mfcc:
                continue
            mfcc_matrix[t] = np.array(seg.mfcc, dtype=np.float32)
        from backend.pipeline.fusion.style_drift import compute_drift
        grid["audio_drift"] = compute_drift(mfcc_matrix, window_sec=15)
    else:
        grid["audio_drift"] = np.zeros(T, dtype=np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_align.py::test_grid_includes_audio_drift -v`
Expected: PASS.

- [ ] **Step 5: Write the failing rule test**

Append to `tests/test_rules.py` inside `class TestRuleAdBlock`:

```python
    def test_audio_drift_alone_with_ocr_can_fire(self):
        """rule_ad_block fires when audio drift is sustained even if visual drift is flat."""
        import numpy as np
        from backend.pipeline.fusion.rules import rule_ad_block

        T = 200
        g = self._build_grid(T=T, drift_high=[], ocr_hit_t=60, hard_cuts=[49, 81])
        g["audio_drift"] = np.zeros(T, dtype=np.float32)
        g["audio_drift"][50:80] = 0.6  # 30s of audio anomaly

        hits = rule_ad_block(g)
        assert len(hits) == 1
        assert hits[0].label == "sponsorship"
        assert "ad_block" == hits[0].rule_name

    def test_neither_drift_does_not_fire(self):
        from backend.pipeline.fusion.rules import rule_ad_block
        import numpy as np

        T = 200
        g = self._build_grid(T=T, drift_high=[], ocr_hit_t=60, hard_cuts=[49, 81])
        g["audio_drift"] = np.zeros(T, dtype=np.float32)
        hits = rule_ad_block(g)
        assert hits == []
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/test_rules.py::TestRuleAdBlock::test_audio_drift_alone_with_ocr_can_fire -v`
Expected: FAIL — current rule_ad_block only inspects `style_drift`.

- [ ] **Step 7: Modify `rule_ad_block` to use either drift signal**

In `backend/pipeline/fusion/rules.py`, replace the body of `rule_ad_block`:

```python
def rule_ad_block(grid: Dict[str, np.ndarray]) -> List[RuleHit]:
    """Multi-signal ad detector.

    Fires when ALL of the following hold over a contiguous run of length ≥ AD_BLOCK_MIN_DURATION:
      1. EITHER style_drift[t] OR audio_drift[t] >= AD_BLOCK_DRIFT_THRESHOLD (visual or acoustic discontinuity).
      2. At least one commercial signal (URL / price / phone / OCR-CTA / brand-lockup / transcript-CTA) within the run.
    Confidence boosts to AD_BLOCK_BOUNDED_CONFIDENCE when both run boundaries are within
    AD_BLOCK_BOUNDARY_TOLERANCE of a hard cut; otherwise AD_BLOCK_BASE_CONFIDENCE.
    """
    visual_drift = grid.get("style_drift")
    audio_drift = grid.get("audio_drift")
    if visual_drift is None and audio_drift is None:
        return []

    T = len(visual_drift) if visual_drift is not None else len(audio_drift)
    v = visual_drift if visual_drift is not None else np.zeros(T, dtype=np.float32)
    a = audio_drift if audio_drift is not None else np.zeros(T, dtype=np.float32)

    hot = ((v >= AD_BLOCK_DRIFT_THRESHOLD) | (a >= AD_BLOCK_DRIFT_THRESHOLD)).astype(np.int8)
    hits: List[RuleHit] = []
    for s, e in _find_runs(hot, AD_BLOCK_MIN_DURATION):
        if not _has_any_commercial_signal(grid, s, e):
            continue
        bounded = (
            _has_bounding_cut(grid, s, AD_BLOCK_BOUNDARY_TOLERANCE)
            and _has_bounding_cut(grid, e, AD_BLOCK_BOUNDARY_TOLERANCE)
        )
        conf = AD_BLOCK_BOUNDED_CONFIDENCE if bounded else AD_BLOCK_BASE_CONFIDENCE
        hits.append(RuleHit(s, e, "sponsorship", "ad_block", confidence=conf))

    return hits
```

- [ ] **Step 8: Run all rule_ad_block tests**

Run: `pytest tests/test_rules.py::TestRuleAdBlock -v`
Expected: 7 passed (5 from Task 8 + 2 new).

- [ ] **Step 9: Commit**

```bash
git add backend/pipeline/fusion/align.py backend/pipeline/fusion/rules.py tests/test_align.py tests/test_rules.py
git commit -m "feat(fusion): add MFCC-based audio_drift signal and use it in rule_ad_block"
```

---

## Task 14: Transcript-side CTA matching exposed to the grid

**Files:**
- Modify: `backend/pipeline/fusion/rules.py` (add `CTA_KEYWORDS` constant)
- Modify: `backend/pipeline/features/ocr_signals.py` (import the shared list)
- Modify: `backend/pipeline/features/text.py` (add cta group)
- Modify: `backend/pipeline/fusion/align.py` (expose `text_has_cta` per second)
- Modify: `backend/pipeline/fusion/rules.py:_has_any_commercial_signal` (also check `text_has_cta`)
- Modify: `tests/test_text.py`, `tests/test_align.py`

The CTA list lives in **one** place — `rules.py` — and both OCR and text-keyword paths import it. This keeps DRY.

- [ ] **Step 1: Add CTA_KEYWORDS to rules.py**

In `backend/pipeline/fusion/rules.py`, near the other keyword tables, add:

```python
# CTA phrases — high-precision, ad-specific. Used by both OCR (ocr_signals.py)
# and transcript matching (text.py). "subscribe" deliberately omitted (YouTube
# self-promo, not a clear ad signal).
CTA_KEYWORDS = [
    "shop now", "buy now", "order now", "order today", "available at",
    "available now", "limited time", "for a limited time", "offer ends",
    "introducing the", "new from", "use promo code", "use code",
    "download the app", "visit our store", "in stores now", "in stores today",
]
```

- [ ] **Step 2: Have ocr_signals.py import the shared list**

In `backend/pipeline/features/ocr_signals.py`, replace the existing local `_CTA_PHRASES` definition with:

```python
from backend.pipeline.fusion.rules import CTA_KEYWORDS as _CTA_PHRASES
```

(Delete the inline list — it's now imported.)

- [ ] **Step 3: Run existing OCR tests for regression**

Run: `pytest tests/test_ocr_signals.py -v`
Expected: all green (Task 2's CTA tests still pass — same phrase list).

- [ ] **Step 4: Write the failing text test**

Append to `tests/test_text.py`:

```python
def test_text_segments_match_cta_keywords(tmp_path, monkeypatch):
    """Transcript segments should expose 'cta:<phrase>' matches in matched_keywords."""
    import json
    from pathlib import Path
    from backend.pipeline.workspace import Workspace
    from backend.pipeline.features.text import extract_text_features
    from backend.pipeline.schemas import TextFeatures

    # Build a fake transcript.json with a CTA phrase.
    ws_root = tmp_path / "ws"
    ws_root.mkdir()
    transcript = {
        "language": "en", "duration_sec": 10.0, "model": "test",
        "full_text": "shop now and save",
        "segments": [
            {"id": 0, "start": 0.0, "end": 5.0, "text": "shop now and save"},
            {"id": 1, "start": 5.0, "end": 10.0, "text": "back to the topic"},
        ],
    }
    ws = Workspace(root=ws_root, video_id="x")
    ws.ensure()
    ws.transcript_path.write_text(json.dumps(transcript))

    extract_text_features(ws, device="cpu")
    feats = TextFeatures.model_validate(json.loads(ws.text_features_path.read_text()))

    seg0_kws = feats.segments[0].matched_keywords
    seg1_kws = feats.segments[1].matched_keywords
    assert any(kw.startswith("cta:") for kw in seg0_kws), f"expected cta hit; got {seg0_kws!r}"
    assert not any(kw.startswith("cta:") for kw in seg1_kws), f"unexpected cta hit; got {seg1_kws!r}"
```

(Adapt the Workspace constructor call to match actual signature — check existing tests in `test_text.py` for the right pattern.)

- [ ] **Step 5: Run test to verify it fails**

Run: `pytest tests/test_text.py::test_text_segments_match_cta_keywords -v`
Expected: FAIL — `cta:` group does not exist.

- [ ] **Step 6: Add cta group to text.py**

In `backend/pipeline/features/text.py`, modify the imports and `_KEYWORD_GROUPS`:

```python
from backend.pipeline.fusion.rules import (
    SPONSOR_KEYWORDS,
    INTRO_KEYWORDS,
    OUTRO_KEYWORDS,
    SELF_PROMO_KEYWORDS,
    RECAP_KEYWORDS,
    CTA_KEYWORDS,
)

DEFAULT_SIMILARITY_THRESHOLD = 0.35

_KEYWORD_GROUPS = {
    "sponsor": SPONSOR_KEYWORDS,
    "intro": INTRO_KEYWORDS,
    "outro": OUTRO_KEYWORDS,
    "self_promo": SELF_PROMO_KEYWORDS,
    "recap": RECAP_KEYWORDS,
    "cta": CTA_KEYWORDS,
}
```

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/test_text.py::test_text_segments_match_cta_keywords -v`
Expected: PASS.

- [ ] **Step 8: Write the failing align test**

Append to `tests/test_align.py`:

```python
def test_grid_includes_text_has_cta():
    import numpy as np
    from backend.pipeline.fusion.align import build_per_second_grid
    from backend.pipeline.schemas import (
        VisualFeatures, AudioFeatures, AudioFeatureSegment,
        TextFeatures, TextFeatureSegment,
    )

    text = TextFeatures(segments=[
        TextFeatureSegment(
            id=0, start=2.0, end=4.0, text="shop now",
            matched_keywords=["cta:shop now"],
        ),
        TextFeatureSegment(
            id=1, start=4.0, end=8.0, text="thanks",
            matched_keywords=[],
        ),
    ])
    audio = AudioFeatures(segments=[AudioFeatureSegment(start=float(t), end=float(t+1)) for t in range(10)])
    visual = VisualFeatures(frames=[])

    grid = build_per_second_grid(visual, audio, text, duration_sec=10.0)

    assert "text_has_cta" in grid
    assert grid["text_has_cta"][2] == 1
    assert grid["text_has_cta"][3] == 1
    assert grid["text_has_cta"][5] == 0
    assert grid["text_has_cta"][0] == 0
```

- [ ] **Step 9: Run test to verify it fails**

Run: `pytest tests/test_align.py::test_grid_includes_text_has_cta -v`
Expected: FAIL — `text_has_cta` not in grid.

- [ ] **Step 10: Wire text_has_cta into align.py**

In `backend/pipeline/fusion/align.py`, inside the existing text-loop section (where `has_text` and `text_sentence_id` are populated), add:

```python
    text_has_cta = np.zeros(T, dtype=np.int8)
    for seg in text.segments:
        s = max(0, int(np.floor(seg.start)))
        e = min(T, int(np.ceil(seg.end)))
        if any(kw.startswith("cta:") for kw in seg.matched_keywords):
            text_has_cta[s:e] = 1
        # ... existing has_text / text_sentence_id / text_sim assignments stay ...
    grid["text_has_cta"] = text_has_cta
```

- [ ] **Step 11: Run test to verify it passes**

Run: `pytest tests/test_align.py::test_grid_includes_text_has_cta -v`
Expected: PASS.

- [ ] **Step 12: Update `_has_any_commercial_signal` to also check transcript CTA**

In `backend/pipeline/fusion/rules.py`, modify `_has_any_commercial_signal` to include `text_has_cta`:

```python
def _has_any_commercial_signal(grid: Dict[str, np.ndarray], s: int, e: int) -> bool:
    """True iff at least one OCR or transcript commercial flag is set anywhere in [s, e)."""
    for key in (
        "has_url", "has_price", "has_phone", "has_cta", "has_brand_lockup",
        "text_has_cta",
    ):
        arr = grid.get(key)
        if arr is None:
            continue
        if arr[s:e].any():
            return True
    return False
```

- [ ] **Step 13: Add a rule test for transcript-CTA-only path**

Append to `tests/test_rules.py` inside `class TestRuleAdBlock`:

```python
    def test_transcript_cta_alone_can_satisfy_commercial_signal(self):
        import numpy as np
        from backend.pipeline.fusion.rules import rule_ad_block

        # 30s of visual drift, no OCR hits at all, but transcript CTA in the middle.
        T = 200
        g = self._build_grid(T=T, drift_high=[(50, 80)], ocr_hit_t=None, hard_cuts=[49, 81])
        g["text_has_cta"] = np.zeros(T, dtype=np.int8)
        g["text_has_cta"][55:60] = 1

        hits = rule_ad_block(g)
        assert len(hits) == 1
        assert hits[0].rule_name == "ad_block"
```

- [ ] **Step 14: Run all rule_ad_block tests**

Run: `pytest tests/test_rules.py::TestRuleAdBlock -v`
Expected: all 8 pass (5 original + 2 from Task 13 + 1 new).

- [ ] **Step 15: Commit**

```bash
git add backend/pipeline/fusion/rules.py backend/pipeline/features/ocr_signals.py backend/pipeline/features/text.py backend/pipeline/fusion/align.py tests/test_text.py tests/test_align.py tests/test_rules.py
git commit -m "feat: add transcript CTA matching as a fourth commercial signal in rule_ad_block"
```

---

## Updated end-to-end verification (replaces the old Task 10 success criteria)

When re-running Task 10 after Tasks 11-14, also verify:

- `audio_features.json` segments now have non-empty `mfcc` arrays
- `text_features.json` segments include at least one `cta:*` match (if the test video has any commercial CTA in transcript — Ad 3's "alliums / million dollars" likely won't, but Ad 1 might)
- The new metadata's sponsorship segments now have `triggered_rules` containing `ad_block` for the multi-signal hits
- Re-clearing visual *and* audio features before re-extraction:
  ```powershell
  Remove-Item workspace/test_004_2/features/visual_features.json -ErrorAction SilentlyContinue
  Remove-Item workspace/test_004_2/features/audio_features.json -ErrorAction SilentlyContinue
  Remove-Item workspace/test_004_2/features/text_features.json -ErrorAction SilentlyContinue
  ```

---

## Risks & Open Questions

1. **RapidOCR ONNX install** is a clean wheel (`rapidocr-onnxruntime` pulls only `onnxruntime` + numpy + opencv-python-headless), so there's no torch/CUDA conflict risk. If the wheel fails to build on the target platform, fall back to `pip install rapidocr-onnxruntime --no-build-isolation` or pin a known-good `onnxruntime` version (e.g. `onnxruntime==1.17.x`).
2. **OCR cost** with stride 3 + luminance gate on CPU: ~0.5 min for a 30 min video, ~1 min for 60 min, ~3 min for 120 min. Comfortably within the demo budget. If a corner case emerges where stride 3 misses a fast-flashing brand frame, lower `OCR_STRIDE_SEC` to 2 (50 % more cost). Do not lower below 2 without GPU acceleration.
3. **Style drift threshold (0.30)** is a guess based on typical CLIP cosine distances. After Task 10 we may need to tune. If FP increases, raise to 0.40; if recall drops, lower to 0.25.
4. **Animation host** edge case: cartoons often have wildly varying CLIP embeddings second-to-second already, which could elevate baseline drift and reduce signal-to-noise. If test videos with cartoon hosts show baseline drift > 0.30, switch from absolute threshold to "z-score of drift relative to 5-min running mean."
5. **Ad 1 in test_004_2** (Will Smith comedy clip) likely has *little* OCR text — actors talking, no logo on screen except briefly. The audio MFCC drift (Task 11-13) is the primary backstop here: a TV studio recording vs a lecture-room recording differs strongly in mid/high-cepstrum coefficients, so audio drift should fire even when visual drift is moderate. If both still fail, fall back to "OCR signal OR transcript CTA OR strong drift on either modality + bounding cuts" in a follow-up.
6. **MFCC scale and drift threshold**: MFCC magnitudes are unbounded (typically -100..+100 for c0; -30..+30 for c1+), unlike CLIP's normalized embeddings. Cosine distance after L2-normalisation still yields values in [0, 2], but the *threshold* of 0.30 may not be optimal. After Task 13 verifies it produces non-trivial drift values on test_004_2, sanity-check by computing `mean(audio_drift)` and `std(audio_drift)` over the whole video; if mean drift is already > 0.20 in lecture regions, raise `AD_BLOCK_DRIFT_THRESHOLD` (which is shared with visual — may need to split into two constants).
7. **Splitting AD_BLOCK_DRIFT_THRESHOLD**: if visual and audio drift have very different scales after Task 13, split into `AD_BLOCK_VISUAL_DRIFT_THRESHOLD` and `AD_BLOCK_AUDIO_DRIFT_THRESHOLD` and update `rule_ad_block` accordingly. This is a 5-min change; defer until empirical evidence demands it.
8. **Transcript CTA precision**: the CTA list (Task 14) is conservative on purpose (no "subscribe", no "click here"). If false-positive transcript CTA matches show up in lecture content, it's likely the lecturer literally saying "shop now" as an example — this is rare enough to ignore initially. If common, gate `text_has_cta` behind co-occurrence with at least one drift signal at the same second.
9. **`librosa.feature.mfcc` cost**: ~50ms per second of audio on CPU. For a 30-min video that's ~90 seconds of extra compute on top of existing audio extraction. Acceptable; no need to batch.
