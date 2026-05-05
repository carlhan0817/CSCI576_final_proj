"""Phase 2: OCR-derived commercial-intent signals.

This module is split into two parts:

    * Pattern detection (`detect_commercial_patterns`) — pure-Python regex over
      strings. Cheap, deterministic, easy to test.
    * Frame-level OCR (`extract_text`) — wraps RapidOCR. Heavy import, lazily
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

# CTA phrases — centralized in fusion.rules.CTA_KEYWORDS so OCR and transcript
# matching share the same list (DRY). "subscribe" is intentionally omitted there.
from backend.pipeline.fusion.rules import CTA_KEYWORDS as _CTA_PHRASES

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
