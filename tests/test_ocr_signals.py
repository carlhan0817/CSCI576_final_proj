"""Tests for backend/pipeline/features/ocr_signals.py — pattern detection only."""
from __future__ import annotations

import os

import cv2
import numpy as np
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
