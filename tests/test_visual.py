"""
Tests for backend/pipeline/features/visual.py

Happy cases: correct feature values on synthetic frames.
Sad cases: empty frames_cache, corrupt image, single frame.
"""
from __future__ import annotations
import json
import cv2
import numpy as np
import pytest
from pathlib import Path

from backend.pipeline.workspace import Workspace
from backend.pipeline.schemas import VisualFeatures
from backend.pipeline.features.visual import (
    SCENE_LABELS,
    _detect_black_frame,
    _compute_motion_intensity,
    _compute_chroma_diff,
    _compute_dct_hf_energy,
    extract_visual_features,
)


# ── Unit tests for helper functions (no model needed) ────────────────────────

class TestDetectBlackFrame:
    def test_all_black_is_detected(self):
        gray = np.zeros((100, 100), dtype=np.uint8)
        mean_lum, var_lum, is_black = _detect_black_frame(gray)
        assert mean_lum == pytest.approx(0.0)
        assert var_lum == pytest.approx(0.0)
        assert is_black is True

    def test_all_white_is_not_black(self):
        gray = np.full((100, 100), 255, dtype=np.uint8)
        _, _, is_black = _detect_black_frame(gray)
        assert is_black is False

    def test_mid_gray_is_not_black(self):
        gray = np.full((100, 100), 128, dtype=np.uint8)
        _, _, is_black = _detect_black_frame(gray)
        assert is_black is False

    def test_near_black_high_variance_not_flagged(self):
        # A low-luminance frame with a bright spot should NOT be flagged.
        gray = np.zeros((100, 100), dtype=np.uint8)
        gray[50, 50] = 200
        _, _, is_black = _detect_black_frame(gray)
        assert is_black is False


class TestMotionIntensity:
    def test_identical_frames_zero_motion(self):
        frame = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        motion = _compute_motion_intensity(frame, frame)
        assert motion == pytest.approx(0.0)

    def test_inverse_frames_max_motion(self):
        prev = np.zeros((64, 64), dtype=np.uint8)
        curr = np.full((64, 64), 255, dtype=np.uint8)
        motion = _compute_motion_intensity(prev, curr)
        assert motion == pytest.approx(255.0)

    def test_partial_change_proportional(self):
        prev = np.zeros((10, 10), dtype=np.uint8)
        curr = np.zeros((10, 10), dtype=np.uint8)
        curr[:5, :] = 100  # half the pixels change by 100
        motion = _compute_motion_intensity(prev, curr)
        assert 40.0 < motion < 60.0  # roughly 50


class TestChromaDiff:
    def test_identical_frames_zero_diff(self):
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        diff = _compute_chroma_diff(frame, frame)
        assert diff == pytest.approx(0.0, abs=1e-5)

    def test_different_colors_nonzero_diff(self):
        red = np.zeros((64, 64, 3), dtype=np.uint8)
        red[:, :, 2] = 255  # BGR: pure red
        blue = np.zeros((64, 64, 3), dtype=np.uint8)
        blue[:, :, 0] = 255  # BGR: pure blue
        diff = _compute_chroma_diff(red, blue)
        assert diff > 0.1  # substantial chroma difference

    def test_same_hue_different_brightness_low_diff(self):
        dark = np.full((64, 64, 3), 50, dtype=np.uint8)
        bright = np.full((64, 64, 3), 200, dtype=np.uint8)
        diff = _compute_chroma_diff(dark, bright)
        # Same neutral gray → chroma channels barely differ
        assert diff < 0.05


class TestDctHfEnergy:
    def test_blank_frame_near_zero_energy(self):
        gray = np.full((64, 64), 128, dtype=np.uint8)
        energy = _compute_dct_hf_energy(gray)
        assert energy < 1.0  # uniform frame has minimal HF content

    def test_noise_frame_high_energy(self):
        np.random.seed(42)
        gray = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        energy = _compute_dct_hf_energy(gray)
        assert energy > 1.0  # random noise is very high-frequency

    def test_returns_zero_on_tiny_frame(self):
        gray = np.zeros((4, 4), dtype=np.uint8)  # smaller than one 8×8 block
        energy = _compute_dct_hf_energy(gray)
        assert energy == pytest.approx(0.0)


class TestSceneLabels:
    def test_all_taxonomy_covering_labels_present(self):
        """All prompts needed by classify.py to reach the full 10-class taxonomy must exist."""
        required_substrings = [
            "presentation slide",
            "person talking",
            "screen recording",
            "blank screen",
            "end credits",
            "title card",
            "advertisement",
            "sponsor",
            "video game",
        ]
        label_text = " ".join(SCENE_LABELS).lower()
        for substr in required_substrings:
            assert substr in label_text, f"SCENE_LABELS missing coverage for: '{substr}'"

    def test_no_duplicate_labels(self):
        assert len(SCENE_LABELS) == len(set(SCENE_LABELS))


# ── Integration tests (require frames on disk; uses phase1_workspace fixture) ─

class TestExtractVisualFeatures:
    @pytest.mark.slow
    def test_output_file_created(self, phase1_workspace):
        out = extract_visual_features(phase1_workspace, device="cpu")
        assert out.exists()

    @pytest.mark.slow
    def test_output_validates_against_schema(self, phase1_workspace):
        extract_visual_features(phase1_workspace, device="cpu")
        data = json.loads(phase1_workspace.visual_features_path.read_text())
        features = VisualFeatures.model_validate(data)
        assert len(features.frames) > 0

    @pytest.mark.slow
    def test_each_frame_has_all_clip_labels(self, phase1_workspace):
        extract_visual_features(phase1_workspace, device="cpu")
        data = json.loads(phase1_workspace.visual_features_path.read_text())
        features = VisualFeatures.model_validate(data)
        for frame in features.frames:
            assert set(frame.clip_labels.keys()) == set(SCENE_LABELS)

    @pytest.mark.slow
    def test_clip_probabilities_sum_to_one(self, phase1_workspace):
        extract_visual_features(phase1_workspace, device="cpu")
        data = json.loads(phase1_workspace.visual_features_path.read_text())
        features = VisualFeatures.model_validate(data)
        for frame in features.frames:
            total = sum(frame.clip_labels.values())
            assert total == pytest.approx(1.0, abs=1e-4)

    @pytest.mark.slow
    def test_new_fields_present_and_non_negative(self, phase1_workspace):
        extract_visual_features(phase1_workspace, device="cpu")
        data = json.loads(phase1_workspace.visual_features_path.read_text())
        features = VisualFeatures.model_validate(data)
        for frame in features.frames:
            assert frame.mean_luminance >= 0.0
            assert frame.luminance_variance >= 0.0
            assert frame.motion_intensity >= 0.0
            assert frame.chroma_diff >= 0.0
            assert frame.dct_hf_energy >= 0.0

    @pytest.mark.slow
    def test_first_frame_has_zero_motion_and_zero_hist_diff(self, phase1_workspace):
        extract_visual_features(phase1_workspace, device="cpu")
        data = json.loads(phase1_workspace.visual_features_path.read_text())
        features = VisualFeatures.model_validate(data)
        first = features.frames[0]
        assert first.hist_diff_to_previous == pytest.approx(0.0)
        assert first.motion_intensity == pytest.approx(0.0)
        assert first.chroma_diff == pytest.approx(0.0)

    @pytest.mark.slow
    def test_cache_skips_on_second_call(self, phase1_workspace):
        extract_visual_features(phase1_workspace, device="cpu")
        mtime1 = phase1_workspace.visual_features_path.stat().st_mtime
        extract_visual_features(phase1_workspace, device="cpu")
        mtime2 = phase1_workspace.visual_features_path.stat().st_mtime
        assert mtime1 == mtime2  # file not rewritten

    # ── Sad cases ─────────────────────────────────────────────────────────────

    def test_empty_frames_cache_returns_early(self, tmp_path):
        """No frames → function returns the (non-existent) output path without crashing."""
        ws = Workspace.for_video(Path("dummy.mp4"), root=tmp_path / "workspace")
        ws.ensure()
        # frames_dir exists but is empty
        out = extract_visual_features(ws, device="cpu")
        assert not out.exists()  # nothing was written

    def test_corrupt_image_is_skipped(self, tmp_path, phase1_workspace):
        """A corrupt JPEG in the cache should be skipped gracefully."""
        import shutil
        # Copy the real workspace frames to a new location so we don't pollute it
        ws2 = Workspace.for_video(Path("corrupt_test.mp4"), root=tmp_path / "ws2")
        ws2.ensure()
        for f in phase1_workspace.frames_dir.glob("frame_*.jpg"):
            shutil.copy(f, ws2.frames_dir / f.name)
        # Overwrite one frame with garbage bytes
        bad = sorted(ws2.frames_dir.glob("frame_*.jpg"))[0]
        bad.write_bytes(b"\xff\xd8garbage")

        out = extract_visual_features(ws2, device="cpu")
        data = json.loads(out.read_text())
        features = VisualFeatures.model_validate(data)
        # One fewer frame; no crash
        good_count = len(list(phase1_workspace.frames_dir.glob("frame_*.jpg")))
        assert len(features.frames) == good_count - 1
