"""
Tests for backend/pipeline/fusion/align.py (Phase 3: time-grid alignment).

These verify that visual @ 1 FPS, audio @ 1 Hz, and sentence-level text are
correctly resampled onto a unified per-second grid.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pytest

from backend.pipeline.workspace import Workspace
from backend.pipeline.schemas import (
    AudioFeatures, AudioFeatureSegment,
    VisualFeatures, VisualFrameFeature,
    TextFeatures, TextFeatureSegment,
)
from backend.pipeline.fusion.align import build_per_second_grid, load_phase2_features


def _audio(n: int, **overrides) -> AudioFeatures:
    """Build n one-second audio segments starting at t=0."""
    segs = []
    for t in range(n):
        seg = AudioFeatureSegment(
            start=float(t),
            end=float(t + 1),
            is_speech=overrides.get("is_speech", False),
            rms_energy=overrides.get("rms_energy", 0.0),
            spectral_centroid=overrides.get("spectral_centroid", 0.0),
            spectral_bandwidth=overrides.get("spectral_bandwidth", 0.0),
            zero_crossing_rate=overrides.get("zero_crossing_rate", 0.0),
            spectral_entropy=overrides.get("spectral_entropy", 0.0),
            audio_class=overrides.get("audio_class", "silence"),
        )
        segs.append(seg)
    return AudioFeatures(segments=segs)


def _visual(n: int, clip_labels: dict[str, float] | None = None) -> VisualFeatures:
    """Build n frames at 1-second intervals starting at t=0."""
    if clip_labels is None:
        clip_labels = {"person talking": 0.5}
    frames = [
        VisualFrameFeature(
            frame_index=i,
            timestamp_sec=float(i),
            hist_diff_to_previous=0.0,
            is_hard_cut=False,
            clip_labels=clip_labels,
            mean_luminance=128.0,
            luminance_variance=100.0,
            is_black_frame=False,
            motion_intensity=0.0,
            chroma_diff=0.0,
            dct_hf_energy=0.0,
        )
        for i in range(n)
    ]
    return VisualFeatures(frames=frames)


def _text(segments: list[dict]) -> TextFeatures:
    return TextFeatures(segments=[TextFeatureSegment(**s) for s in segments])


class TestBuildGridShape:
    def test_grid_length_matches_duration(self):
        grid = build_per_second_grid(_visual(5), _audio(5), _text([]), duration_sec=5.0)
        assert len(grid["is_speech"]) == 5

    def test_non_integer_duration_rounded_up(self):
        # ceil(7.3) = 8
        grid = build_per_second_grid(_visual(7), _audio(7), _text([]), duration_sec=7.3)
        assert len(grid["is_speech"]) == 8

    def test_required_columns_present(self):
        grid = build_per_second_grid(_visual(3), _audio(3), _text([]), duration_sec=3.0)
        for key in ["is_speech", "rms_energy", "hist_diff", "is_hard_cut",
                    "has_text", "text_sim_to_next", "is_music"]:
            assert key in grid


class TestAudioAlignment:
    def test_speech_flag_per_second(self):
        audio = AudioFeatures(segments=[
            AudioFeatureSegment(start=0, end=1, is_speech=True, rms_energy=0.5),
            AudioFeatureSegment(start=1, end=2, is_speech=False, rms_energy=0.0),
            AudioFeatureSegment(start=2, end=3, is_speech=True, rms_energy=0.4),
        ])
        grid = build_per_second_grid(_visual(3), audio, _text([]), duration_sec=3.0)
        np.testing.assert_array_equal(grid["is_speech"], np.array([1, 0, 1], dtype=np.int8))

    def test_rms_carried_through(self):
        audio = AudioFeatures(segments=[
            AudioFeatureSegment(start=0, end=1, rms_energy=0.7),
            AudioFeatureSegment(start=1, end=2, rms_energy=0.001),
        ])
        grid = build_per_second_grid(_visual(2), audio, _text([]), duration_sec=2.0)
        assert grid["rms_energy"][0] == pytest.approx(0.7)
        assert grid["rms_energy"][1] == pytest.approx(0.001)

    def test_audio_class_music_translated(self):
        audio = AudioFeatures(segments=[
            AudioFeatureSegment(start=0, end=1, audio_class="music"),
            AudioFeatureSegment(start=1, end=2, audio_class="speech"),
        ])
        grid = build_per_second_grid(_visual(2), audio, _text([]), duration_sec=2.0)
        np.testing.assert_array_equal(grid["is_music"], np.array([1, 0], dtype=np.int8))


class TestVisualAlignment:
    def test_clip_labels_become_grid_columns(self):
        labels = {"person talking": 0.4, "advertisement": 0.6}
        visual = _visual(2, clip_labels=labels)
        grid = build_per_second_grid(visual, _audio(2), _text([]), duration_sec=2.0)
        assert "clip_person talking" in grid
        assert "clip_advertisement" in grid
        np.testing.assert_array_almost_equal(grid["clip_advertisement"], [0.6, 0.6])

    def test_hard_cuts_recorded(self):
        frames = [
            VisualFrameFeature(frame_index=0, timestamp_sec=0.0, is_hard_cut=False),
            VisualFrameFeature(frame_index=1, timestamp_sec=1.0, is_hard_cut=True),
            VisualFrameFeature(frame_index=2, timestamp_sec=2.0, is_hard_cut=False),
        ]
        visual = VisualFeatures(frames=frames)
        grid = build_per_second_grid(visual, _audio(3), _text([]), duration_sec=3.0)
        np.testing.assert_array_equal(grid["is_hard_cut"], np.array([0, 1, 0], dtype=np.int8))

    def test_visual_with_no_frames_uses_default_zeros(self):
        # Empty visual: no clip columns at all, but other visual columns exist as zeros.
        visual = VisualFeatures(frames=[])
        grid = build_per_second_grid(visual, _audio(3), _text([]), duration_sec=3.0)
        assert "hist_diff" in grid
        assert (grid["hist_diff"] == 0).all()
        # No clip_* columns when there are no frames.
        assert not any(k.startswith("clip_") for k in grid.keys())


class TestTextAlignment:
    def test_has_text_set_for_covered_seconds(self):
        text = _text([
            {"id": 0, "start": 0.0, "end": 2.0, "text": "first", "similarity_to_next": 0.4, "is_potential_boundary": False},
            {"id": 1, "start": 2.0, "end": 4.0, "text": "second"},
        ])
        grid = build_per_second_grid(_visual(5), _audio(5), text, duration_sec=5.0)
        np.testing.assert_array_equal(grid["has_text"], np.array([1, 1, 1, 1, 0], dtype=np.int8))

    def test_text_sim_propagated(self):
        text = _text([
            {"id": 0, "start": 0.0, "end": 2.0, "text": "a", "similarity_to_next": 0.2, "is_potential_boundary": True},
            {"id": 1, "start": 2.0, "end": 3.0, "text": "b", "similarity_to_next": None, "is_potential_boundary": False},
        ])
        grid = build_per_second_grid(_visual(3), _audio(3), text, duration_sec=3.0)
        assert grid["text_sim_to_next"][0] == pytest.approx(0.2)
        assert grid["text_sim_to_next"][1] == pytest.approx(0.2)
        # Last segment had None → NaN
        assert np.isnan(grid["text_sim_to_next"][2])

    def test_sentence_id_recorded(self):
        text = _text([
            {"id": 0, "start": 0.0, "end": 2.0, "text": "a"},
            {"id": 1, "start": 2.0, "end": 4.0, "text": "b"},
        ])
        grid = build_per_second_grid(_visual(5), _audio(5), text, duration_sec=5.0)
        np.testing.assert_array_equal(
            grid["text_sentence_id"][:4], np.array([0, 0, 1, 1], dtype=np.int32)
        )
        assert grid["text_sentence_id"][4] == -1


class TestLoadPhase2Features:
    def test_loads_all_three(self, tmp_path):
        ws = Workspace.for_video(Path("dummy.mp4"), root=tmp_path / "ws")
        ws.ensure()
        ws.audio_features_path.write_text(_audio(2).model_dump_json())
        ws.visual_features_path.write_text(_visual(2).model_dump_json())
        ws.text_features_path.write_text(_text([]).model_dump_json())

        visual, audio, text = load_phase2_features(ws)
        assert len(visual.frames) == 2
        assert len(audio.segments) == 2
        assert text.segments == []