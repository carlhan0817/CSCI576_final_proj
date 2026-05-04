"""
Tests for backend/pipeline/features/audio.py

Happy cases: correct feature values on known synthetic audio.
Sad cases: zero-duration audio, silent audio, missing file.
"""
from __future__ import annotations
import json
import wave
import math
import numpy as np
import pytest
from pathlib import Path

from backend.pipeline.workspace import Workspace
from backend.pipeline.schemas import AudioFeatures
from backend.pipeline.features.audio import (
    _classify_audio_class,
    _compute_second_features,
    extract_audio_features,
    SILENCE_RMS_THRESHOLD,
    SPEECH_ZCR_THRESHOLD,
    MUSIC_RMS_THRESHOLD,
)


def _write_wav(path: Path, samples: np.ndarray, sr: int = 16000) -> None:
    int_samples = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(int_samples.tobytes())


# ── Unit tests for _classify_audio_class ─────────────────────────────────────

class TestClassifyAudioClass:
    def test_silence(self):
        assert _classify_audio_class(rms=0.0, zcr=0.0, centroid=0.0) == "silence"

    def test_below_rms_threshold_is_silence(self):
        assert _classify_audio_class(
            rms=SILENCE_RMS_THRESHOLD / 2, zcr=0.2, centroid=2000
        ) == "silence"

    def test_high_zcr_is_speech(self):
        rms = SILENCE_RMS_THRESHOLD * 10
        zcr = SPEECH_ZCR_THRESHOLD * 1.5
        assert _classify_audio_class(rms=rms, zcr=zcr, centroid=1500) == "speech"

    def test_low_zcr_high_rms_is_music(self):
        rms = MUSIC_RMS_THRESHOLD * 5
        zcr = SPEECH_ZCR_THRESHOLD * 0.3
        assert _classify_audio_class(rms=rms, zcr=zcr, centroid=2000) == "music"

    def test_very_low_rms_non_speech_is_noise(self):
        rms = SILENCE_RMS_THRESHOLD * 3
        zcr = SPEECH_ZCR_THRESHOLD * 0.3
        assert _classify_audio_class(rms=rms, zcr=zcr, centroid=500) == "noise"


# ── Unit tests for _compute_second_features ──────────────────────────────────

class TestComputeSecondFeatures:
    SR = 16000

    def test_silence_has_near_zero_rms(self):
        chunk = np.zeros(self.SR, dtype=np.float32)
        feat = _compute_second_features(chunk, sr=self.SR)
        assert feat["rms"] < SILENCE_RMS_THRESHOLD

    def test_sine_wave_rms_near_amplitude_over_sqrt2(self):
        t = np.arange(self.SR) / self.SR
        amplitude = 0.8
        chunk = (amplitude * np.sin(2 * math.pi * 440 * t)).astype(np.float32)
        feat = _compute_second_features(chunk, sr=self.SR)
        expected_rms = amplitude / math.sqrt(2)
        assert feat["rms"] == pytest.approx(expected_rms, rel=0.05)

    def test_sine_wave_centroid_near_440hz(self):
        t = np.arange(self.SR) / self.SR
        chunk = (0.5 * np.sin(2 * math.pi * 440 * t)).astype(np.float32)
        feat = _compute_second_features(chunk, sr=self.SR)
        assert 350 < feat["centroid"] < 550

    def test_all_features_non_negative(self):
        np.random.seed(0)
        chunk = np.random.randn(self.SR).astype(np.float32) * 0.1
        feat = _compute_second_features(chunk, sr=self.SR)
        for key, val in feat.items():
            assert val >= 0.0, f"Feature {key} is negative: {val}"

    def test_empty_chunk_returns_zeros(self):
        feat = _compute_second_features(np.array([], dtype=np.float32), sr=self.SR)
        assert all(v == 0.0 for v in feat.values())

    def test_noise_entropy_higher_than_tone(self):
        t = np.arange(self.SR) / self.SR
        tone = (0.3 * np.sin(2 * math.pi * 440 * t)).astype(np.float32)
        noise = np.random.randn(self.SR).astype(np.float32) * 0.3
        feat_tone = _compute_second_features(tone, sr=self.SR)
        feat_noise = _compute_second_features(noise, sr=self.SR)
        # White noise has much higher spectral entropy than a pure tone
        assert feat_noise["entropy"] > feat_tone["entropy"]


# ── Integration tests ─────────────────────────────────────────────────────────

class TestExtractAudioFeatures:
    @pytest.mark.slow
    def test_output_file_created(self, phase1_workspace):
        out = extract_audio_features(phase1_workspace, device="cpu")
        assert out.exists()

    @pytest.mark.slow
    def test_output_validates_against_schema(self, phase1_workspace):
        extract_audio_features(phase1_workspace, device="cpu")
        data = json.loads(phase1_workspace.audio_features_path.read_text())
        features = AudioFeatures.model_validate(data)
        assert len(features.segments) > 0

    @pytest.mark.slow
    def test_one_segment_per_second(self, phase1_workspace):
        extract_audio_features(phase1_workspace, device="cpu")
        data = json.loads(phase1_workspace.audio_features_path.read_text())
        features = AudioFeatures.model_validate(data)
        for i, seg in enumerate(features.segments):
            assert seg.start == pytest.approx(float(i))
            assert seg.end == pytest.approx(float(i + 1))

    @pytest.mark.slow
    def test_sine_wave_has_nonzero_rms(self, phase1_workspace):
        """The synthetic MP4 has a 440 Hz sine tone; all FULL seconds should have energy.

        The last segment may be a sub-second tail (e.g. 0.018s of samples); we
        exclude it because RMS over a near-empty buffer is meaningless.
        """
        extract_audio_features(phase1_workspace, device="cpu")
        data = json.loads(phase1_workspace.audio_features_path.read_text())
        features = AudioFeatures.model_validate(data)
        # Skip the trailing partial-second segment (audio ends mid-second).
        full_seconds = features.segments[:-1] if len(features.segments) > 1 else features.segments
        assert len(full_seconds) > 0
        rms_values = [s.rms_energy for s in full_seconds]
        assert all(r > SILENCE_RMS_THRESHOLD for r in rms_values)

    @pytest.mark.slow
    def test_sine_wave_not_classified_as_silence(self, phase1_workspace):
        extract_audio_features(phase1_workspace, device="cpu")
        data = json.loads(phase1_workspace.audio_features_path.read_text())
        features = AudioFeatures.model_validate(data)
        # Skip the trailing partial-second segment (see test_sine_wave_has_nonzero_rms).
        full_seconds = features.segments[:-1] if len(features.segments) > 1 else features.segments
        for seg in full_seconds:
            assert seg.audio_class != "silence"

    @pytest.mark.slow
    def test_spectral_entropy_present_and_non_negative(self, phase1_workspace):
        extract_audio_features(phase1_workspace, device="cpu")
        data = json.loads(phase1_workspace.audio_features_path.read_text())
        features = AudioFeatures.model_validate(data)
        for seg in features.segments:
            assert seg.spectral_entropy >= 0.0

    @pytest.mark.slow
    def test_cache_skips_on_second_call(self, phase1_workspace):
        extract_audio_features(phase1_workspace, device="cpu")
        mtime1 = phase1_workspace.audio_features_path.stat().st_mtime
        extract_audio_features(phase1_workspace, device="cpu")
        mtime2 = phase1_workspace.audio_features_path.stat().st_mtime
        assert mtime1 == mtime2

    # ── Sad cases ─────────────────────────────────────────────────────────────

    def test_silent_audio_all_segments_are_silence(self, tmp_path, silent_wav):
        ws = Workspace.for_video(Path("silent.mp4"), root=tmp_path / "ws")
        ws.ensure()
        import shutil
        shutil.copy(silent_wav, ws.audio_path)

        out = extract_audio_features(ws, device="cpu")
        data = json.loads(out.read_text())
        features = AudioFeatures.model_validate(data)
        assert len(features.segments) > 0
        for seg in features.segments:
            assert seg.audio_class == "silence"
            assert seg.rms_energy < SILENCE_RMS_THRESHOLD

    def test_missing_wav_raises_file_not_found(self, tmp_path):
        ws = Workspace.for_video(Path("missing.mp4"), root=tmp_path / "ws")
        ws.ensure()
        with pytest.raises(FileNotFoundError, match="audio_processed.wav"):
            extract_audio_features(ws, device="cpu")

    def test_empty_segments_on_zero_duration(self, tmp_path):
        """A WAV with zero frames should produce an empty segment list."""
        ws = Workspace.for_video(Path("zero.mp4"), root=tmp_path / "ws")
        ws.ensure()
        _write_wav(ws.audio_path, np.array([], dtype=np.float32))

        out = extract_audio_features(ws, device="cpu")
        data = json.loads(out.read_text())
        features = AudioFeatures.model_validate(data)
        assert len(features.segments) == 0
