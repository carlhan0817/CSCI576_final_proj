from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import torch

from backend.pipeline.logging_setup import get_logger
from backend.pipeline.schemas import AudioFeatures, AudioFeatureSegment
from backend.pipeline.workspace import Workspace

# ── Classification thresholds ─────────────────────────────────────────────────
SILENCE_RMS_THRESHOLD = 0.001   # RMS below this → silence
SPEECH_ZCR_THRESHOLD = 0.10     # ZCR above this → speech (voiced content)
MUSIC_RMS_THRESHOLD = 0.01      # non-speech, non-silence with energy → music


def _load_silero_vad(device: str = "cpu"):
    """Load Silero VAD from PyTorch Hub (cached after first download)."""
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    return model.to(device), utils


def _classify_audio_class(rms: float, zcr: float, centroid: float) -> str:
    """
    Heuristic 4-class audio classifier.

    Decision tree (in order):
      1. Low RMS              → silence   (no meaningful signal)
      2. High ZCR             → speech    (voiced consonants/vowels)
      3. Substantial energy   → music     (instrument/tonal content)
      4. Fallback             → noise
    """
    if rms < SILENCE_RMS_THRESHOLD:
        return "silence"
    if zcr > SPEECH_ZCR_THRESHOLD:
        return "speech"
    if rms > MUSIC_RMS_THRESHOLD:
        return "music"
    return "noise"


def _compute_second_features(chunk: np.ndarray, sr: int = 16000) -> dict:
    """
    Compute all spectral features for a 1-second audio chunk.
    Returns a dict with keys: rms, centroid, bandwidth, zcr, entropy.
    """
    if len(chunk) == 0:
        return dict(rms=0.0, centroid=0.0, bandwidth=0.0, zcr=0.0, entropy=0.0)

    # RMS energy
    rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))

    # Short-time Fourier transform (512-point FFT, 50 % overlap)
    n_fft = min(512, len(chunk))
    hop = n_fft // 2
    S = np.abs(librosa.stft(chunk, n_fft=n_fft, hop_length=hop))  # (freq_bins, frames)

    mean_spectrum = S.mean(axis=1) + 1e-10  # shape: (freq_bins,)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Spectral centroid (Hz)
    centroid = float(np.sum(freqs * mean_spectrum) / mean_spectrum.sum())

    # Spectral bandwidth (Hz): RMS spread around centroid
    bandwidth = float(
        np.sqrt(np.sum(((freqs - centroid) ** 2) * mean_spectrum) / mean_spectrum.sum())
    )

    # Zero-crossing rate
    zcr = float(librosa.feature.zero_crossing_rate(chunk, hop_length=hop)[0].mean())

    # [V2.1] Spectral entropy — Shannon entropy of the normalized power spectrum.
    # A pure tone has low entropy (energy concentrated at one frequency).
    # White noise or broadband signals have high entropy.
    # Speech sits between: harmonic structure but spread across formants.
    ps = mean_spectrum / mean_spectrum.sum()
    entropy = float(-np.sum(ps * np.log2(ps + 1e-12)))

    return dict(rms=rms, centroid=centroid, bandwidth=bandwidth, zcr=zcr, entropy=entropy)


def extract_audio_features(workspace: Workspace, device: str = "cpu") -> Path:
    """
    Reads audio_processed.wav, runs Silero VAD + librosa spectral analysis,
    and writes one AudioFeatureSegment per second to features/audio_features.json.

    Outputs per second:
    - is_speech    (Silero VAD)
    - rms_energy
    - spectral_centroid (Hz)
    - spectral_bandwidth (Hz)
    - zero_crossing_rate
    - spectral_entropy  [V2.1]
    - audio_class  (speech | music | silence | noise)
    """
    log = get_logger(workspace.log_path)
    out_path = workspace.audio_features_path

    if out_path.exists():
        log.info("Audio features already exist. Skipping.")
        return out_path

    if not workspace.audio_path.exists():
        raise FileNotFoundError(f"audio_processed.wav not found: {workspace.audio_path}")

    # ── Load audio with librosa ───────────────────────────────────────────────
    log.info("Stage 2 (Audio): Loading WAV with librosa...")
    y, sr = librosa.load(str(workspace.audio_path), sr=16000, mono=True)
    duration_sec = len(y) / sr
    T = int(np.ceil(duration_sec))

    if T == 0:
        log.warning("Audio has zero duration. Writing empty feature file.")
        out_path.write_text(AudioFeatures(segments=[]).model_dump_json(indent=2))
        return out_path

    # ── Run Silero VAD ────────────────────────────────────────────────────────
    log.info("Stage 2 (Audio): Running Silero VAD...")
    vad_model, utils = _load_silero_vad(device)
    get_speech_timestamps, _, read_audio, _, _ = utils

    wav_tensor = read_audio(str(workspace.audio_path)).to(device)
    speech_timestamps = get_speech_timestamps(
        wav_tensor, vad_model, sampling_rate=16000, return_seconds=True
    )

    # Build per-second is_speech flag from VAD intervals
    is_speech_arr = np.zeros(T, dtype=bool)
    for ts in speech_timestamps:
        s = max(0, int(np.floor(ts["start"])))
        e = min(T, int(np.ceil(ts["end"])))
        is_speech_arr[s:e] = True

    # Release VAD model before spectral computation
    del vad_model, utils, wav_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Per-second spectral features ──────────────────────────────────────────
    log.info("Stage 2 (Audio): Computing per-second spectral features...")
    segments: list[AudioFeatureSegment] = []

    for t in range(T):
        start_sample = t * sr
        end_sample = min((t + 1) * sr, len(y))
        chunk = y[start_sample:end_sample]

        feat = _compute_second_features(chunk, sr=sr)
        audio_class = _classify_audio_class(feat["rms"], feat["zcr"], feat["centroid"])

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
            )
        )

    result = AudioFeatures(segments=segments)
    out_path.write_text(result.model_dump_json(indent=2))
    log.info("Audio features saved to %s (%d seconds).", out_path.name, len(segments))
    return out_path
