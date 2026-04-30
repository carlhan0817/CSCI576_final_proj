# Phase 2: Feature Extraction — Fixes & Implementation (V2.1)

**Addresses:** `needed_to_fix.md §4.2` — all four subsections  
**Date:** 2026-04-30  
**Implemented:** 2026-04-30  
**Demo deadline:** 2026-05-06  
**Status:** ✅ All changes implemented into source files  

---

## Table of Contents

1. [Prerequisite: Rename the `features ` Directory](#0-prerequisite-rename-the-features--directory)
2. [New Dependencies](#1-new-dependencies)
3. [Schema Changes (`schemas.py`)](#2-schema-changes-schemaspy)
4. [Workspace Addition (`workspace.py`)](#3-workspace-addition-workspacepy)
5. [§4.2.1 Visual Pipeline (`features/visual.py`)](#4-421-visual-pipeline-featuresvisualpy)
6. [§4.2.2 Audio Pipeline (`features/audio.py`)](#5-422-audio-pipeline-featuresaudiopy)
7. [§4.2.3 Text Pipeline (`features/text.py`)](#6-423-text-pipeline-featurestextpy)
8. [§4.2.4 CLI Orchestrator (`features/run.py`)](#7-424-cli-orchestrator-featuresrunpy)
9. [Downstream: `align.py` Update](#8-downstream-alignpy-update)
10. [Test Suite](#9-test-suite)

---

## 0. Prerequisite: Rename the `features ` Directory

The directory is currently named `backend/pipeline/features ` (trailing space). This breaks Python imports on most systems. Before anything else:

```bash
# From the repo root
mv "backend/pipeline/features " "backend/pipeline/features"
```

All imports in `visual.py`, `audio.py`, and `text.py` already use the correct path — the rename is the only step needed. Add `backend/pipeline/features/__init__.py` (empty) if it doesn't exist.

---

## 1. New Dependencies

Add `librosa` to `pyproject.toml`. No other new top-level package is required — `scipy` is already pulled in transitively by `scikit-learn`, and `cv2.dct()` covers the DCT requirement without needing `scipy.fft`.

```toml
# pyproject.toml — add inside [project] dependencies list
"librosa>=0.10",
```

Full updated dependencies block:

```toml
dependencies = [
    # Core
    "numpy>=1.26",
    "pydantic>=2.6",

    # Media I/O (Phase 1)
    "ffmpeg-python>=0.2.0",
    "imageio-ffmpeg>=0.5.1",
    "opencv-python>=4.9",
    "Pillow>=10.0",

    # ASR (Phase 1)
    "faster-whisper>=1.0.3",

    # Phase 2: Visual features (CLIP)
    "torch>=2.2",
    "torchaudio>=2.2",
    "transformers>=4.40",

    # Phase 2: Audio spectral features   ← NEW
    "librosa>=0.10",

    # Phase 2: Text features (semantic embedding)
    "sentence-transformers>=2.7",
    "scikit-learn>=1.4",
]
```

---

## 2. Schema Changes (`schemas.py`)

Three schemas change. The `Literal` import must move above the Phase 2 class definitions since `AudioFeatureSegment` now uses it.

### 2.1 Move `Literal` import to the top of the file

```python
# At the top of schemas.py, with the other imports
from typing import Dict, List, Literal, Optional
```

Remove the duplicate `from typing import Literal` that currently appears at line 73.

### 2.2 `AudioFeatureSegment` — extend to a per-second grid row

**Old:**
```python
class AudioFeatureSegment(BaseModel):
    start: float = Field(..., ge=0.0)
    end: float = Field(..., ge=0.0)
    is_speech: bool = Field(True, description="True if human speech is detected in this window.")
```

**New:** Each segment represents exactly one second `[t, t+1)` and carries all spectral features for that second.

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
```

**Why per-second:** `align.py` builds a per-second grid; having one `AudioFeatureSegment` per second makes the alignment trivial and removes the interval-expansion loop.

### 2.3 `VisualFrameFeature` — add luminance, motion, and V2.1 DCT/chroma fields

```python
class VisualFrameFeature(BaseModel):
    frame_index: int = Field(..., ge=0)
    timestamp_sec: float = Field(..., ge=0.0)
    hist_diff_to_previous: float = Field(0.0, description="HSV histogram correlation distance from the previous frame.")
    is_hard_cut: bool = Field(False, description="True if hist_diff exceeds the cut threshold.")
    clip_labels: Dict[str, float] = Field(default_factory=dict, description="Zero-shot CLIP probabilities per scene prompt.")
    # --- New fields ---
    mean_luminance: float = Field(0.0, ge=0.0, description="Mean pixel brightness in [0, 255] (grayscale).")
    luminance_variance: float = Field(0.0, ge=0.0, description="Variance of pixel brightness.")
    is_black_frame: bool = Field(False, description="True if mean_luminance < 10 and luminance_variance < 50.")
    motion_intensity: float = Field(0.0, ge=0.0, description="Mean absolute pixel difference from the previous frame.")
    # V2.1 course-content fields
    chroma_diff: float = Field(0.0, ge=0.0, description="[V2.1] Mean chroma channel difference after 4:2:0 downsampling.")
    dct_hf_energy: float = Field(0.0, ge=0.0, description="[V2.1] Normalized high-frequency DCT energy of the frame.")
```

### 2.4 `TextFeatureSegment` — add `matched_keywords` field

```python
class TextFeatureSegment(BaseModel):
    id: int = Field(..., ge=0)
    start: float = Field(..., ge=0.0)
    end: float = Field(..., ge=0.0)
    text: str
    similarity_to_next: Optional[float] = Field(None, description="Cosine similarity to the next segment. None for the last.")
    is_potential_boundary: bool = Field(False, description="True if similarity_to_next is below threshold.")
    matched_keywords: List[str] = Field(
        default_factory=list,
        description="Keyword group matches found in this segment's text (e.g. 'sponsor:sponsored by')."
    )
```

---

## 3. Workspace Addition (`workspace.py`)

Add a property for the persisted MiniLM embedding array:

```python
@property
def text_embeddings_path(self) -> Path:
    return self.features_dir / "text_embeddings.npy"
```

---

## 4. §4.2.1 Visual Pipeline (`features/visual.py`)

### What changes

| Gap | Fix |
|---|---|
| CLIP prompt set has only 4 labels | Expand to 10 prompts covering all taxonomy classes |
| No black-frame / pure-color detection | Add per-frame luminance mean + variance |
| No motion intensity | Add absolute frame difference |
| [V2.1] Color subsampling | Add YCbCr 4:2:0 chroma channel diff |
| [V2.1] DCT analysis | Add 8×8 block DCT high-frequency energy |

### Full revised `backend/pipeline/features/visual.py`

```python
from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

from backend.pipeline.workspace import Workspace
from backend.pipeline.schemas import VisualFeatures, VisualFrameFeature
from backend.pipeline.logging_setup import get_logger

# ── Thresholds ────────────────────────────────────────────────────────────────
DEFAULT_CUT_THRESHOLD = 0.6
BLACK_FRAME_LUMINANCE_MAX = 10.0   # mean brightness below this → black frame
BLACK_FRAME_VARIANCE_MAX = 50.0    # variance below this → pure-color frame

# ── CLIP scene prompts (10 labels, covering all taxonomy classes) ─────────────
# Expanded from 4 → 10 to make all taxonomy labels reachable via the CLIP
# fallback classifier in classify.py.
SCENE_LABELS = [
    # Core content
    "a presentation slide",
    "a person talking to a camera",
    "a screen recording of software",
    # Transition / structural
    "a blank screen",
    "an end credits screen",
    "a title card",
    # Sponsor / promo
    "an advertisement slide",
    "a sponsor logo or product",
    # Other non-content
    "a video game interface",
    "an animated scene",
]


# ── Helper functions ──────────────────────────────────────────────────────────

def _load_clip_model(device: str) -> Tuple:
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)
    return processor, model


def _calculate_histogram(img_bgr: np.ndarray) -> np.ndarray:
    """Normalized 3D HSV histogram for scene-cut detection."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def _detect_black_frame(
    gray: np.ndarray,
    lum_max: float = BLACK_FRAME_LUMINANCE_MAX,
    var_max: float = BLACK_FRAME_VARIANCE_MAX,
) -> Tuple[float, float, bool]:
    """Return (mean_luminance, variance, is_black_frame)."""
    mean_lum = float(gray.mean())
    var_lum = float(gray.var())
    return mean_lum, var_lum, (mean_lum < lum_max and var_lum < var_max)


def _compute_motion_intensity(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """Mean absolute pixel difference between consecutive grayscale frames."""
    return float(cv2.absdiff(prev_gray, curr_gray).mean())


def _compute_chroma_diff(prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> float:
    """
    [V2.1] Chroma-channel difference with 4:2:0 downsampling.

    Mirrors the YCbCr 4:2:0 color subsampling used in MPEG/H.264 video
    (CSCI 576 §3.2). By operating on the downsampled chroma channels we
    reduce sensitivity to luma-only changes (lighting shifts) and focus on
    true color content changes — a better signal for scene-cut detection than
    luma alone.
    """
    def to_420_chroma(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ycbcr = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        _, cb, cr = cv2.split(ycbcr)
        h, w = cb.shape
        # 2× spatial downsampling in both axes (4:2:0)
        cb_ds = cv2.resize(cb, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
        cr_ds = cv2.resize(cr, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
        return cb_ds, cr_ds

    cb_p, cr_p = to_420_chroma(prev_bgr)
    cb_c, cr_c = to_420_chroma(curr_bgr)
    cb_diff = cv2.absdiff(cb_p.astype(np.float32), cb_c.astype(np.float32)).mean()
    cr_diff = cv2.absdiff(cr_p.astype(np.float32), cr_c.astype(np.float32)).mean()
    return float((cb_diff + cr_diff) / 2.0 / 255.0)


def _compute_dct_hf_energy(gray: np.ndarray, block_size: int = 8) -> float:
    """
    [V2.1] Normalized high-frequency DCT energy over the frame.

    Divides the frame into 8×8 blocks (identical to JPEG/MPEG block structure
    from CSCI 576 §4) and computes cv2.dct() on each. The bottom-right quadrant
    of each DCT block carries high-frequency coefficients; their summed energy
    indicates texture/detail richness. A title card or blank screen will have
    near-zero HF energy; a busy animated frame will have high energy.
    """
    h, w = gray.shape
    h_crop = (h // block_size) * block_size
    w_crop = (w // block_size) * block_size
    if h_crop == 0 or w_crop == 0:
        return 0.0
    gray_f = gray[:h_crop, :w_crop].astype(np.float32)

    total_hf = 0.0
    count = 0
    for i in range(0, h_crop, block_size):
        for j in range(0, w_crop, block_size):
            block = gray_f[i : i + block_size, j : j + block_size]
            dct_block = cv2.dct(block)
            # High-frequency region: lower-right 4×4 of the 8×8 block
            hf = dct_block[block_size // 2 :, block_size // 2 :]
            total_hf += float(np.sum(hf ** 2))
            count += 1

    # Normalise by block count and block area squared so the value is scale-invariant
    return total_hf / max(count, 1) / (block_size ** 4)


# ── Main extractor ────────────────────────────────────────────────────────────

def extract_visual_features(
    workspace: Workspace,
    device: str = "cpu",
    cut_threshold: float = DEFAULT_CUT_THRESHOLD,
) -> Path:
    """
    Reads 1 FPS JPEGs from frames_cache/, computes per-frame features, and
    writes features/visual_features.json.

    Per-frame outputs:
    - HSV histogram diff + is_hard_cut
    - CLIP zero-shot scene probabilities (10 labels)
    - Mean luminance, variance, is_black_frame
    - Motion intensity (absolute frame diff)
    - [V2.1] 4:2:0 chroma diff
    - [V2.1] 8×8 DCT high-frequency energy
    """
    log = get_logger(workspace.log_path)
    out_path = workspace.visual_features_path

    if out_path.exists():
        log.info("Visual features already exist. Skipping.")
        return out_path

    frame_files = sorted(workspace.frames_dir.glob("frame_*.jpg"))
    if not frame_files:
        log.warning("No frames found in frames_cache/. Skipping visual features.")
        return out_path

    log.info("Stage 2 (Visual): Loading CLIP model (%s)...", device)
    processor, clip_model = _load_clip_model(device)

    log.info("Processing %d frames...", len(frame_files))
    frame_features = []
    prev_hist: Optional[np.ndarray] = None
    prev_gray: Optional[np.ndarray] = None
    prev_bgr: Optional[np.ndarray] = None

    for frame_path in frame_files:
        frame_idx = int(frame_path.stem.split("_")[1])
        timestamp_sec = float(frame_idx)

        # ── Load image ──────────────────────────────────────────────────────
        img_bgr = cv2.imread(str(frame_path))
        if img_bgr is None:
            log.warning("Could not read frame %s — skipping.", frame_path.name)
            continue
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # ── HSV histogram diff ───────────────────────────────────────────────
        curr_hist = _calculate_histogram(img_bgr)
        hist_diff = 0.0
        is_cut = False
        if prev_hist is not None:
            similarity = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
            hist_diff = max(0.0, 1.0 - similarity)
            is_cut = hist_diff > cut_threshold

        # ── Black-frame detection ────────────────────────────────────────────
        mean_lum, var_lum, is_black = _detect_black_frame(gray)

        # ── Motion intensity ─────────────────────────────────────────────────
        motion = 0.0
        if prev_gray is not None:
            motion = _compute_motion_intensity(prev_gray, gray)

        # ── [V2.1] Chroma diff (4:2:0) ───────────────────────────────────────
        chroma_diff = 0.0
        if prev_bgr is not None:
            chroma_diff = _compute_chroma_diff(prev_bgr, img_bgr)

        # ── [V2.1] DCT high-frequency energy ────────────────────────────────
        dct_energy = _compute_dct_hf_energy(gray)

        # ── CLIP zero-shot scene classification ──────────────────────────────
        pil_image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        inputs = processor(
            text=SCENE_LABELS, images=pil_image, return_tensors="pt", padding=True
        ).to(device)
        with torch.no_grad():
            outputs = clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0].cpu().numpy()
        clip_results = {label: float(prob) for label, prob in zip(SCENE_LABELS, probs)}

        # ── Assemble feature record ──────────────────────────────────────────
        frame_features.append(
            VisualFrameFeature(
                frame_index=frame_idx,
                timestamp_sec=timestamp_sec,
                hist_diff_to_previous=round(hist_diff, 6),
                is_hard_cut=is_cut,
                clip_labels=clip_results,
                mean_luminance=round(mean_lum, 3),
                luminance_variance=round(var_lum, 3),
                is_black_frame=is_black,
                motion_intensity=round(motion, 4),
                chroma_diff=round(chroma_diff, 6),
                dct_hf_energy=round(dct_energy, 6),
            )
        )

        prev_hist = curr_hist
        prev_gray = gray
        prev_bgr = img_bgr.copy()

    # ── Release CLIP model before next stage ────────────────────────────────
    del clip_model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = VisualFeatures(frames=frame_features)
    out_path.write_text(result.model_dump_json(indent=2))
    log.info("Visual features saved to %s (%d frames).", out_path.name, len(frame_features))
    return out_path
```

---

## 5. §4.2.2 Audio Pipeline (`features/audio.py`)

### What changes

| Gap | Fix |
|---|---|
| Only Silero VAD; no spectral features | Add RMS, spectral centroid, bandwidth, ZCR via librosa |
| No music vs. speech classification | Add `_classify_audio_class()` heuristic |
| `audio_class` field missing | Added to `AudioFeatureSegment` (see §2.2) |
| [V2.1] Signal entropy | Add Shannon entropy of the power spectrum |
| Output is sparse intervals | Change to dense per-second grid (one segment per second) |

### Thresholds and reasoning

```
SILENCE_RMS  = 0.001   — below this amplitude, classify as silence regardless of other features
SPEECH_ZCR   = 0.10    — voiced speech has many zero crossings (0.10–0.25 typical)
                          music and tones sit lower (0.02–0.09 for a 440 Hz sine ≈ 0.055)
MUSIC_RMS    = 0.01    — music has substantial energy but ZCR below the speech threshold
```

These are starting values tuned for 16 kHz mono audio. Adjust after testing on real lecture/podcast material.

### Full revised `backend/pipeline/features/audio.py`

```python
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import List
import torch
import librosa

from backend.pipeline.workspace import Workspace
from backend.pipeline.schemas import AudioFeatures, AudioFeatureSegment
from backend.pipeline.logging_setup import get_logger

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
    # Reference: CSCI 576 information-theoretic signal analysis.
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
    segments: List[AudioFeatureSegment] = []

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
```

---

## 6. §4.2.3 Text Pipeline (`features/text.py`)

### What changes

| Gap | Fix |
|---|---|
| Keyword matching lives in `fusion/rules.py` | Move matching into this stage; emit `matched_keywords` per segment |
| MiniLM embeddings are discarded | Persist as `features/text_embeddings.npy` |

**Note on keyword lists:** The lists (`SPONSOR_KEYWORDS`, `INTRO_KEYWORDS`, etc.) are kept in `fusion/rules.py` as the authority. `text.py` imports them from there. If you prefer the text stage to own the lists, move them to a shared `pipeline/keywords.py` module and import from both places.

### Full revised `backend/pipeline/features/text.py`

```python
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from backend.pipeline.workspace import Workspace
from backend.pipeline.schemas import Transcript, TextFeatures, TextFeatureSegment
from backend.pipeline.logging_setup import get_logger

# Import keyword groups from the rules module (single source of truth).
# text.py detects matches; rules.py still uses the same lists for rule-hit
# intervals — they no longer re-scan the text, they read matched_keywords.
from backend.pipeline.fusion.rules import (
    SPONSOR_KEYWORDS,
    INTRO_KEYWORDS,
    OUTRO_KEYWORDS,
    SELF_PROMO_KEYWORDS,
    RECAP_KEYWORDS,
)

DEFAULT_SIMILARITY_THRESHOLD = 0.35

_KEYWORD_GROUPS = {
    "sponsor": SPONSOR_KEYWORDS,
    "intro": INTRO_KEYWORDS,
    "outro": OUTRO_KEYWORDS,
    "self_promo": SELF_PROMO_KEYWORDS,
    "recap": RECAP_KEYWORDS,
}


def _match_keywords(text: str) -> List[str]:
    """
    Return a list of '<group>:<keyword>' strings for every keyword group that
    matches anywhere in `text` (case-insensitive). At most one match per group.
    """
    text_lower = text.lower()
    matched = []
    for group, keywords in _KEYWORD_GROUPS.items():
        for kw in keywords:
            if kw in text_lower:
                matched.append(f"{group}:{kw}")
                break  # one hit per group is enough
    return matched


def _load_minilm_model(device: str = "cpu") -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2", device=device)


def extract_text_features(
    workspace: Workspace,
    device: str = "cpu",
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> Path:
    """
    Reads transcript.json, generates MiniLM embeddings, computes per-sentence
    cosine similarity to detect topic boundaries, matches keyword groups, and
    writes:
      - features/text_features.json   (TextFeatures schema)
      - features/text_embeddings.npy  (float32 array, shape [N, 384])
    """
    log = get_logger(workspace.log_path)
    out_path = workspace.text_features_path
    embeddings_path = workspace.text_embeddings_path

    if out_path.exists() and embeddings_path.exists():
        log.info("Text features already exist. Skipping.")
        return out_path

    log.info("Stage 2 (Text): Loading transcript...")
    with open(workspace.transcript_path, "r", encoding="utf-8") as f:
        transcript = Transcript.model_validate(json.load(f))

    if not transcript.segments:
        log.warning("Transcript is empty. Writing empty text features.")
        empty = TextFeatures(segments=[])
        out_path.write_text(empty.model_dump_json(indent=2))
        np.save(str(embeddings_path), np.empty((0, 384), dtype=np.float32))
        return out_path

    # ── Encode all sentences ──────────────────────────────────────────────────
    log.info("Stage 2 (Text): Encoding %d segments with MiniLM...", len(transcript.segments))
    model = _load_minilm_model(device)
    sentences = [seg.text for seg in transcript.segments]
    embeddings: np.ndarray = model.encode(sentences, convert_to_numpy=True)  # (N, 384)

    # Persist embeddings for supervised classifier and cross-episode matching
    np.save(str(embeddings_path), embeddings.astype(np.float32))
    log.info("Saved MiniLM embeddings to %s", embeddings_path.name)

    # ── Per-sentence features ─────────────────────────────────────────────────
    text_feature_segments = []
    for i, seg in enumerate(transcript.segments):
        sim_to_next = None
        is_boundary = False

        if i < len(transcript.segments) - 1:
            vec_curr = embeddings[i].reshape(1, -1)
            vec_next = embeddings[i + 1].reshape(1, -1)
            sim_to_next = float(cosine_similarity(vec_curr, vec_next)[0][0])
            is_boundary = sim_to_next < threshold

        matched_kws = _match_keywords(seg.text)

        text_feature_segments.append(
            TextFeatureSegment(
                id=seg.id,
                start=seg.start,
                end=seg.end,
                text=seg.text,
                similarity_to_next=sim_to_next,
                is_potential_boundary=is_boundary,
                matched_keywords=matched_kws,
            )
        )

    result = TextFeatures(segments=text_feature_segments)
    out_path.write_text(result.model_dump_json(indent=2))
    log.info("Text features saved to %s (%d segments).", out_path.name, len(text_feature_segments))
    return out_path
```

---

## 7. §4.2.4 CLI Orchestrator (`features/run.py`)

### What changes

| Gap | Fix |
|---|---|
| No single command runs all three sub-pipelines | Add `backend/pipeline/features/run.py` |
| Models accumulate in VRAM | Each extractor calls `del model; torch.cuda.empty_cache()` before the next |
| `ingest.py` only runs Phase 1 | Chain Phase 2 + 3 into `ingest.py` |

### `backend/pipeline/features/run.py`

```python
"""
Phase 2 CLI orchestrator.

Runs all three feature extractors in sequence, releasing GPU memory between
stages, then optionally chains into Phase 3 fusion.

Usage:
    python -m backend.pipeline.features.run path/to/video.mp4
    python -m backend.pipeline.features.run path/to/video.mp4 --device cuda
    python -m backend.pipeline.features.run path/to/video.mp4 --force
"""
from __future__ import annotations
import argparse
import gc
import sys
from pathlib import Path

import torch

from backend.pipeline.workspace import Workspace
from backend.pipeline.logging_setup import get_logger
from backend.pipeline.features.visual import extract_visual_features
from backend.pipeline.features.audio import extract_audio_features
from backend.pipeline.features.text import extract_text_features


DEFAULT_WORKSPACE_ROOT = Path(__file__).resolve().parents[3] / "workspace"


def run_phase2(
    workspace: Workspace,
    device: str = "cpu",
    force: bool = False,
) -> None:
    """Run all three Phase 2 sub-pipelines. Models are unloaded between stages."""
    log = get_logger(workspace.log_path)

    if force:
        for p in [
            workspace.visual_features_path,
            workspace.audio_features_path,
            workspace.text_features_path,
            workspace.text_embeddings_path,
        ]:
            if p.exists():
                p.unlink()
                log.info("Removed cached file: %s", p.name)

    log.info("=" * 60)
    log.info("Phase 2: Feature extraction  device=%s", device)

    log.info("Phase 2 [1/3]: Visual features (CLIP + OpenCV)")
    extract_visual_features(workspace, device=device)
    _release_gpu()

    log.info("Phase 2 [2/3]: Audio features (Silero VAD + librosa)")
    extract_audio_features(workspace, device=device)
    _release_gpu()

    log.info("Phase 2 [3/3]: Text features (MiniLM)")
    extract_text_features(workspace, device=device)
    _release_gpu()

    log.info("Phase 2 done.")


def _release_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m backend.pipeline.features.run",
        description="Phase 2: extract multimodal features from a Phase 1 workspace.",
    )
    p.add_argument("video", type=Path, help="Path to the original MP4 file.")
    p.add_argument(
        "--workspace", type=Path, default=DEFAULT_WORKSPACE_ROOT,
        help="Workspace root directory (default: <repo>/workspace).",
    )
    p.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"],
        help="Inference device (default: auto).",
    )
    p.add_argument("--force", action="store_true", help="Delete cached features and re-run.")
    args = p.parse_args(argv)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ws = Workspace.for_video(args.video, root=args.workspace)
    if not ws.audio_path.exists():
        print(
            f"ERROR: Phase 1 artifacts not found in {ws.root}.\n"
            "Run Phase 1 first: python -m backend.pipeline.ingest <video.mp4>",
            file=sys.stderr,
        )
        return 1

    run_phase2(ws, device=device, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Chaining Phase 2 into `ingest.py`

Add Phase 2 and Phase 3 calls at the end of `run_pipeline()` in `ingest.py`:

```python
# At the end of run_pipeline(), after Stage 4/4 transcription:

from backend.pipeline.features.run import run_phase2
from backend.pipeline.fusion.run import run_fusion

if force or not ws.visual_features_path.exists() \
         or not ws.audio_features_path.exists() \
         or not ws.text_features_path.exists():
    log.info("Phase 2: Extracting multimodal features")
    run_phase2(ws, device=resolved_device, force=force)
else:
    log.info("Phase 2: skipped (all feature files exist)")

if force or not ws.metadata_path.exists():
    log.info("Phase 3: Running fusion pipeline")
    run_fusion(ws)
else:
    log.info("Phase 3: skipped (metadata.json exists)")
```

---

## 8. Downstream: `align.py` Update

`align.py` must be updated to read the new audio grid columns from the per-second schema. The `is_speech` extraction logic is unchanged (the old loop still works for 1-second segments). Add the new grid keys after:

```python
# In build_per_second_grid(), after the existing is_speech block:

rms_energy      = np.zeros(T, dtype=np.float32)
spectral_centroid = np.zeros(T, dtype=np.float32)
spectral_bandwidth = np.zeros(T, dtype=np.float32)
zero_crossing_rate = np.zeros(T, dtype=np.float32)
spectral_entropy   = np.zeros(T, dtype=np.float32)
is_music           = np.zeros(T, dtype=np.int8)

for seg in audio.segments:
    t = int(np.floor(seg.start))
    if t >= T:
        continue
    is_speech[t] = int(seg.is_speech)
    rms_energy[t]       = seg.rms_energy
    spectral_centroid[t] = seg.spectral_centroid
    spectral_bandwidth[t] = seg.spectral_bandwidth
    zero_crossing_rate[t] = seg.zero_crossing_rate
    spectral_entropy[t]  = seg.spectral_entropy
    is_music[t]          = int(seg.audio_class == "music")

grid["rms_energy"]       = rms_energy
grid["spectral_centroid"] = spectral_centroid
grid["spectral_bandwidth"] = spectral_bandwidth
grid["zero_crossing_rate"] = zero_crossing_rate
grid["spectral_entropy"]   = spectral_entropy
grid["is_music"]           = is_music
```

This unblocks the §4.3 `rule_dead_air` refinement (`rms_energy < threshold AND is_speech == 0`) and the `boundaries.py` music→silence transition detection.

---

## 9. Test Suite

Tests are split across three files matching the existing layout. Each file replaces the current stub content entirely.

### 9.1 Shared fixtures (`tests/conftest.py` additions)

Add the new imports to the **top** of `conftest.py` (alongside the existing ones), then add the three fixtures after `synthetic_mp4`:

```python
# ── Add to the top of conftest.py (alongside existing imports) ────────────────
import wave
import cv2
import numpy as np
from backend.pipeline.workspace import Workspace
from backend.pipeline.audio import extract_audio as phase1_extract_audio
from backend.pipeline.frames import sample_frames
```

```python
# ── Add after the existing synthetic_mp4 fixture ─────────────────────────────

@pytest.fixture(scope="session")
def phase1_workspace(synthetic_mp4, tmp_path_factory) -> Workspace:
    """A workspace that has already run Phase 1 (audio WAV + 1 FPS frames)."""
    root = tmp_path_factory.mktemp("phase1_ws")
    ws = Workspace.for_video(synthetic_mp4, root=root / "workspace")
    ws.ensure()
    phase1_extract_audio(synthetic_mp4, ws)
    sample_frames(synthetic_mp4, ws, fps=1, longest_side=512, jpeg_quality=85)
    return ws


@pytest.fixture()
def silent_wav(tmp_path) -> Path:
    """A 3-second all-silent 16 kHz mono WAV."""
    path = tmp_path / "silent.wav"
    n_frames = 16000 * 3
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(bytes(2 * n_frames))  # 2 bytes per int16 sample, all zeros
    return path
```

---

### 9.2 `tests/test_visual.py` — full replacement

```python
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
```

---

### 9.3 `tests/test_audio_phase2.py` — full replacement

```python
"""
Tests for backend/pipeline/features/audio.py

Happy cases: correct feature values on known synthetic audio.
Sad cases: zero-duration audio, silent audio, missing file.
"""
from __future__ import annotations
import json
import wave
import struct
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
        """The synthetic MP4 has a 440 Hz sine tone; all seconds should have energy."""
        extract_audio_features(phase1_workspace, device="cpu")
        data = json.loads(phase1_workspace.audio_features_path.read_text())
        features = AudioFeatures.model_validate(data)
        rms_values = [s.rms_energy for s in features.segments]
        assert all(r > SILENCE_RMS_THRESHOLD for r in rms_values)

    @pytest.mark.slow
    def test_sine_wave_not_classified_as_silence(self, phase1_workspace):
        extract_audio_features(phase1_workspace, device="cpu")
        data = json.loads(phase1_workspace.audio_features_path.read_text())
        features = AudioFeatures.model_validate(data)
        for seg in features.segments:
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
        # Plant the silent WAV as the Phase 1 artifact
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
```

---

### 9.4 `tests/test_text.py` — full replacement

```python
"""
Tests for backend/pipeline/features/text.py

Happy cases: keyword detection, boundary detection, embedding persistence.
Sad cases: empty transcript, single segment, missing transcript file.
"""
from __future__ import annotations
import json
import numpy as np
import pytest
from pathlib import Path

from backend.pipeline.workspace import Workspace
from backend.pipeline.schemas import Transcript, TranscriptSegment, TextFeatures
from backend.pipeline.features.text import (
    _match_keywords,
    extract_text_features,
    DEFAULT_SIMILARITY_THRESHOLD,
)


def _make_transcript(segments: list[dict], tmp_path: Path) -> Workspace:
    """Write a fake transcript.json and return the workspace."""
    ws = Workspace.for_video(Path("test.mp4"), root=tmp_path / "workspace")
    ws.ensure()
    transcript = Transcript(
        language="en",
        duration_sec=float(max(s["end"] for s in segments)) if segments else 0.0,
        model="base",
        segments=[TranscriptSegment(**s) for s in segments],
        full_text=" ".join(s["text"] for s in segments),
    )
    ws.transcript_path.write_text(transcript.model_dump_json(indent=2))
    return ws


# ── Unit tests for _match_keywords ───────────────────────────────────────────

class TestMatchKeywords:
    def test_sponsor_keyword_detected(self):
        matches = _match_keywords("Today's video is brought to you by NordVPN.")
        assert any("sponsor" in m for m in matches)

    def test_intro_keyword_detected(self):
        matches = _match_keywords("Welcome back to the channel, in today's video...")
        assert any("intro" in m for m in matches)

    def test_outro_keyword_detected(self):
        matches = _match_keywords("Thanks for watching, don't forget to subscribe!")
        assert any("outro" in m for m in matches)

    def test_self_promo_keyword_detected(self):
        matches = _match_keywords("Check out my Patreon for more content.")
        assert any("self_promo" in m for m in matches)

    def test_recap_keyword_detected(self):
        matches = _match_keywords("Last time we discussed Fourier transforms.")
        assert any("recap" in m for m in matches)

    def test_no_match_returns_empty_list(self):
        matches = _match_keywords("Today we will study signal processing.")
        assert matches == []

    def test_case_insensitive(self):
        matches = _match_keywords("SPONSORED BY MegaCorp Inc.")
        assert any("sponsor" in m for m in matches)

    def test_at_most_one_match_per_group(self):
        # Even if multiple sponsor keywords appear, only one sponsor entry is emitted.
        text = "This video is sponsored. Brought to you by. Use code SAVE."
        matches = _match_keywords(text)
        sponsor_matches = [m for m in matches if m.startswith("sponsor:")]
        assert len(sponsor_matches) == 1

    def test_multiple_groups_can_match(self):
        text = "Welcome back! Thanks for watching. Last time we covered..."
        matches = _match_keywords(text)
        groups = {m.split(":")[0] for m in matches}
        assert len(groups) >= 2


# ── Integration tests (all call MiniLM — mark slow so they're skippable) ───────

class TestExtractTextFeatures:
    def _default_segments(self):
        return [
            {"id": 0, "start": 0.0, "end": 3.0, "text": "Welcome back to the channel."},
            {"id": 1, "start": 3.0, "end": 6.0, "text": "Today we discuss signal processing."},
            {"id": 2, "start": 6.0, "end": 9.0, "text": "Thanks for watching, like and subscribe!"},
        ]

    @pytest.mark.slow
    def test_output_json_created(self, tmp_path):
        ws = _make_transcript(self._default_segments(), tmp_path)
        out = extract_text_features(ws, device="cpu")
        assert out.exists()

    @pytest.mark.slow
    def test_output_validates_against_schema(self, tmp_path):
        ws = _make_transcript(self._default_segments(), tmp_path)
        extract_text_features(ws, device="cpu")
        data = json.loads(ws.text_features_path.read_text())
        features = TextFeatures.model_validate(data)
        assert len(features.segments) == 3

    @pytest.mark.slow
    def test_embeddings_npy_created(self, tmp_path):
        ws = _make_transcript(self._default_segments(), tmp_path)
        extract_text_features(ws, device="cpu")
        assert ws.text_embeddings_path.exists()

    @pytest.mark.slow
    def test_embeddings_shape_matches_segment_count(self, tmp_path):
        ws = _make_transcript(self._default_segments(), tmp_path)
        extract_text_features(ws, device="cpu")
        emb = np.load(str(ws.text_embeddings_path))
        assert emb.shape == (3, 384)

    @pytest.mark.slow
    def test_last_segment_similarity_to_next_is_none(self, tmp_path):
        ws = _make_transcript(self._default_segments(), tmp_path)
        extract_text_features(ws, device="cpu")
        data = json.loads(ws.text_features_path.read_text())
        features = TextFeatures.model_validate(data)
        assert features.segments[-1].similarity_to_next is None

    @pytest.mark.slow
    def test_intro_keyword_matched_in_first_segment(self, tmp_path):
        ws = _make_transcript(self._default_segments(), tmp_path)
        extract_text_features(ws, device="cpu")
        data = json.loads(ws.text_features_path.read_text())
        features = TextFeatures.model_validate(data)
        first = features.segments[0]
        assert any("intro" in kw for kw in first.matched_keywords)

    @pytest.mark.slow
    def test_outro_keyword_matched_in_last_segment(self, tmp_path):
        ws = _make_transcript(self._default_segments(), tmp_path)
        extract_text_features(ws, device="cpu")
        data = json.loads(ws.text_features_path.read_text())
        features = TextFeatures.model_validate(data)
        last = features.segments[-1]
        assert any("outro" in kw for kw in last.matched_keywords)

    @pytest.mark.slow
    def test_no_keywords_in_neutral_segment(self, tmp_path):
        neutral = [
            {"id": 0, "start": 0.0, "end": 3.0, "text": "The Fourier transform decomposes signals."},
            {"id": 1, "start": 3.0, "end": 6.0, "text": "Convolution in time equals multiplication in frequency."},
        ]
        ws = _make_transcript(neutral, tmp_path)
        extract_text_features(ws, device="cpu")
        data = json.loads(ws.text_features_path.read_text())
        features = TextFeatures.model_validate(data)
        for seg in features.segments:
            assert seg.matched_keywords == []

    @pytest.mark.slow
    def test_semantically_distant_pair_flagged_as_boundary(self, tmp_path):
        """A sponsor sentence after a technical sentence should trigger a boundary."""
        segments = [
            {"id": 0, "start": 0.0, "end": 3.0,
             "text": "The discrete cosine transform is used in JPEG compression."},
            {"id": 1, "start": 3.0, "end": 6.0,
             "text": "This video is brought to you by NordVPN, click the link below."},
        ]
        ws = _make_transcript(segments, tmp_path)
        extract_text_features(ws, device="cpu", threshold=DEFAULT_SIMILARITY_THRESHOLD)
        data = json.loads(ws.text_features_path.read_text())
        features = TextFeatures.model_validate(data)
        # First segment's similarity to the sponsor sentence should be low
        sim = features.segments[0].similarity_to_next
        assert sim is not None and sim < DEFAULT_SIMILARITY_THRESHOLD

    @pytest.mark.slow
    def test_semantically_similar_pair_not_flagged(self, tmp_path):
        segments = [
            {"id": 0, "start": 0.0, "end": 3.0,
             "text": "Sampling rate determines the highest frequency we can capture."},
            {"id": 1, "start": 3.0, "end": 6.0,
             "text": "The Nyquist theorem states we need twice the maximum frequency."},
        ]
        ws = _make_transcript(segments, tmp_path)
        extract_text_features(ws, device="cpu", threshold=DEFAULT_SIMILARITY_THRESHOLD)
        data = json.loads(ws.text_features_path.read_text())
        features = TextFeatures.model_validate(data)
        sim = features.segments[0].similarity_to_next
        assert sim is not None and sim > DEFAULT_SIMILARITY_THRESHOLD

    @pytest.mark.slow
    def test_cache_skips_on_second_call(self, tmp_path):
        ws = _make_transcript(self._default_segments(), tmp_path)
        extract_text_features(ws, device="cpu")
        mtime1 = ws.text_features_path.stat().st_mtime
        extract_text_features(ws, device="cpu")
        mtime2 = ws.text_features_path.stat().st_mtime
        assert mtime1 == mtime2

    # ── Sad cases ─────────────────────────────────────────────────────────────

    def test_empty_transcript_produces_empty_features(self, tmp_path):
        ws = _make_transcript([], tmp_path)
        out = extract_text_features(ws, device="cpu")
        data = json.loads(out.read_text())
        features = TextFeatures.model_validate(data)
        assert features.segments == []

    def test_empty_transcript_produces_empty_embeddings(self, tmp_path):
        ws = _make_transcript([], tmp_path)
        extract_text_features(ws, device="cpu")
        emb = np.load(str(ws.text_embeddings_path))
        assert emb.shape == (0, 384)

    def test_single_segment_similarity_is_none(self, tmp_path):
        ws = _make_transcript(
            [{"id": 0, "start": 0.0, "end": 5.0, "text": "Only one sentence here."}],
            tmp_path,
        )
        extract_text_features(ws, device="cpu")
        data = json.loads(ws.text_features_path.read_text())
        features = TextFeatures.model_validate(data)
        assert features.segments[0].similarity_to_next is None
        assert features.segments[0].is_potential_boundary is False

    def test_missing_transcript_raises_file_not_found(self, tmp_path):
        ws = Workspace.for_video(Path("no_transcript.mp4"), root=tmp_path / "ws")
        ws.ensure()
        # transcript_path does not exist
        with pytest.raises((FileNotFoundError, OSError)):
            extract_text_features(ws, device="cpu")
```

---

## Summary Table

| File | Change type | Status |
|---|---|---|
| `pyproject.toml` | Add `librosa>=0.10` | ✅ Done |
| Rename `features ` → `features` | Directory rename | ✅ Done |
| `backend/pipeline/schemas.py` | Extend 3 models, move `Literal` import | ✅ Done |
| `backend/pipeline/workspace.py` | Add `text_embeddings_path` property | ✅ Done |
| `backend/pipeline/features/visual.py` | Expand CLIP prompts + 4 new frame features | ✅ Done |
| `backend/pipeline/features/audio.py` | Full rewrite: librosa + per-second grid | ✅ Done |
| `backend/pipeline/features/text.py` | Add keyword matching + persist embeddings | ✅ Done |
| `backend/pipeline/features/run.py` | New file — Phase 2 CLI orchestrator | ✅ Done |
| `backend/pipeline/ingest.py` | Chain Phase 2 + Phase 3 into single command | ✅ Done |
| `backend/pipeline/fusion/align.py` | Read new audio grid columns | ✅ Done |
| `tests/conftest.py` | Add Phase 1 workspace + silent/black fixtures | ✅ Done |
| `tests/test_visual.py` | Replace stub with real tests | ✅ Done |
| `tests/test_audio_phase2.py` | Replace empty file with real tests | ✅ Done |
| `tests/test_text.py` | Replace stub with real tests | ✅ Done |

### Implementation notes

- **`align.py`**: The old interval-expansion loop (`is_speech[s:e] = 1`) was replaced entirely with a direct per-second assignment loop, since the new `AudioFeatureSegment` schema is already one row per second. All 7 new columns (`rms_energy`, `spectral_centroid`, `spectral_bandwidth`, `zero_crossing_rate`, `spectral_entropy`, `is_music`) are populated in the same loop.
- **`ingest.py`**: The Phase 3 `run_fusion` call is wrapped in a `try/except` so that a missing or incomplete `fusion/run.py` does not block Phase 1 + Phase 2 runs. Phase 2 has no guard — if Phase 2 fails, the pipeline raises immediately.
- **`conftest.py`**: The `black_frame_jpg` / `white_frame_jpg` fixtures from an earlier draft were dropped — they were unused by the final test suite.
