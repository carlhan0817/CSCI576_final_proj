# Phase 1: Data Ingestion & Demuxing — Walkthrough

**Date completed:** 2026-04-24
**Branch:** main
**Commits:** `14eb003` → `a86b515` (12 commits)
**Test suite:** 14 passed, 0 failed (fast), 1 slow test available (`@pytest.mark.slow`)

---

## What Phase 1 Does

Phase 1 is the **offline pre-processing stage** of the CSCI 576 multimodal video segmentation system. Given a single MP4 file, it produces four time-aligned cached artifacts that all downstream analysis stages (Phase 2 feature extraction, Phase 3 fusion) read from disk:

| Artifact | Path | Contents |
|---|---|---|
| `meta_raw.json` | `workspace/<stem>/meta_raw.json` | Video duration, fps, resolution, codecs |
| `audio_processed.wav` | `workspace/<stem>/audio_processed.wav` | Mono 16 kHz 16-bit PCM audio |
| `frames_cache/` | `workspace/<stem>/frames_cache/frame_XXXXXX.jpg` | 1 FPS key frames, longest-side 512 px |
| `transcript.json` | `workspace/<stem>/transcript.json` | Sentence-level ASR with timestamps |

Once generated, these artifacts are **cached**: re-running the pipeline skips any stage whose output already exists (per-stage granularity). Use `--force` to re-run everything.

---

## How to Run

```bash
# 1. Activate the virtual environment
# Windows:
.venv\Scripts\activate

# 2. Run the pipeline on a video
python -m backend.pipeline.ingest path/to/your_video.mp4

# 3. With options
python -m backend.pipeline.ingest path/to/video.mp4 \
  --model base \
  --device auto \
  --force
```

**CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--workspace DIR` | `./workspace` | Override output root |
| `--model NAME` | `base` | Whisper model: `tiny`, `base`, `small`, `medium` |
| `--device {auto,cpu,cuda}` | `auto` | Inference device (auto-detects GPU) |
| `--force` | off | Re-run all stages even if cached |

**Example output structure** (for `video.mp4`):
```
workspace/
└── video/
    ├── meta_raw.json
    ├── audio_processed.wav
    ├── transcript.json
    ├── frames_cache/
    │   ├── frame_000000.jpg
    │   ├── frame_000001.jpg
    │   └── ...
    └── ingest.log
```

---

## Architecture

```
MP4 file
    │
    ├──[probe.py]──────────→ meta_raw.json
    ├──[audio.py]──────────→ audio_processed.wav
    ├──[frames.py]─────────→ frames_cache/frame_XXXXXX.jpg
    └──[transcribe.py]─────→ transcript.json
            ↑
    (reads audio_processed.wav)

All stages orchestrated by ingest.py with per-stage cache logic.
```

The pipeline is intentionally **linear** (probe → audio → frames → transcribe in that order) so that the audio file exists before transcription begins. Each module is independent: it reads from disk, writes to disk, has no in-memory coupling.

---

## File-by-File Guide

### `backend/pipeline/schemas.py`

Defines the **pydantic v2 data contracts** for the two JSON artifacts. All models use `ConfigDict(extra="forbid")` — extra fields cause a validation error, preventing silent schema drift.

**`MetaRaw`** — written by `probe.py`, read by downstream stages:
```python
MetaRaw(
    filename="video.mp4",
    duration_sec=3600.0,
    fps=29.97,
    width=1920, height=1080,
    video_codec="h264",
    audio_codec="aac",
    has_audio=True,
    ingested_at="2026-04-24T10:00:00Z",
)
```

**`Transcript`** / **`TranscriptSegment`** — written by `transcribe.py`:
```python
Transcript(
    language="en",
    duration_sec=3600.0,
    model="base",
    segments=[
        TranscriptSegment(id=0, start=0.0, end=3.24, text="Hello and welcome.")
    ],
    full_text="Hello and welcome.",
)
```

Field validators enforce non-negative durations (`ge=0.0`), positive dimensions (`gt=0`), non-negative segment IDs.

---

### `backend/pipeline/workspace.py`

**Single source of truth for all file paths.** A frozen dataclass keyed by the MP4 filename stem:

```python
ws = Workspace.for_video(Path("video.mp4"), root=Path("workspace"))
# ws.root             → workspace/video/
# ws.meta_raw_path    → workspace/video/meta_raw.json
# ws.audio_path       → workspace/video/audio_processed.wav
# ws.transcript_path  → workspace/video/transcript.json
# ws.frames_dir       → workspace/video/frames_cache/
# ws.log_path         → workspace/video/ingest.log
ws.ensure()  # creates root + frames_cache dirs
```

Frozen (`@dataclass(frozen=True)`) so a workspace object can't be mutated after construction — prevents accidental path corruption in the orchestrator.

---

### `backend/pipeline/logging_setup.py`

Creates a **dual-sink logger**: one `FileHandler` writing to `workspace/<stem>/ingest.log` and one `StreamHandler` for console output. Both share the same ISO-8601 timestamp formatter. Calling `get_logger()` multiple times clears existing handlers first, preventing log duplication in long-running sessions.

```python
log = get_logger(ws.log_path)
log.info("Stage 1/4: probing video")
```

---

### `backend/pipeline/probe.py`

Extracts video metadata using `imageio_ffmpeg.read_frames()`. This approach was chosen over `ffmpeg-python`'s `ffmpeg.probe()` because the bundled `imageio_ffmpeg` binary on Windows does not include a separate `ffprobe` executable — using `read_frames()` to parse the ffmpeg stderr header achieves the same result without a system ffmpeg install.

**Key fields extracted:**
- `duration` — total video length in seconds
- `fps` — frame rate (used for frame count estimation in downstream stages)
- `source_size` — `(width, height)` tuple
- `codec` — video codec string (e.g., `"h264"`)
- `audio_codec` — audio codec string (e.g., `"aac"`) or `None` if no audio track

```python
meta = probe_video(mp4_path, workspace)
# → writes meta_raw.json, returns MetaRaw object
```

---

### `backend/pipeline/audio.py`

Extracts and converts the audio track to **mono 16 kHz 16-bit PCM WAV** — the exact format expected by:
- `faster-whisper` (Whisper internally resamples to this anyway; pre-converting avoids the overhead)
- `librosa` (Phase 2 audio feature extraction)

Uses `subprocess.run()` with the bundled `imageio_ffmpeg` binary directly (same reason as probe.py — no system ffmpeg required):

```bash
# Equivalent shell command:
ffmpeg -y -i input.mp4 -vn -ac 1 -ar 16000 -acodec pcm_s16le output.wav
```

```python
wav_path = extract_audio(mp4_path, workspace)
# → workspace/video/audio_processed.wav
```

---

### `backend/pipeline/frames.py`

Samples frames at **1 FPS** (one frame per second), resizes so the **longest side ≤ 512 px** (aspect ratio preserved), and saves as **JPEG quality 85**.

**Why 1 FPS:** A 1-hour video has ~108,000 frames at 30 fps. Processing all of them would consume gigabytes of RAM and hours of CLIP inference time. At 1 FPS, the same video produces 3,600 frames — manageable for downstream CLIP zero-shot classification while still capturing scene-level changes.

**Why 512 px longest side:** CLIP ViT-B/32 uses 224×224 internally but benefits from higher input resolution for visual detail. 512 px is a good balance — preserves enough visual context for scene type classification while keeping disk usage reasonable (~30–50 KB per JPEG).

**Filename convention:** `frame_XXXXXX.jpg` (6-digit zero-padded index, 0-based). A 1-hour video produces `frame_000000.jpg` through `frame_003599.jpg`. The index directly encodes the second of the video.

```python
paths = sample_frames(mp4_path, workspace, fps=1, longest_side=512, jpeg_quality=85)
# → [workspace/video/frames_cache/frame_000000.jpg, ...]
```

**FFmpeg scale filter used:**
```
scale='if(gt(iw,ih),512,-2)':'if(gt(iw,ih),-2,512)'
```
This conditionally caps either width or height depending on orientation, using `-2` to ensure even dimensions (required by some codecs).

---

### `backend/pipeline/transcribe.py`

Runs **faster-whisper** (CTranslate2 backend) on the extracted WAV file. faster-whisper is ~4× faster than the original `openai-whisper` on CPU, uses less memory, and supports `int8` quantization for further speedup on CPU inference.

**Model cache:** Downloaded to `workspace/models/` (project-local) on first run. The `HF_HOME` and `XDG_CACHE_HOME` environment variables are set at module import time to redirect all HuggingFace downloads here, ensuring the demo is fully offline after the first run.

**`_load_model()` is a module-level function** (not a method) so it can be monkeypatched in tests without loading the real 140 MB model:

```python
# In tests:
monkeypatch.setattr(transcribe_mod, "_load_model", lambda name, device: FakeModel())
```

**Output segments** are sentence-level (Whisper's natural output granularity), each with `start`/`end` timestamps in seconds. These are Phase 3's primary text signal for semantic topic boundary detection.

```python
transcript = transcribe(wav_path, workspace, model_name="base", device="cpu")
# → workspace/video/transcript.json
```

---

### `backend/pipeline/ingest.py`

The **CLI entrypoint and orchestrator**. Calls the four stages in dependency order with per-stage cache logic:

```
Stage 1: probe    → skip if meta_raw.json exists
Stage 2: audio    → skip if audio_processed.wav exists
Stage 3: frames   → skip if any frame_*.jpg files exist in frames_cache/
Stage 4: transcribe → skip if transcript.json exists
```

`--force` bypasses all cache checks and re-runs every stage.

`_resolve_device()` auto-detects CUDA if `torch` is installed and `torch.cuda.is_available()` returns True; falls back to CPU if torch is not installed (Phase 1 doesn't require torch).

**Run as module:**
```bash
python -m backend.pipeline.ingest video.mp4
```

**Import as library (used by Phase 3 tests and future orchestration):**
```python
from backend.pipeline.ingest import run_pipeline
ws = run_pipeline(Path("video.mp4"), force=True)
```

---

## JSON Output Schemas

### `meta_raw.json`
```json
{
  "filename": "lecture.mp4",
  "duration_sec": 3720.5,
  "fps": 29.97,
  "width": 1920,
  "height": 1080,
  "video_codec": "h264",
  "audio_codec": "aac",
  "has_audio": true,
  "ingested_at": "2026-04-24T14:30:00Z"
}
```

### `transcript.json`
```json
{
  "language": "en",
  "duration_sec": 3720.5,
  "model": "base",
  "segments": [
    {"id": 0, "start": 0.0,  "end": 3.24,  "text": "Hello and welcome to today's lecture."},
    {"id": 1, "start": 3.24, "end": 7.81,  "text": "Today we'll be covering multimodal segmentation."}
  ],
  "full_text": "Hello and welcome to today's lecture. Today we'll be covering multimodal segmentation."
}
```

---

## Test Suite

```bash
# Fast tests (mocked Whisper, uses synthetic MP4 fixture)
pytest -v -m "not slow"
# Expected: 14 passed

# Slow test (real faster-whisper base model, downloads ~140 MB first run)
pytest -v -m slow
# Expected: 1 passed (structural validity only — synthetic audio is a sine tone)
```

**Test breakdown:**

| File | Tests | What it verifies |
|---|---|---|
| `test_schemas.py` | 3 | Pydantic model round-trips, negative-start rejection |
| `test_workspace.py` | 3 | Path correctness, ensure() dir creation, fixture sanity |
| `test_probe.py` | 1 | meta_raw.json written with correct fields from synthetic MP4 |
| `test_audio.py` | 1 | WAV is mono, 16kHz, 16-bit, ~3 sec duration |
| `test_frames.py` | 2 | Frame count ±1, filenames, dimensions ≤512, aspect ratio |
| `test_transcribe.py` | 1+1s | Mocked transcript JSON structure; slow=real Whisper |
| `test_ingest_cli.py` | 3 | End-to-end artifact creation, cache skip, --force re-run |

**Synthetic MP4 fixture** (`tests/conftest.py`): generated at test-session scope using the bundled ffmpeg binary — a 3-second, 320×240 H.264/AAC video with a `testsrc` color pattern and 440 Hz sine audio. Deterministic, no binary files committed to git.

---

## Windows-Specific Notes

The standard `ffmpeg-python` workflow calls `ffprobe` for metadata, but the `imageio_ffmpeg` bundled binary on Windows ships only `ffmpeg-win-x86_64-v7.1.exe` — no separate `ffprobe`. The implementation adapts:

- `probe.py` uses `imageio_ffmpeg.read_frames()` to parse the ffmpeg stderr header for metadata
- `audio.py` and `frames.py` call the bundled exe via `subprocess.run()` instead of the `ffmpeg-python` wrapper

This means **no system ffmpeg installation is required** — everything runs from `.venv`.

---

## Known Minor Issues (Non-blocking)

1. **`transcribe.py` creates `workspace/models/` on import** (module-level `mkdir`). This is a harmless side effect but unexpected for a library module. Future improvement: move the `mkdir` inside `_load_model()`.

2. **`frames.py` `fps` parameter typed as `int`** — float would be more correct (e.g., for 0.5 FPS). Current spec uses integer 1, so no practical impact.

3. **Model cache path computed via `__file__` resolution** rather than through `Workspace` — means all videos share the same global model cache regardless of `--workspace` override. This is desirable behaviour (don't re-download the model per video) but makes it slightly inconsistent with the "everything via Workspace" convention.

---

## What Phase 2 Consumes

Phase 2 (Multimodal Feature Extraction) reads directly from the Phase 1 workspace:

| Phase 2 sub-pipeline | Reads from Phase 1 |
|---|---|
| Visual (CLIP, shot detection) | `frames_cache/frame_XXXXXX.jpg` |
| Audio (librosa, Silero VAD) | `audio_processed.wav` |
| Text (MiniLM embeddings, keywords) | `transcript.json` |
| Any stage needing duration/fps | `meta_raw.json` |

Phase 2 writes its outputs back to the same `workspace/<stem>/features/` directory. Phase 1 artifacts are never modified after creation.
