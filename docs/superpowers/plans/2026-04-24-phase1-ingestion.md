# Phase 1: Data Ingestion & Demuxing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the offline ingestion stage that turns a single MP4 into four time-aligned cached artifacts (`meta_raw.json`, `audio_processed.wav`, `transcript.json`, `frames_cache/*.jpg`) consumable by Phase 2.

**Architecture:** A CLI entrypoint (`python -m backend.pipeline.ingest <video.mp4>`) orchestrates four pure stages (probe, audio demux, transcribe, frame sample). Each stage reads from disk, writes to disk, and is independently cache-aware. Data contracts between stages are `pydantic` models serialized as JSON; filesystem layout is `workspace/<video_stem>/`.

**Tech Stack:** Python 3.12, `faster-whisper`, `ffmpeg-python`, `imageio-ffmpeg` (bundled binary), `opencv-python`, `pydantic` v2, `pytest`, `typer`-free stdlib `argparse` CLI, `setuptools` build backend (keeping existing `pyproject.toml` backend choice).

---

## Decisions Locked In

| # | Decision |
|---|---|
| Python | 3.12 (`requires-python = ">=3.12"`) |
| Build backend | `setuptools` (preserve existing `pyproject.toml`) |
| FFmpeg | Bundled via `imageio-ffmpeg` (no system install required) |
| ASR engine | `faster-whisper` (CTranslate2, CPU-friendly, 4× faster than `openai-whisper`) |
| Whisper model | `base` (English default) |
| Model cache | Project-local: `workspace/models/` |
| Frame sample rate | Fixed **1 FPS** |
| Frame format | JPEG quality 85, resized so longest side = 512 px, aspect preserved |
| Audio format | Mono, 16 kHz, PCM 16-bit WAV |
| CLI | `python -m backend.pipeline.ingest <path> [--force] [--device auto|cpu|cuda] [--model base]` |
| Cache behavior | Per-stage skip if output exists; `--force` re-runs all |
| Workspace key | MP4 filename stem (e.g., `foo.mp4` → `workspace/foo/`) |
| Test fixture | Synthetic MP4 generated at test-session-scope via ffmpeg |
| Whisper in tests | Mocked in fast tests; real Whisper behind `@pytest.mark.slow` |
| Test framework | `pytest` |
| Logging | `logging` module; per-video log at `workspace/<stem>/ingest.log` |

---

## File Structure

```
Final_proj/
├── pyproject.toml                        [MODIFY] upgrade deps, pin Python 3.12
├── backend/
│   ├── __init__.py                       [CREATE] empty
│   └── pipeline/
│       ├── __init__.py                   [CREATE] empty
│       ├── schemas.py                    [CREATE] pydantic models for JSON artifacts
│       ├── workspace.py                  [CREATE] workspace path + cache-skip logic
│       ├── logging_setup.py              [CREATE] per-video file logger
│       ├── probe.py                      [CREATE] ffmpeg probe → meta_raw.json
│       ├── audio.py                      [CREATE] ffmpeg audio demux → WAV
│       ├── transcribe.py                 [CREATE] faster-whisper → transcript.json
│       ├── frames.py                     [CREATE] ffmpeg 1 FPS + resize → JPEGs
│       └── ingest.py                     [CREATE] CLI orchestrator
└── tests/
    ├── __init__.py                       [CREATE] empty
    ├── conftest.py                       [CREATE] synthetic MP4 fixture
    ├── test_schemas.py                   [CREATE]
    ├── test_workspace.py                 [CREATE]
    ├── test_probe.py                     [CREATE]
    ├── test_audio.py                     [CREATE]
    ├── test_frames.py                    [CREATE]
    ├── test_transcribe.py                [CREATE]
    └── test_ingest_cli.py                [CREATE] end-to-end CLI test (mocked Whisper)
```

**Responsibility per file:**

- `schemas.py` — all pydantic models; no I/O, no ffmpeg. Pure types.
- `workspace.py` — resolves workspace paths for a given video; single source of truth for where artifacts live.
- `logging_setup.py` — attaches a `FileHandler` for `workspace/<stem>/ingest.log` and a console handler.
- `probe.py` — one public function `probe_video(mp4_path, workspace) -> MetaRaw` that writes `meta_raw.json`.
- `audio.py` — one public function `extract_audio(mp4_path, workspace) -> Path` that writes `audio_processed.wav`.
- `transcribe.py` — one public function `transcribe(wav_path, workspace, model_name, device) -> Transcript` that writes `transcript.json`.
- `frames.py` — one public function `sample_frames(mp4_path, workspace, fps=1, longest_side=512, jpeg_quality=85) -> list[Path]` that writes `frames_cache/frame_XXXXXX.jpg`.
- `ingest.py` — `argparse` CLI + `run_pipeline()` that calls the four stages in order with cache-skip.

---

## Data Contracts

### `meta_raw.json` (model: `MetaRaw`)
```json
{
  "filename": "example.mp4",
  "duration_sec": 1234.56,
  "fps": 29.97,
  "width": 1920,
  "height": 1080,
  "video_codec": "h264",
  "audio_codec": "aac",
  "has_audio": true,
  "ingested_at": "2026-04-24T10:00:00Z"
}
```

### `transcript.json` (model: `Transcript`)
```json
{
  "language": "en",
  "duration_sec": 1234.56,
  "model": "base",
  "segments": [
    {"id": 0, "start": 0.0, "end": 3.24, "text": "Hello and welcome."}
  ],
  "full_text": "Hello and welcome."
}
```

---

## Task 0: Update `pyproject.toml` and rebuild `.venv`

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Replace `pyproject.toml` with Phase 1 dependency set**

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "multimedia-segmentation"
version = "0.1.0"
description = "Multimodal Long-Form Video Segmentation System"
requires-python = ">=3.12"
dependencies = [
    # Core
    "numpy>=1.26",
    "pydantic>=2.6",

    # Media I/O (Phase 1)
    "ffmpeg-python>=0.2.0",
    "imageio-ffmpeg>=0.5.1",
    "opencv-python>=4.9",

    # ASR (Phase 1)
    "faster-whisper>=1.0.3",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-mock>=3.12",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["backend*"]

[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
]
testpaths = ["tests"]
```

- [ ] **Step 2: Reinstall dependencies into `.venv`**

Run (inside activated `.venv`):
```
pip install -e ".[dev]"
```
Expected: installs clean; no resolver conflicts. If `numpy` / `faster-whisper` fail to build, stop and report.

- [ ] **Step 3: Verify core imports work**

Run:
```
python -c "import faster_whisper, ffmpeg, cv2, pydantic, imageio_ffmpeg; print('ok')"
```
Expected: prints `ok`.

- [ ] **Step 4: Commit**

```
git add pyproject.toml
git commit -m "chore: pin Phase 1 dependencies for Python 3.12"
```

---

## Task 1: Create `backend` package skeleton

**Files:**
- Create: `backend/__init__.py`
- Create: `backend/pipeline/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create empty package files**

`backend/__init__.py`:
```python
```

`backend/pipeline/__init__.py`:
```python
```

`tests/__init__.py`:
```python
```

- [ ] **Step 2: Verify package importable**

Run:
```
python -c "import backend.pipeline; print('ok')"
```
Expected: prints `ok`.

- [ ] **Step 3: Commit**

```
git add backend/__init__.py backend/pipeline/__init__.py tests/__init__.py
git commit -m "chore: scaffold backend.pipeline package"
```

---

## Task 2: Pydantic schemas for JSON artifacts

**Files:**
- Create: `backend/pipeline/schemas.py`
- Create: `tests/test_schemas.py`

- [ ] **Step 1: Write failing test for `MetaRaw` round-trip**

`tests/test_schemas.py`:
```python
import json
from backend.pipeline.schemas import MetaRaw, Transcript, TranscriptSegment


def test_meta_raw_roundtrip():
    payload = {
        "filename": "foo.mp4",
        "duration_sec": 12.5,
        "fps": 30.0,
        "width": 640,
        "height": 480,
        "video_codec": "h264",
        "audio_codec": "aac",
        "has_audio": True,
        "ingested_at": "2026-04-24T10:00:00Z",
    }
    m = MetaRaw.model_validate(payload)
    assert m.duration_sec == 12.5
    assert m.has_audio is True
    assert json.loads(m.model_dump_json()) == payload


def test_transcript_roundtrip():
    payload = {
        "language": "en",
        "duration_sec": 3.5,
        "model": "base",
        "segments": [
            {"id": 0, "start": 0.0, "end": 1.2, "text": "hi"},
            {"id": 1, "start": 1.2, "end": 3.5, "text": "there"},
        ],
        "full_text": "hi there",
    }
    t = Transcript.model_validate(payload)
    assert len(t.segments) == 2
    assert t.segments[0].text == "hi"
    assert json.loads(t.model_dump_json()) == payload


def test_transcript_segment_rejects_negative_start():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        TranscriptSegment(id=0, start=-0.1, end=1.0, text="x")
```

- [ ] **Step 2: Run test — expect FAIL (module missing)**

Run:
```
pytest tests/test_schemas.py -v
```
Expected: `ModuleNotFoundError: backend.pipeline.schemas`.

- [ ] **Step 3: Implement schemas**

`backend/pipeline/schemas.py`:
```python
from __future__ import annotations
from pydantic import BaseModel, Field, ConfigDict


class MetaRaw(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filename: str
    duration_sec: float = Field(ge=0.0)
    fps: float = Field(gt=0.0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    video_codec: str
    audio_codec: str
    has_audio: bool
    ingested_at: str  # ISO-8601 UTC string


class TranscriptSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: int = Field(ge=0)
    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)
    text: str


class Transcript(BaseModel):
    model_config = ConfigDict(extra="forbid")
    language: str
    duration_sec: float = Field(ge=0.0)
    model: str
    segments: list[TranscriptSegment]
    full_text: str
```

- [ ] **Step 4: Run test — expect PASS**

Run:
```
pytest tests/test_schemas.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```
git add backend/pipeline/schemas.py tests/test_schemas.py
git commit -m "feat(pipeline): pydantic schemas for MetaRaw and Transcript"
```

---

## Task 3: Workspace path resolver

**Files:**
- Create: `backend/pipeline/workspace.py`
- Create: `tests/test_workspace.py`

- [ ] **Step 1: Write failing tests**

`tests/test_workspace.py`:
```python
from pathlib import Path
from backend.pipeline.workspace import Workspace


def test_workspace_paths_from_mp4(tmp_path):
    video = tmp_path / "my_video.mp4"
    video.write_bytes(b"fake")
    ws = Workspace.for_video(video, root=tmp_path / "workspace")

    assert ws.root == tmp_path / "workspace" / "my_video"
    assert ws.meta_raw_path == ws.root / "meta_raw.json"
    assert ws.audio_path == ws.root / "audio_processed.wav"
    assert ws.transcript_path == ws.root / "transcript.json"
    assert ws.frames_dir == ws.root / "frames_cache"
    assert ws.log_path == ws.root / "ingest.log"


def test_workspace_ensure_creates_dirs(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"fake")
    ws = Workspace.for_video(video, root=tmp_path / "workspace")
    ws.ensure()
    assert ws.root.is_dir()
    assert ws.frames_dir.is_dir()
```

- [ ] **Step 2: Run test — expect FAIL**

Run:
```
pytest tests/test_workspace.py -v
```
Expected: `ModuleNotFoundError: backend.pipeline.workspace`.

- [ ] **Step 3: Implement Workspace**

`backend/pipeline/workspace.py`:
```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Workspace:
    source_video: Path
    root: Path

    @classmethod
    def for_video(cls, video: Path, root: Path) -> "Workspace":
        stem = video.stem
        return cls(source_video=video, root=root / stem)

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    @property
    def meta_raw_path(self) -> Path:
        return self.root / "meta_raw.json"

    @property
    def audio_path(self) -> Path:
        return self.root / "audio_processed.wav"

    @property
    def transcript_path(self) -> Path:
        return self.root / "transcript.json"

    @property
    def frames_dir(self) -> Path:
        return self.root / "frames_cache"

    @property
    def log_path(self) -> Path:
        return self.root / "ingest.log"
```

- [ ] **Step 4: Run test — expect PASS**

Run:
```
pytest tests/test_workspace.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```
git add backend/pipeline/workspace.py tests/test_workspace.py
git commit -m "feat(pipeline): Workspace path resolver"
```

---

## Task 4: Test fixture — synthetic MP4 generator

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 1: Write the conftest fixture**

`tests/conftest.py`:
```python
import subprocess
from pathlib import Path
import pytest
import imageio_ffmpeg


FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


@pytest.fixture(scope="session")
def synthetic_mp4(tmp_path_factory) -> Path:
    """Deterministic 3-second MP4: 320x240 color-pattern video + 440Hz sine audio."""
    out_dir = tmp_path_factory.mktemp("fixtures")
    out = out_dir / "synth.mp4"
    cmd = [
        FFMPEG, "-y",
        "-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=10",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3:sample_rate=44100",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    assert out.exists() and out.stat().st_size > 0
    return out
```

- [ ] **Step 2: Write a sanity test that uses the fixture**

Append to `tests/test_workspace.py`:
```python
def test_synthetic_mp4_fixture_exists(synthetic_mp4):
    assert synthetic_mp4.exists()
    assert synthetic_mp4.suffix == ".mp4"
    assert synthetic_mp4.stat().st_size > 1000
```

- [ ] **Step 3: Run test — expect PASS**

Run:
```
pytest tests/test_workspace.py::test_synthetic_mp4_fixture_exists -v
```
Expected: 1 passed. (First run compiles the MP4; subsequent session uses the cache.)

- [ ] **Step 4: Commit**

```
git add tests/conftest.py tests/test_workspace.py
git commit -m "test: synthetic MP4 fixture via ffmpeg lavfi"
```

---

## Task 5: Logging setup

**Files:**
- Create: `backend/pipeline/logging_setup.py`

- [ ] **Step 1: Implement logger (no test — pure wiring)**

`backend/pipeline/logging_setup.py`:
```python
from __future__ import annotations
import logging
from pathlib import Path


def get_logger(log_path: Path, name: str = "pipeline") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.propagate = False
    return logger
```

- [ ] **Step 2: Smoke test manually**

Run:
```
python -c "from pathlib import Path; from backend.pipeline.logging_setup import get_logger; l = get_logger(Path('/tmp/x.log')); l.info('hi')"
```
Expected: prints a log line. (On Windows use `./x.log` instead of `/tmp/x.log`.)

- [ ] **Step 3: Commit**

```
git add backend/pipeline/logging_setup.py
git commit -m "feat(pipeline): per-video file logger"
```

---

## Task 6: Probe stage (`meta_raw.json`)

**Files:**
- Create: `backend/pipeline/probe.py`
- Create: `tests/test_probe.py`

- [ ] **Step 1: Write failing test**

`tests/test_probe.py`:
```python
import json
from backend.pipeline.workspace import Workspace
from backend.pipeline.probe import probe_video


def test_probe_writes_valid_meta_raw(tmp_path, synthetic_mp4):
    ws = Workspace.for_video(synthetic_mp4, root=tmp_path / "workspace")
    ws.ensure()

    meta = probe_video(synthetic_mp4, ws)

    assert ws.meta_raw_path.exists()
    data = json.loads(ws.meta_raw_path.read_text())
    assert data["filename"] == "synth.mp4"
    assert 2.5 <= data["duration_sec"] <= 3.5
    assert data["width"] == 320
    assert data["height"] == 240
    assert data["has_audio"] is True
    assert data["video_codec"] == "h264"
    assert data["ingested_at"].endswith("Z")

    # Returned object matches file
    assert meta.width == 320
```

- [ ] **Step 2: Run — expect FAIL**

Run:
```
pytest tests/test_probe.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement probe**

`backend/pipeline/probe.py`:
```python
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import ffmpeg

from backend.pipeline.schemas import MetaRaw
from backend.pipeline.workspace import Workspace


def _parse_fps(r_frame_rate: str) -> float:
    num, _, den = r_frame_rate.partition("/")
    num_f = float(num)
    den_f = float(den) if den else 1.0
    return num_f / den_f if den_f else 0.0


def probe_video(mp4_path: Path, workspace: Workspace) -> MetaRaw:
    info = ffmpeg.probe(str(mp4_path))
    streams = info["streams"]
    video = next(s for s in streams if s["codec_type"] == "video")
    audio = next((s for s in streams if s["codec_type"] == "audio"), None)

    meta = MetaRaw(
        filename=mp4_path.name,
        duration_sec=float(info["format"]["duration"]),
        fps=_parse_fps(video["r_frame_rate"]),
        width=int(video["width"]),
        height=int(video["height"]),
        video_codec=video["codec_name"],
        audio_codec=audio["codec_name"] if audio else "none",
        has_audio=audio is not None,
        ingested_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    workspace.meta_raw_path.write_text(meta.model_dump_json(indent=2))
    return meta
```

- [ ] **Step 4: Ensure ffmpeg-python uses bundled ffmpeg binary**

`ffmpeg-python` calls `ffmpeg` from PATH. Prepend `imageio_ffmpeg`'s binary directory to `os.environ["PATH"]` so no system install is needed. Update `backend/pipeline/probe.py` — insert at top of the module (below imports):

```python
import os
import imageio_ffmpeg

_FFMPEG_DIR = str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)
if _FFMPEG_DIR not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")
```

- [ ] **Step 5: Run — expect PASS**

Run:
```
pytest tests/test_probe.py -v
```
Expected: 1 passed.

- [ ] **Step 6: Commit**

```
git add backend/pipeline/probe.py tests/test_probe.py
git commit -m "feat(pipeline): probe stage writes meta_raw.json"
```

---

## Task 7: Audio demux stage (`audio_processed.wav`)

**Files:**
- Create: `backend/pipeline/audio.py`
- Create: `tests/test_audio.py`

- [ ] **Step 1: Write failing test**

`tests/test_audio.py`:
```python
import wave
from backend.pipeline.workspace import Workspace
from backend.pipeline.audio import extract_audio


def test_extract_audio_produces_mono_16k_wav(tmp_path, synthetic_mp4):
    ws = Workspace.for_video(synthetic_mp4, root=tmp_path / "workspace")
    ws.ensure()

    out = extract_audio(synthetic_mp4, ws)

    assert out == ws.audio_path
    assert out.exists()
    with wave.open(str(out), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == 16000
        assert w.getsampwidth() == 2  # 16-bit
        # Duration ~3 seconds
        duration = w.getnframes() / w.getframerate()
        assert 2.5 <= duration <= 3.5
```

- [ ] **Step 2: Run — expect FAIL**

Run:
```
pytest tests/test_audio.py -v
```

- [ ] **Step 3: Implement audio extraction**

`backend/pipeline/audio.py`:
```python
from __future__ import annotations
import os
from pathlib import Path
import ffmpeg
import imageio_ffmpeg

from backend.pipeline.workspace import Workspace

_FFMPEG_DIR = str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)
if _FFMPEG_DIR not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")


def extract_audio(mp4_path: Path, workspace: Workspace) -> Path:
    out = workspace.audio_path
    (
        ffmpeg
        .input(str(mp4_path))
        .output(
            str(out),
            ac=1,             # mono
            ar=16000,         # 16 kHz
            acodec="pcm_s16le",
            vn=None,          # drop video
        )
        .overwrite_output()
        .run(quiet=True)
    )
    return out
```

- [ ] **Step 4: Run — expect PASS**

Run:
```
pytest tests/test_audio.py -v
```
Expected: 1 passed.

- [ ] **Step 5: Commit**

```
git add backend/pipeline/audio.py tests/test_audio.py
git commit -m "feat(pipeline): audio demux to mono 16kHz WAV"
```

---

## Task 8: Frame sampling stage (`frames_cache/`)

**Files:**
- Create: `backend/pipeline/frames.py`
- Create: `tests/test_frames.py`

- [ ] **Step 1: Write failing test**

`tests/test_frames.py`:
```python
import cv2
from backend.pipeline.workspace import Workspace
from backend.pipeline.frames import sample_frames


def test_sample_frames_writes_one_jpeg_per_second(tmp_path, synthetic_mp4):
    ws = Workspace.for_video(synthetic_mp4, root=tmp_path / "workspace")
    ws.ensure()

    paths = sample_frames(synthetic_mp4, ws, fps=1, longest_side=512, jpeg_quality=85)

    # 3-second video at 1 FPS → 3 frames (allow ±1 for encoder edge effects)
    assert 2 <= len(paths) <= 4
    for p in paths:
        assert p.exists()
        assert p.suffix == ".jpg"
        assert p.name.startswith("frame_")
        img = cv2.imread(str(p))
        assert img is not None
        h, w = img.shape[:2]
        # Longest side capped at 512
        assert max(h, w) <= 512
        # Aspect preserved (source is 320x240 → 4:3)
        assert abs((w / h) - (320 / 240)) < 0.05


def test_sample_frames_filename_encodes_index(tmp_path, synthetic_mp4):
    ws = Workspace.for_video(synthetic_mp4, root=tmp_path / "workspace")
    ws.ensure()
    paths = sample_frames(synthetic_mp4, ws, fps=1, longest_side=512, jpeg_quality=85)
    names = sorted(p.name for p in paths)
    assert names[0] == "frame_000000.jpg"
    assert names[1] == "frame_000001.jpg"
```

- [ ] **Step 2: Run — expect FAIL**

Run:
```
pytest tests/test_frames.py -v
```

- [ ] **Step 3: Implement frame sampler**

`backend/pipeline/frames.py`:
```python
from __future__ import annotations
import os
from pathlib import Path
import ffmpeg
import imageio_ffmpeg

from backend.pipeline.workspace import Workspace

_FFMPEG_DIR = str(Path(imageio_ffmpeg.get_ffmpeg_exe()).parent)
if _FFMPEG_DIR not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")


def sample_frames(
    mp4_path: Path,
    workspace: Workspace,
    fps: int = 1,
    longest_side: int = 512,
    jpeg_quality: int = 85,
) -> list[Path]:
    out_dir = workspace.frames_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "frame_%06d.jpg")

    # Scale: longest side = longest_side, preserve aspect. ffmpeg "scale=w:h:force_original_aspect_ratio"
    # Use conditional scale so portrait/landscape both capped.
    scale_expr = (
        f"scale='if(gt(iw,ih),{longest_side},-2)':'if(gt(iw,ih),-2,{longest_side})'"
    )
    # ffmpeg JPEG quality: -q:v 2..31 (lower = better). Map quality 85 → q:v 3.
    qv = max(2, min(31, int(round((100 - jpeg_quality) / 3))))

    (
        ffmpeg
        .input(str(mp4_path))
        .output(
            pattern,
            vf=f"fps={fps},{scale_expr}",
            **{"q:v": qv},
            start_number=0,
        )
        .overwrite_output()
        .run(quiet=True)
    )
    return sorted(out_dir.glob("frame_*.jpg"))
```

- [ ] **Step 4: Run — expect PASS**

Run:
```
pytest tests/test_frames.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```
git add backend/pipeline/frames.py tests/test_frames.py
git commit -m "feat(pipeline): 1 FPS frame sampling with aspect-preserving resize"
```

---

## Task 9: Transcription stage (`transcript.json`)

**Files:**
- Create: `backend/pipeline/transcribe.py`
- Create: `tests/test_transcribe.py`

- [ ] **Step 1: Write failing tests (fast + slow)**

`tests/test_transcribe.py`:
```python
import json
from types import SimpleNamespace
import pytest

from backend.pipeline.workspace import Workspace
from backend.pipeline.schemas import Transcript
from backend.pipeline import transcribe as transcribe_mod


def _fake_segment(id_, start, end, text):
    return SimpleNamespace(id=id_, start=start, end=end, text=text)


def test_transcribe_writes_valid_transcript_json_mocked(tmp_path, synthetic_mp4, monkeypatch):
    ws = Workspace.for_video(synthetic_mp4, root=tmp_path / "workspace")
    ws.ensure()
    # Fake the WAV by creating an empty file at expected path
    ws.audio_path.write_bytes(b"RIFF0000WAVEfmt ")

    class FakeModel:
        def transcribe(self, wav_path, **kwargs):
            segs = [
                _fake_segment(0, 0.0, 1.2, "hello"),
                _fake_segment(1, 1.2, 2.5, "world"),
            ]
            info = SimpleNamespace(language="en", duration=2.5)
            return iter(segs), info

    monkeypatch.setattr(transcribe_mod, "_load_model", lambda name, device: FakeModel())

    t = transcribe_mod.transcribe(ws.audio_path, ws, model_name="base", device="cpu")

    assert ws.transcript_path.exists()
    data = json.loads(ws.transcript_path.read_text())
    assert data["language"] == "en"
    assert data["model"] == "base"
    assert len(data["segments"]) == 2
    assert data["segments"][0]["text"] == "hello"
    assert data["full_text"] == "hello world"
    # Returned matches file
    assert isinstance(t, Transcript)


@pytest.mark.slow
def test_transcribe_real_whisper_on_synthetic(tmp_path, synthetic_mp4):
    """Real faster-whisper run. Synthetic audio is a sine tone so the model may
    produce empty or nonsense text — we only assert structural validity."""
    from backend.pipeline.audio import extract_audio

    ws = Workspace.for_video(synthetic_mp4, root=tmp_path / "workspace")
    ws.ensure()
    extract_audio(synthetic_mp4, ws)

    t = transcribe_mod.transcribe(ws.audio_path, ws, model_name="base", device="cpu")

    assert ws.transcript_path.exists()
    assert t.model == "base"
    assert t.language  # non-empty
```

- [ ] **Step 2: Run — expect FAIL (module missing)**

Run:
```
pytest tests/test_transcribe.py -v -m "not slow"
```

- [ ] **Step 3: Implement transcription**

`backend/pipeline/transcribe.py`:
```python
from __future__ import annotations
import os
from pathlib import Path

from backend.pipeline.schemas import Transcript, TranscriptSegment
from backend.pipeline.workspace import Workspace

# Project-local model cache
_MODEL_CACHE = Path(__file__).resolve().parents[2] / "workspace" / "models"
_MODEL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_MODEL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(_MODEL_CACHE))


def _load_model(model_name: str, device: str):
    from faster_whisper import WhisperModel

    compute_type = "int8" if device == "cpu" else "float16"
    return WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        download_root=str(_MODEL_CACHE),
    )


def transcribe(
    wav_path: Path,
    workspace: Workspace,
    model_name: str = "base",
    device: str = "cpu",
) -> Transcript:
    model = _load_model(model_name, device)
    segments_iter, info = model.transcribe(str(wav_path), beam_size=1)

    segments: list[TranscriptSegment] = []
    parts: list[str] = []
    for i, seg in enumerate(segments_iter):
        text = seg.text.strip()
        segments.append(TranscriptSegment(
            id=i,
            start=float(seg.start),
            end=float(seg.end),
            text=text,
        ))
        parts.append(text)

    transcript = Transcript(
        language=info.language,
        duration_sec=float(info.duration),
        model=model_name,
        segments=segments,
        full_text=" ".join(parts),
    )
    workspace.transcript_path.write_text(transcript.model_dump_json(indent=2))
    return transcript
```

- [ ] **Step 4: Run fast tests — expect PASS**

Run:
```
pytest tests/test_transcribe.py -v -m "not slow"
```
Expected: 1 passed, 1 deselected.

- [ ] **Step 5: Run slow test once manually to verify real Whisper works**

Run:
```
pytest tests/test_transcribe.py -v -m slow
```
Expected: 1 passed (will download `base` model on first run to `workspace/models/`). If this fails due to `faster-whisper` / CT2 install issues, STOP and report.

- [ ] **Step 6: Commit**

```
git add backend/pipeline/transcribe.py tests/test_transcribe.py
git commit -m "feat(pipeline): faster-whisper transcription stage"
```

---

## Task 10: CLI orchestrator with per-stage cache

**Files:**
- Create: `backend/pipeline/ingest.py`
- Create: `tests/test_ingest_cli.py`

- [ ] **Step 1: Write failing end-to-end test (Whisper mocked)**

`tests/test_ingest_cli.py`:
```python
import json
from types import SimpleNamespace

from backend.pipeline import transcribe as transcribe_mod
from backend.pipeline import ingest as ingest_mod


def _fake_model():
    class FakeModel:
        def transcribe(self, wav_path, **kwargs):
            info = SimpleNamespace(language="en", duration=3.0)
            segs = [SimpleNamespace(id=0, start=0.0, end=3.0, text="test")]
            return iter(segs), info
    return FakeModel()


def test_end_to_end_pipeline_creates_all_artifacts(tmp_path, synthetic_mp4, monkeypatch):
    monkeypatch.setattr(transcribe_mod, "_load_model", lambda n, d: _fake_model())

    workspace_root = tmp_path / "workspace"
    ws = ingest_mod.run_pipeline(
        synthetic_mp4,
        workspace_root=workspace_root,
        model_name="base",
        device="cpu",
        force=False,
    )

    assert ws.meta_raw_path.exists()
    assert ws.audio_path.exists()
    assert ws.transcript_path.exists()
    assert ws.frames_dir.is_dir()
    assert len(list(ws.frames_dir.glob("frame_*.jpg"))) >= 2
    assert ws.log_path.exists()

    # meta_raw validates
    meta = json.loads(ws.meta_raw_path.read_text())
    assert meta["filename"] == "synth.mp4"


def test_cache_skip_reruns_nothing_when_complete(tmp_path, synthetic_mp4, monkeypatch):
    monkeypatch.setattr(transcribe_mod, "_load_model", lambda n, d: _fake_model())
    workspace_root = tmp_path / "workspace"

    ingest_mod.run_pipeline(synthetic_mp4, workspace_root=workspace_root,
                            model_name="base", device="cpu", force=False)

    # Break the source mp4: pipeline should not re-probe / re-extract
    synthetic_mp4_copy_broken = synthetic_mp4  # same path; we check by call count instead
    calls = {"probe": 0, "audio": 0, "frames": 0, "transcribe": 0}

    from backend.pipeline import probe, audio, frames
    monkeypatch.setattr(probe, "probe_video",
                        lambda *a, **kw: calls.__setitem__("probe", calls["probe"] + 1))
    monkeypatch.setattr(audio, "extract_audio",
                        lambda *a, **kw: calls.__setitem__("audio", calls["audio"] + 1))
    monkeypatch.setattr(frames, "sample_frames",
                        lambda *a, **kw: calls.__setitem__("frames", calls["frames"] + 1))
    monkeypatch.setattr(transcribe_mod, "transcribe",
                        lambda *a, **kw: calls.__setitem__("transcribe", calls["transcribe"] + 1))

    ingest_mod.run_pipeline(synthetic_mp4_copy_broken, workspace_root=workspace_root,
                            model_name="base", device="cpu", force=False)

    assert calls == {"probe": 0, "audio": 0, "frames": 0, "transcribe": 0}


def test_force_reruns_all_stages(tmp_path, synthetic_mp4, monkeypatch):
    monkeypatch.setattr(transcribe_mod, "_load_model", lambda n, d: _fake_model())
    workspace_root = tmp_path / "workspace"

    ingest_mod.run_pipeline(synthetic_mp4, workspace_root=workspace_root,
                            model_name="base", device="cpu", force=False)

    calls = {"probe": 0}
    from backend.pipeline import probe
    original_probe = probe.probe_video

    def counting_probe(*a, **kw):
        calls["probe"] += 1
        return original_probe(*a, **kw)

    monkeypatch.setattr(probe, "probe_video", counting_probe)

    ingest_mod.run_pipeline(synthetic_mp4, workspace_root=workspace_root,
                            model_name="base", device="cpu", force=True)
    assert calls["probe"] == 1
```

- [ ] **Step 2: Run — expect FAIL (module missing)**

Run:
```
pytest tests/test_ingest_cli.py -v
```

- [ ] **Step 3: Implement orchestrator**

`backend/pipeline/ingest.py`:
```python
from __future__ import annotations
import argparse
import sys
from pathlib import Path

from backend.pipeline.workspace import Workspace
from backend.pipeline.logging_setup import get_logger
from backend.pipeline import probe as probe_stage
from backend.pipeline import audio as audio_stage
from backend.pipeline import frames as frames_stage
from backend.pipeline import transcribe as transcribe_stage


DEFAULT_WORKSPACE_ROOT = Path(__file__).resolve().parents[2] / "workspace"


def run_pipeline(
    mp4_path: Path,
    workspace_root: Path = DEFAULT_WORKSPACE_ROOT,
    model_name: str = "base",
    device: str = "auto",
    force: bool = False,
) -> Workspace:
    if not mp4_path.exists():
        raise FileNotFoundError(f"Input MP4 not found: {mp4_path}")

    ws = Workspace.for_video(mp4_path, root=workspace_root)
    ws.ensure()
    log = get_logger(ws.log_path)
    log.info("Pipeline start: %s  workspace=%s  force=%s", mp4_path, ws.root, force)

    resolved_device = _resolve_device(device)
    log.info("Device: %s", resolved_device)

    # Stage 1: probe
    if force or not ws.meta_raw_path.exists():
        log.info("Stage 1/4: probing video")
        probe_stage.probe_video(mp4_path, ws)
    else:
        log.info("Stage 1/4: skipped (meta_raw.json exists)")

    # Stage 2: audio
    if force or not ws.audio_path.exists():
        log.info("Stage 2/4: extracting audio")
        audio_stage.extract_audio(mp4_path, ws)
    else:
        log.info("Stage 2/4: skipped (audio_processed.wav exists)")

    # Stage 3: frames
    existing_frames = list(ws.frames_dir.glob("frame_*.jpg"))
    if force or not existing_frames:
        log.info("Stage 3/4: sampling frames at 1 FPS")
        frames_stage.sample_frames(mp4_path, ws, fps=1, longest_side=512, jpeg_quality=85)
    else:
        log.info("Stage 3/4: skipped (%d frames cached)", len(existing_frames))

    # Stage 4: transcribe
    if force or not ws.transcript_path.exists():
        log.info("Stage 4/4: transcribing audio with faster-whisper (%s)", model_name)
        transcribe_stage.transcribe(ws.audio_path, ws, model_name=model_name, device=resolved_device)
    else:
        log.info("Stage 4/4: skipped (transcript.json exists)")

    log.info("Pipeline done: %s", ws.root)
    return ws


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m backend.pipeline.ingest",
        description="Phase 1: ingest an MP4 and cache artifacts for downstream stages.",
    )
    p.add_argument("video", type=Path, help="Path to input MP4 file")
    p.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE_ROOT,
                   help="Workspace root directory (default: <repo>/workspace)")
    p.add_argument("--model", default="base", help="faster-whisper model name (default: base)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="Device for Whisper inference (default: auto)")
    p.add_argument("--force", action="store_true", help="Re-run all stages even if cached")
    args = p.parse_args(argv)

    run_pipeline(
        args.video,
        workspace_root=args.workspace,
        model_name=args.model,
        device=args.device,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run CLI tests — expect PASS**

Run:
```
pytest tests/test_ingest_cli.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Smoke-test the actual CLI**

Run (with a real short MP4 in `videos/` — or use the synthetic fixture path):
```
python -m backend.pipeline.ingest tests/fixtures/synth.mp4 --device cpu
```
Expected: creates `workspace/synth/` with all four artifacts + `ingest.log`. (If you don't have a real MP4 yet, skip this smoke step and rely on the pytest end-to-end.)

- [ ] **Step 6: Commit**

```
git add backend/pipeline/ingest.py tests/test_ingest_cli.py
git commit -m "feat(pipeline): CLI orchestrator with per-stage cache and --force"
```

---

## Task 11: Run full fast test suite as regression gate

- [ ] **Step 1: Run all fast tests**

Run:
```
pytest -v -m "not slow"
```
Expected: all tests pass (count: schemas 3, workspace 3, probe 1, audio 1, frames 2, transcribe 1, ingest 3 = **14 passed**).

- [ ] **Step 2: If any fail, diagnose and fix before proceeding**

Do not proceed to Phase 2 with a red test suite.

- [ ] **Step 3: Commit any fixes**

```
git add -A
git commit -m "fix(pipeline): address Phase 1 regressions"
```

(Skip if no changes.)

---

## Task 12: Documentation — run instructions

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read current README**

Run:
```
cat README.md
```

- [ ] **Step 2: Append Phase 1 usage section to README.md**

Append:
```markdown
## Phase 1: Ingestion

Process a single MP4 into cached artifacts under `workspace/<video_stem>/`:

```bash
# Activate venv first
python -m backend.pipeline.ingest path/to/video.mp4
```

Optional flags:
- `--workspace DIR` — override workspace root (default: `./workspace`)
- `--model NAME` — faster-whisper model: `tiny`, `base`, `small`, `medium` (default: `base`)
- `--device {auto,cpu,cuda}` — inference device (default: `auto`)
- `--force` — ignore cache and re-run all stages

Artifacts produced:
- `workspace/<stem>/meta_raw.json` — video metadata
- `workspace/<stem>/audio_processed.wav` — mono 16 kHz PCM
- `workspace/<stem>/frames_cache/frame_XXXXXX.jpg` — 1 FPS, longest-side 512 px
- `workspace/<stem>/transcript.json` — sentence-level transcript with timestamps
- `workspace/<stem>/ingest.log` — run log

### Running tests

```bash
pip install -e ".[dev]"
pytest -v -m "not slow"       # fast suite (~seconds)
pytest -v -m slow             # real-Whisper test (downloads ~140 MB first run)
```
```

- [ ] **Step 3: Commit**

```
git add README.md
git commit -m "docs: Phase 1 usage instructions"
```

---

## Self-Review Results

**1. Spec coverage (PDD §4.1):**
- ✅ Output `meta_raw.json` — Task 6
- ✅ Output `audio_processed.wav` (mono 16kHz) — Task 7
- ✅ Output `transcript.json` (sentence-level + timestamps) — Task 9
- ✅ Output `frames_cache/frame_XXXXXX.jpg` at 1 FPS — Task 8
- ✅ Cache / skip behavior — Task 10
- ✅ `.venv` + `pyproject.toml` — Task 0
- ✅ Offline capability — model cache pinned to `workspace/models/`, ffmpeg bundled
- ✅ Multi-genre — no genre-specific logic in Phase 1

**2. Placeholder scan:** No TBDs, no "implement later", every code step has complete code, every test step has assertions. The Task 12 README section has nested triple backticks inside fenced markdown — that's intentional for the appended doc.

**3. Type consistency:**
- `Workspace.for_video(video, root=...)` signature — consistent across Tasks 3, 6, 7, 8, 9, 10 ✅
- `Workspace.ensure()` — consistent ✅
- `probe_video(mp4_path, workspace) -> MetaRaw` — consistent ✅
- `extract_audio(mp4_path, workspace) -> Path` — consistent ✅
- `sample_frames(mp4_path, workspace, fps, longest_side, jpeg_quality) -> list[Path]` — consistent ✅
- `transcribe(wav_path, workspace, model_name, device) -> Transcript` — consistent ✅
- `run_pipeline(mp4_path, workspace_root, model_name, device, force) -> Workspace` — consistent ✅
- All pydantic model field names match between schema definition (Task 2) and writers (Tasks 6, 9) ✅

---

**Plan complete and saved to `docs/superpowers/plans/2026-04-24-phase1-ingestion.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
