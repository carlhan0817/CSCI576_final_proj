start file
usage of each folder:
    test:the unit test for the app and the testing video
    prompts:save all documentation markdown file here
    app: main program
    data: temp folder for the new generateed data


1. Set Up the Environment
Since you will likely be managing this in VSCode, setting up your Python environment and kernel correctly from day one will save your team a lot of headaches.

Initialize the Virtual Environment: Run python -m venv .venv in your root folder. Make sure your VSCode interpreter is pointed to this specific .venv.

**I run python -m venv .venv under src folder.**

Create pyproject.toml: Instead of a messy requirements.txt, define your dependencies here. You will need to install:

Core: numpy, pandas, pydantic

Media: opencv-python, librosa, ffmpeg-python (or imageio-ffmpeg)

AI Models: openai-whisper, open_clip_torch, sentence-transformers, torch

Backend: fastapi, uvicorn

**Now open terminal in VSCode and cd into CSCI576_final_proj folder, run the following command in order:**
python -m pip install --upgrade pip setuptools wheel
pip install -e .    



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

## Phase 4.5: Player

Run the player after a video has been ingested + analyzed end-to-end (Phases 1–3 produce `workspace/<stem>/metadata.json`).

```bash
# 1. Drop the source MP4 into ./videos/<stem>.mp4 (must match the workspace stem)
# 2. Run the server
uvicorn backend.server.app:app --port 8000

# 3. Open http://localhost:8000/ in Chrome
```

Endpoints:
- `GET  /api/videos`              — list analyzed videos
- `GET  /api/metadata/{video_id}` — read metadata.json
- `POST /api/metadata/{video_id}` — write metadata.json (full replacement; server forces `verified_by_human=true`)
- `GET  /videos/<stem>.mp4`       — static MP4 (browser uses Range for seek)

Tests: `pytest tests/test_server.py -v`

