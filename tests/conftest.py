import subprocess
import wave
from pathlib import Path

import imageio_ffmpeg
import pytest

from backend.pipeline.audio import extract_audio as phase1_extract_audio
from backend.pipeline.frames import sample_frames
from backend.pipeline.workspace import Workspace

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
