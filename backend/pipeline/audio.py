from __future__ import annotations

import subprocess
from pathlib import Path

import imageio_ffmpeg

from backend.pipeline.workspace import Workspace

_FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()


def extract_audio(mp4_path: Path, workspace: Workspace) -> Path:
    out = workspace.audio_path
    cmd = [
        _FFMPEG_EXE,
        "-y",
        "-i", str(mp4_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-acodec", "pcm_s16le",
        str(out),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return out
