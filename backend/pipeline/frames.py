from __future__ import annotations

import subprocess
from pathlib import Path

import imageio_ffmpeg

from backend.pipeline.workspace import Workspace

_FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()


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

    scale_expr = (
        f"scale='if(gt(iw,ih),{longest_side},-2)':'if(gt(iw,ih),-2,{longest_side})'"
    )
    qv = max(2, min(31, round((100 - jpeg_quality) / 3)))

    cmd = [
        _FFMPEG_EXE,
        "-y",
        "-i", str(mp4_path),
        "-vf", f"fps={fps},{scale_expr}",
        "-q:v", str(qv),
        "-start_number", "0",
        pattern,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return sorted(out_dir.glob("frame_*.jpg"))
