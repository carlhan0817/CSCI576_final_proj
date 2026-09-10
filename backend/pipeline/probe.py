from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import imageio_ffmpeg

from backend.pipeline.schemas import MetaRaw
from backend.pipeline.workspace import Workspace


def probe_video(mp4_path: Path, workspace: Workspace) -> MetaRaw:
    gen = imageio_ffmpeg.read_frames(str(mp4_path))
    try:
        info = gen.__next__()  # first yield is the metadata dict
    finally:
        gen.close()

    audio_codec = info.get("audio_codec", "none") or "none"
    has_audio = bool(info.get("audio_codec"))

    meta = MetaRaw(
        filename=mp4_path.name,
        duration_sec=float(info["duration"]),
        fps=float(info["fps"]),
        width=int(info["source_size"][0]),
        height=int(info["source_size"][1]),
        video_codec=info["codec"],
        audio_codec=audio_codec,
        has_audio=has_audio,
        ingested_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    workspace.meta_raw_path.write_text(meta.model_dump_json(indent=2))
    return meta
