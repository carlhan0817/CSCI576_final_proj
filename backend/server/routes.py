"""FastAPI router for player endpoints."""
from __future__ import annotations
import logging
import os
from typing import List

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ValidationError

from backend.pipeline.schemas import Metadata
from backend.server.config import ServerConfig


_log = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


class VideoSummary(BaseModel):
    video_id: str
    filename: str
    duration_sec: float
    verified_by_human: bool


def _config(request: Request) -> ServerConfig:
    return request.app.state.config


@router.get("/metadata/{video_id}", response_model=Metadata)
def get_metadata(video_id: str, request: Request) -> Metadata:
    cfg = _config(request)
    meta_path = cfg.workspace_root / video_id / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"metadata not found for video '{video_id}'")
    try:
        return Metadata.model_validate_json(meta_path.read_text())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"metadata invalid: {exc}")


def _atomic_write_text(path, text: str) -> None:
    """Write to <path>.tmp then os.replace onto <path> — crash-safe."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(str(tmp), str(path))


@router.post("/metadata/{video_id}", response_model=Metadata)
def save_metadata(video_id: str, body: Metadata, request: Request) -> Metadata:
    cfg = _config(request)
    workspace_dir = cfg.workspace_root / video_id
    if not workspace_dir.exists():
        raise HTTPException(status_code=404, detail=f"workspace for '{video_id}' not found")

    # Server-enforced invariant: any successful POST means a human reviewed it.
    body.video_info.verified_by_human = True

    meta_path = workspace_dir / "metadata.json"
    _atomic_write_text(meta_path, body.model_dump_json(indent=2))
    return body


@router.get("/videos", response_model=List[VideoSummary])
def list_videos(request: Request) -> List[VideoSummary]:
    cfg = _config(request)
    out: List[VideoSummary] = []
    if not cfg.videos_root.exists():
        return out
    for mp4 in sorted(cfg.videos_root.glob("*.mp4")):
        stem = mp4.stem
        meta_path = cfg.workspace_root / stem / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            md = Metadata.model_validate_json(meta_path.read_text())
        except ValidationError:
            _log.warning("Skipping %s: metadata failed schema validation", meta_path)
            continue
        out.append(VideoSummary(
            video_id=stem,
            filename=md.video_info.filename,
            duration_sec=md.video_info.duration_sec,
            verified_by_human=md.video_info.verified_by_human,
        ))
    return out
