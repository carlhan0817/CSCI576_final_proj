"""FastAPI router for player endpoints."""
from __future__ import annotations
import logging
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
