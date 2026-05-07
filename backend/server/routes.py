"""FastAPI router for player endpoints."""
from __future__ import annotations
import logging
import os
import threading
from typing import Dict, List, Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, UploadFile
from pydantic import BaseModel, ValidationError

from backend.pipeline.schemas import Metadata
from backend.server.config import ServerConfig


_log = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# In-memory job tracker: video_id → {status, error}
_jobs: Dict[str, Dict] = {}
_jobs_lock = threading.Lock()


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


class UploadStatus(BaseModel):
    video_id: str
    status: Literal["processing", "done", "failed"]
    progress: int = 0       # 0–100
    message: str = ""
    error: str = ""


def _run_pipeline_bg(video_path, workspace_root, video_id: str) -> None:
    from backend.pipeline.ingest import run_pipeline

    def _cb(pct: int, msg: str) -> None:
        with _jobs_lock:
            _jobs[video_id]["progress"] = pct
            _jobs[video_id]["message"]  = msg

    try:
        run_pipeline(video_path, workspace_root=workspace_root, progress_cb=_cb)
        with _jobs_lock:
            _jobs[video_id] = {"status": "done", "progress": 100, "message": "Done", "error": ""}
    except Exception as exc:
        _log.exception("Pipeline failed for %s", video_id)
        with _jobs_lock:
            _jobs[video_id]["status"] = "failed"
            _jobs[video_id]["error"]  = str(exc)


@router.post("/upload", response_model=UploadStatus)
async def upload_video(file: UploadFile, background_tasks: BackgroundTasks, request: Request) -> UploadStatus:
    cfg = _config(request)

    if not file.filename or not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only .mp4 files are accepted")

    safe_name = os.path.basename(file.filename)
    video_id  = safe_name[:-4]  # strip .mp4
    dest      = cfg.videos_root / safe_name

    # Write the uploaded bytes to disk.
    content = await file.read()
    dest.write_bytes(content)

    with _jobs_lock:
        if _jobs.get(video_id, {}).get("status") == "processing":
            return UploadStatus(video_id=video_id, status="processing")
        _jobs[video_id] = {"status": "processing", "progress": 0, "message": "Starting…", "error": ""}

    background_tasks.add_task(_run_pipeline_bg, dest, cfg.workspace_root, video_id)
    return UploadStatus(video_id=video_id, status="processing")


@router.get("/upload/{video_id}/status", response_model=UploadStatus)
def upload_status(video_id: str) -> UploadStatus:
    with _jobs_lock:
        job = _jobs.get(video_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No upload job found for '{video_id}'")
    return UploadStatus(video_id=video_id, **job)


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
