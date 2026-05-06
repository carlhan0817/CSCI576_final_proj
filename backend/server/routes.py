"""FastAPI router for player endpoints."""
from __future__ import annotations
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, ValidationError

from backend.pipeline.schemas import Metadata
from backend.server.config import ServerConfig


_log = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# Upload chunk size for streaming multipart to disk
_UPLOAD_CHUNK_BYTES = 1 << 20  # 1 MB

# Allowed characters in a sanitized filename stem (alnum, dash, underscore, dot)
_FNAME_SAFE = re.compile(r"[^A-Za-z0-9._-]")


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
        return Metadata.model_validate_json(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"metadata invalid: {exc}")


def _atomic_write_text(path, text: str) -> None:
    """Write to <path>.tmp then os.replace onto <path> — crash-safe."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
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


def _sanitize_stem(name: str) -> str:
    stem = Path(name).stem
    stem = _FNAME_SAFE.sub("_", stem).strip("._-")
    return stem or "video"


def _unique_video_path(videos_root: Path, stem: str) -> Path:
    candidate = videos_root / f"{stem}.mp4"
    if not candidate.exists():
        return candidate
    i = 2
    while (videos_root / f"{stem}_{i}.mp4").exists():
        i += 1
    return videos_root / f"{stem}_{i}.mp4"


def _run_pipeline_job(
    job_id: str,
    mp4_path: Path,
    workspace_root: Path,
    jobs: dict,
    lock: threading.Lock,
) -> None:
    """Background worker — drives Phase 1→4 for one uploaded MP4."""
    from backend.pipeline.ingest import run_pipeline

    try:
        with lock:
            jobs[job_id]["status"] = "running"
            jobs[job_id]["started_at"] = time.time()
        run_pipeline(mp4_path, workspace_root=workspace_root)
        with lock:
            jobs[job_id]["status"] = "done"
            jobs[job_id]["finished_at"] = time.time()
    except Exception as exc:
        _log.exception("Job %s failed", job_id)
        with lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = f"{type(exc).__name__}: {exc}"
            jobs[job_id]["finished_at"] = time.time()


@router.post("/upload")
async def upload_video(request: Request, file: UploadFile = File(...)) -> dict:
    """Accept a multipart MP4, save under videos/, kick off background analysis."""
    cfg = _config(request)
    if not file.filename or not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="only .mp4 files are accepted")

    cfg.videos_root.mkdir(parents=True, exist_ok=True)
    stem = _sanitize_stem(file.filename)
    mp4_path = _unique_video_path(cfg.videos_root, stem)

    bytes_written = 0
    try:
        with open(mp4_path, "wb") as f:
            while True:
                chunk = await file.read(_UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                f.write(chunk)
                bytes_written += len(chunk)
    except Exception:
        if mp4_path.exists():
            mp4_path.unlink(missing_ok=True)
        raise

    if bytes_written == 0:
        mp4_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="empty upload")

    video_id = mp4_path.stem
    job_id = uuid.uuid4().hex[:12]

    jobs = request.app.state.jobs
    lock = request.app.state.jobs_lock
    with lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "video_id": video_id,
            "filename": mp4_path.name,
            "bytes": bytes_written,
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "error": None,
        }

    threading.Thread(
        target=_run_pipeline_job,
        args=(job_id, mp4_path, cfg.workspace_root, jobs, lock),
        daemon=True,
    ).start()

    return {
        "job_id": job_id,
        "video_id": video_id,
        "filename": mp4_path.name,
        "bytes": bytes_written,
    }


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request) -> dict:
    jobs = request.app.state.jobs
    lock = request.app.state.jobs_lock
    with lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        snapshot = dict(job)
    if snapshot["started_at"] is not None:
        end = snapshot["finished_at"] or time.time()
        snapshot["elapsed_sec"] = round(end - snapshot["started_at"], 1)
    else:
        snapshot["elapsed_sec"] = 0.0
    return snapshot


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
            md = Metadata.model_validate_json(meta_path.read_text(encoding="utf-8"))
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
