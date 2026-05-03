"""Phase 4.5 player backend tests (happy + sad paths)."""
from __future__ import annotations
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# ──────────────────────────────────────────────────────────────────────────────
# ServerConfig
# ──────────────────────────────────────────────────────────────────────────────

def test_server_config_resolves_defaults_to_repo_paths(tmp_path, monkeypatch):
    monkeypatch.delenv("MM_WORKSPACE", raising=False)
    monkeypatch.delenv("MM_VIDEOS", raising=False)
    monkeypatch.delenv("MM_FRONTEND", raising=False)

    from backend.server.config import ServerConfig
    cfg = ServerConfig.from_env()
    # Defaults should land somewhere under the repo root and be absolute.
    assert cfg.workspace_root.is_absolute()
    assert cfg.videos_root.is_absolute()
    assert cfg.frontend_root.is_absolute()
    assert cfg.workspace_root.name == "workspace"
    assert cfg.videos_root.name == "videos"
    assert cfg.frontend_root.name == "frontend"


def test_server_config_honors_env_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("MM_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("MM_VIDEOS", str(tmp_path / "vid"))
    monkeypatch.setenv("MM_FRONTEND", str(tmp_path / "fe"))

    from backend.server.config import ServerConfig
    cfg = ServerConfig.from_env()
    assert cfg.workspace_root == (tmp_path / "ws").resolve()
    assert cfg.videos_root == (tmp_path / "vid").resolve()
    assert cfg.frontend_root == (tmp_path / "fe").resolve()


# ──────────────────────────────────────────────────────────────────────────────
# Shared test fixture
# ──────────────────────────────────────────────────────────────────────────────

VALID_METADATA: dict = {
    "video_info": {
        "filename": "demo.mp4",
        "duration_sec": 60.0,
        "fps": 30.0,
        "width": 1280,
        "height": 720,
        "analysis_version": "0.3.0",
        "verified_by_human": False,
    },
    "segments": [
        {
            "segment_id": 0,
            "start_sec": 0.0,
            "end_sec": 30.0,
            "label": "core_content",
            "confidence": 0.9,
            "evidence": {
                "visual_score": 0.5,
                "audio_score": 0.5,
                "text_score": 0.5,
                "triggered_rules": [],
            },
            "summary": "first half",
            "user_corrected": False,
        },
        {
            "segment_id": 1,
            "start_sec": 30.0,
            "end_sec": 60.0,
            "label": "sponsorship",
            "confidence": 0.8,
            "evidence": {
                "visual_score": 0.4,
                "audio_score": 0.6,
                "text_score": 0.7,
                "triggered_rules": ["sponsor_keyword"],
            },
            "summary": "ad read",
            "user_corrected": False,
        },
    ],
    "chapters": [],
    "skip_suggestions": [1],
}


@pytest.fixture
def fake_repo(tmp_path):
    """Build workspace/videos/frontend dirs with one analyzed video."""
    workspace = tmp_path / "workspace"
    videos = tmp_path / "videos"
    frontend = tmp_path / "frontend"
    static_dir = frontend / "static"
    (workspace / "demo").mkdir(parents=True)
    videos.mkdir()
    static_dir.mkdir(parents=True)

    (workspace / "demo" / "metadata.json").write_text(json.dumps(VALID_METADATA))
    (videos / "demo.mp4").write_bytes(b"\x00" * 16)  # stub bytes; range req works on any size
    (frontend / "index.html").write_text("<html><body>hi</body></html>")
    return tmp_path


def _make_client(workspace: Path, videos: Path, frontend: Path) -> TestClient:
    """Build a TestClient with an explicit ServerConfig — no env vars, no module reload."""
    from backend.server.config import ServerConfig
    from backend.server.app import create_app
    cfg = ServerConfig(
        workspace_root=workspace,
        videos_root=videos,
        frontend_root=frontend,
    )
    return TestClient(create_app(cfg))


@pytest.fixture
def client(fake_repo):
    return _make_client(
        workspace=fake_repo / "workspace",
        videos=fake_repo / "videos",
        frontend=fake_repo / "frontend",
    )


@pytest.fixture
def empty_client(tmp_path):
    """Server pointed at empty workspace — for sad-path 'no videos' coverage."""
    (tmp_path / "workspace").mkdir()
    (tmp_path / "videos").mkdir()
    (tmp_path / "frontend" / "static").mkdir(parents=True)
    (tmp_path / "frontend" / "index.html").write_text("<html></html>")
    return _make_client(
        workspace=tmp_path / "workspace",
        videos=tmp_path / "videos",
        frontend=tmp_path / "frontend",
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /api/videos
# ──────────────────────────────────────────────────────────────────────────────

def test_get_videos_returns_seeded_video_happy(client):
    r = client.get("/api/videos")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    assert len(body) == 1
    item = body[0]
    assert item["video_id"] == "demo"
    assert item["filename"] == "demo.mp4"
    assert item["duration_sec"] == 60.0
    assert item["verified_by_human"] is False


def test_get_videos_returns_empty_list_when_no_videos_sad_a(empty_client):
    r = empty_client.get("/api/videos")
    assert r.status_code == 200
    assert r.json() == []
