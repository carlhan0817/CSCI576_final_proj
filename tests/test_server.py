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
