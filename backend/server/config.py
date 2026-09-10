"""Resolved filesystem paths used by the player backend."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ServerConfig:
    workspace_root: Path
    videos_root: Path
    frontend_root: Path

    @classmethod
    def from_env(cls) -> ServerConfig:
        return cls(
            workspace_root=Path(os.environ.get("MM_WORKSPACE", _REPO_ROOT / "workspace")).resolve(),
            videos_root=Path(os.environ.get("MM_VIDEOS", _REPO_ROOT / "videos")).resolve(),
            frontend_root=Path(os.environ.get("MM_FRONTEND", _REPO_ROOT / "frontend")).resolve(),
        )
