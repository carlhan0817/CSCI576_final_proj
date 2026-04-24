from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Workspace:
    source_video: Path
    root: Path

    @classmethod
    def for_video(cls, video: Path, root: Path) -> "Workspace":
        stem = video.stem
        return cls(source_video=video, root=root / stem)

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    @property
    def meta_raw_path(self) -> Path:
        return self.root / "meta_raw.json"

    @property
    def audio_path(self) -> Path:
        return self.root / "audio_processed.wav"

    @property
    def transcript_path(self) -> Path:
        return self.root / "transcript.json"

    @property
    def frames_dir(self) -> Path:
        return self.root / "frames_cache"

    @property
    def log_path(self) -> Path:
        return self.root / "ingest.log"
