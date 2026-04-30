"""
Phase 2 CLI orchestrator.

Runs all three feature extractors in sequence, releasing GPU memory between
stages, then optionally chains into Phase 3 fusion.

Usage:
    python -m backend.pipeline.features.run path/to/video.mp4
    python -m backend.pipeline.features.run path/to/video.mp4 --device cuda
    python -m backend.pipeline.features.run path/to/video.mp4 --force
"""
from __future__ import annotations
import argparse
import gc
import sys
from pathlib import Path

import torch

from backend.pipeline.workspace import Workspace
from backend.pipeline.logging_setup import get_logger
from backend.pipeline.features.visual import extract_visual_features
from backend.pipeline.features.audio import extract_audio_features
from backend.pipeline.features.text import extract_text_features


DEFAULT_WORKSPACE_ROOT = Path(__file__).resolve().parents[3] / "workspace"


def run_phase2(
    workspace: Workspace,
    device: str = "cpu",
    force: bool = False,
) -> None:
    """Run all three Phase 2 sub-pipelines. Models are unloaded between stages."""
    log = get_logger(workspace.log_path)

    if force:
        for p in [
            workspace.visual_features_path,
            workspace.audio_features_path,
            workspace.text_features_path,
            workspace.text_embeddings_path,
        ]:
            if p.exists():
                p.unlink()
                log.info("Removed cached file: %s", p.name)

    log.info("=" * 60)
    log.info("Phase 2: Feature extraction  device=%s", device)

    log.info("Phase 2 [1/3]: Visual features (CLIP + OpenCV)")
    extract_visual_features(workspace, device=device)
    _release_gpu()

    log.info("Phase 2 [2/3]: Audio features (Silero VAD + librosa)")
    extract_audio_features(workspace, device=device)
    _release_gpu()

    log.info("Phase 2 [3/3]: Text features (MiniLM)")
    extract_text_features(workspace, device=device)
    _release_gpu()

    log.info("Phase 2 done.")


def _release_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m backend.pipeline.features.run",
        description="Phase 2: extract multimodal features from a Phase 1 workspace.",
    )
    p.add_argument("video", type=Path, help="Path to the original MP4 file.")
    p.add_argument(
        "--workspace", type=Path, default=DEFAULT_WORKSPACE_ROOT,
        help="Workspace root directory (default: <repo>/workspace).",
    )
    p.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"],
        help="Inference device (default: auto).",
    )
    p.add_argument("--force", action="store_true", help="Delete cached features and re-run.")
    args = p.parse_args(argv)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ws = Workspace.for_video(args.video, root=args.workspace)
    if not ws.audio_path.exists():
        print(
            f"ERROR: Phase 1 artifacts not found in {ws.root}.\n"
            "Run Phase 1 first: python -m backend.pipeline.ingest <video.mp4>",
            file=sys.stderr,
        )
        return 1

    run_phase2(ws, device=device, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
