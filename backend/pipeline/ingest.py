from __future__ import annotations
import argparse
import sys
from pathlib import Path

from backend.pipeline.workspace import Workspace
from backend.pipeline.logging_setup import get_logger
from backend.pipeline import probe as probe_stage
from backend.pipeline import audio as audio_stage
from backend.pipeline import frames as frames_stage
from backend.pipeline import transcribe as transcribe_stage


DEFAULT_WORKSPACE_ROOT = Path(__file__).resolve().parents[2] / "workspace"


def run_pipeline(
    mp4_path: Path,
    workspace_root: Path = DEFAULT_WORKSPACE_ROOT,
    model_name: str = "base",
    device: str = "auto",
    force: bool = False,
) -> Workspace:
    if not mp4_path.exists():
        raise FileNotFoundError(f"Input MP4 not found: {mp4_path}")

    ws = Workspace.for_video(mp4_path, root=workspace_root)
    ws.ensure()
    log = get_logger(ws.log_path)
    log.info("Pipeline start: %s  workspace=%s  force=%s", mp4_path, ws.root, force)

    resolved_device = _resolve_device(device)
    log.info("Device: %s", resolved_device)

    if force or not ws.meta_raw_path.exists():
        log.info("Stage 1/4: probing video")
        probe_stage.probe_video(mp4_path, ws)
    else:
        log.info("Stage 1/4: skipped (meta_raw.json exists)")

    if force or not ws.audio_path.exists():
        log.info("Stage 2/4: extracting audio")
        audio_stage.extract_audio(mp4_path, ws)
    else:
        log.info("Stage 2/4: skipped (audio_processed.wav exists)")

    existing_frames = list(ws.frames_dir.glob("frame_*.jpg"))
    if force or not existing_frames:
        log.info("Stage 3/4: sampling frames at 1 FPS")
        frames_stage.sample_frames(mp4_path, ws, fps=1, longest_side=512, jpeg_quality=85)
    else:
        log.info("Stage 3/4: skipped (%d frames cached)", len(existing_frames))

    if force or not ws.transcript_path.exists():
        log.info("Stage 4/4: transcribing audio with faster-whisper (%s)", model_name)
        transcribe_stage.transcribe(ws.audio_path, ws, model_name=model_name, device=resolved_device)
    else:
        log.info("Stage 4/4: skipped (transcript.json exists)")

    from backend.pipeline.features.run import run_phase2
    if force or not ws.visual_features_path.exists() \
             or not ws.audio_features_path.exists() \
             or not ws.text_features_path.exists():
        log.info("Phase 2: Extracting multimodal features")
        run_phase2(ws, device=resolved_device, force=force)
    else:
        log.info("Phase 2: skipped (all feature files exist)")

    try:
        from backend.pipeline.fusion.run import run_fusion
        if force or not ws.metadata_path.exists():
            log.info("Phase 3: Running fusion pipeline")
            run_fusion(ws)
        else:
            log.info("Phase 3: skipped (metadata.json exists)")
    except Exception as exc:
        log.warning("Phase 3 skipped due to error: %s", exc)

    log.info("Pipeline done: %s", ws.root)
    return ws


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m backend.pipeline.ingest",
        description="Phase 1: ingest an MP4 and cache artifacts for downstream stages.",
    )
    p.add_argument("video", type=Path, help="Path to input MP4 file")
    p.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE_ROOT,
                   help="Workspace root directory (default: <repo>/workspace)")
    p.add_argument("--model", default="base", help="faster-whisper model name (default: base)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="Device for Whisper inference (default: auto)")
    p.add_argument("--force", action="store_true", help="Re-run all stages even if cached")
    args = p.parse_args(argv)

    run_pipeline(
        args.video,
        workspace_root=args.workspace,
        model_name=args.model,
        device=args.device,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
