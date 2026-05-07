"""
Phase 3 main entry point: run the full fusion pipeline on a workspace.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from backend.pipeline.workspace import Workspace
from backend.pipeline.logging_setup import get_logger
from backend.pipeline.schemas import MetaRaw, Transcript

from backend.pipeline.fusion.align import load_phase2_features, build_per_second_grid
from backend.pipeline.fusion.rules import run_all_rules
from backend.pipeline.fusion.boundaries import find_boundaries
from backend.pipeline.fusion.classify import classify_segments
from backend.pipeline.fusion.smooth import smooth_pipeline
from backend.pipeline.fusion.export import export_metadata


def run_fusion(workspace: Workspace) -> Path:
    log = get_logger(workspace.log_path)
    log.info("=" * 60)
    log.info("Phase 3: Fusion pipeline starting")

    # 0. Load Phase 1 metadata + transcript
    meta_raw = MetaRaw.model_validate(json.loads(workspace.meta_raw_path.read_text()))
    transcript = Transcript.model_validate(json.loads(workspace.transcript_path.read_text()))
    log.info("Loaded meta_raw and transcript: %.1fs duration, %d sentences",
             meta_raw.duration_sec, len(transcript.segments))

    # 1. Load Phase 2 features
    visual, audio, text = load_phase2_features(workspace)
    log.info("Loaded Phase 2 features: %d frames, %d audio segs, %d text segs",
             len(visual.frames), len(audio.segments), len(text.segments))

    # 2. Build per-second grid
    grid = build_per_second_grid(visual, audio, text, meta_raw.duration_sec)
    log.info("Built per-second grid: T=%d seconds, columns=%d",
             len(grid["is_speech"]), len(grid))

    # 3. Run hard rules
    rule_hits = run_all_rules(grid, text)
    log.info("Hard rules fired: %d hits", len(rule_hits))
    for h in rule_hits:
        log.info("  rule [%s] %ds-%ds → %s (conf=%.2f)",
                 h.rule_name, h.start_sec, h.end_sec, h.label, h.confidence)

    # 4. Find boundaries
    raw_boundaries = find_boundaries(grid)
    log.info("Boundary candidates: %d", len(raw_boundaries))

    # Build the hard-cut second set so classify can stamp has_hard_cut_before.
    T = len(grid["is_speech"])
    hard_cut_set = {t for t in range(T) if grid["is_hard_cut"][t]}

    # 5. Classify segments
    segments = classify_segments(grid, raw_boundaries, rule_hits, transcript.segments, hard_cut_set)
    log.info("Initial segments: %d", len(segments))

    # 6. Smooth (Fix 1 + Fix 3 + Fix 4 all live inside smooth_pipeline)
    segments = smooth_pipeline(segments, raw_boundaries=raw_boundaries, grid=grid)
    log.info("Smoothed segments: %d", len(segments))

    # 7. Export (Fix 2: raw_boundaries → natural_break_candidates; ad_slots → ad_insertion_candidates)
    # n_ads=4: supports videos with up to 4 inserted ad breaks; MIN_COMPOSITE_SCORE in
    # ad_slots.py prevents spurious low-confidence candidates from filling the extra slot.
    out_path = export_metadata(workspace, meta_raw, segments,
                               raw_boundaries=raw_boundaries, grid=grid, n_ads=4)
    log.info("Wrote metadata.json to %s", out_path)
    log.info("Phase 3 done.")
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m backend.pipeline.fusion.run",
        description="Phase 3: fuse Phase 2 features into metadata.json.",
    )
    p.add_argument("video", type=Path, help="Path to the original MP4 (used to find workspace)")
    p.add_argument("--workspace", type=Path, default=None,
                   help="Workspace root directory (default: <repo>/workspace)")
    args = p.parse_args(argv)

    if args.workspace is None:
        args.workspace = Path(__file__).resolve().parents[3] / "workspace"
    
    ws = Workspace.for_video(args.video, root=args.workspace)
    if not ws.audio_features_path.exists():
        print(f"ERROR: Phase 2 audio features not found at {ws.audio_features_path}", file=sys.stderr)
        print("Run Phase 1 + Phase 2 first.", file=sys.stderr)
        return 1
    
    run_fusion(ws)
    return 0


if __name__ == "__main__":
    sys.exit(main())