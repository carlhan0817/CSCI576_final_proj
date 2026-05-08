"""Re-run Phase 3 fusion on every workspace under workspace——ocr/.

Phase 1/2 features already exist. Only fusion needs to re-run after the
export.py ad-end-detection improvements (Fixes A/B/C/D from 5_7_improvement.md).
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.pipeline.workspace import Workspace
from backend.pipeline.fusion.run import run_fusion


def main() -> int:
    ws_root = REPO / "workspace——ocr"
    if not ws_root.exists():
        print(f"ERROR: workspace root {ws_root} not found", file=sys.stderr)
        return 1

    stems = sorted(
        p.name for p in ws_root.iterdir()
        if p.is_dir() and (p / "meta_raw.json").exists()
    )
    print(f"Workspaces found: {stems}")

    for stem in stems:
        fake_video = ws_root.parent / "videos" / f"{stem}.mp4"
        ws = Workspace.for_video(fake_video, root=ws_root)
        t0 = time.time()
        print(f"\n=== {stem} ===")
        try:
            run_fusion(ws)
            print(f"  fusion OK in {time.time() - t0:.1f}s")
        except Exception as e:
            print(f"  fusion FAILED: {e!r}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
