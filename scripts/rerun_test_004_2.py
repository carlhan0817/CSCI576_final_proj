"""One-off: re-run Phase 2 + Phase 3 on workspace/test_004_2 with new signals.

Drops stale feature files, re-extracts visual (CLIP embedding + OCR), audio
(MFCC), text (CTA group), and re-runs fusion. Prints a compact comparison
of sponsorship segments before/after.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.pipeline.workspace import Workspace
from backend.pipeline.features.visual import extract_visual_features
from backend.pipeline.features.audio import extract_audio_features
from backend.pipeline.features.text import extract_text_features
from backend.pipeline.fusion.run import run_fusion


def main():
    ws_root = REPO / "workspace"
    video_dir = ws_root / "test_004_2"
    if not video_dir.exists():
        print(f"ERROR: {video_dir} not found", file=sys.stderr)
        return 1

    # Workspace.for_video appends the stem to root, so pass video named test_004_2.mp4
    fake_video = video_dir.parent / "test_004_2.mp4"
    ws = Workspace.for_video(fake_video, root=ws_root)
    print(f"Workspace: {ws.root}")

    # Drop stale features
    for path in (ws.visual_features_path, ws.audio_features_path, ws.text_features_path):
        if path.exists():
            print(f"  removing stale {path.name}")
            path.unlink()

    # Phase 2 — visual
    t0 = time.time()
    print("\n[Phase 2] visual extraction (CLIP + OCR + embedding) …")
    extract_visual_features(ws, device="cpu")
    print(f"  done in {time.time() - t0:.1f}s")

    # Phase 2 — audio
    t0 = time.time()
    print("\n[Phase 2] audio extraction (Silero VAD + spectral + MFCC) …")
    extract_audio_features(ws, device="cpu")
    print(f"  done in {time.time() - t0:.1f}s")

    # Phase 2 — text
    t0 = time.time()
    print("\n[Phase 2] text extraction (sentence-BERT + keyword groups incl. CTA) …")
    extract_text_features(ws, device="cpu")
    print(f"  done in {time.time() - t0:.1f}s")

    # Phase 3 fusion
    t0 = time.time()
    print("\n[Phase 3] fusion …")
    run_fusion(ws)
    print(f"  done in {time.time() - t0:.1f}s")

    # Inspect sponsorship segments
    meta = json.loads((ws.root / "metadata.json").read_text())
    segs = [s for s in meta["segments"] if s["label"] == "sponsorship"]
    print(f"\nSponsorship segments: {len(segs)}")
    for s in segs:
        triggers = s.get("evidence", {}).get("triggered_rules", [])
        print(
            f"  seg #{s['segment_id']:>2} {s['start_sec']:>5.0f}-{s['end_sec']:>5.0f}s  "
            f"conf={s['confidence']:.2f}  rules={triggers}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
