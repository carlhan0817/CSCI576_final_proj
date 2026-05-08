"""Compare metadata.json against ground_truth/<stem>.json.

Reports for each test video:
  1. Ad anchor accuracy: how well detected ad start/end times match ground truth.
     Tolerance buckets: ≤2s, ≤5s, ≤10s.
  2. Ad-window IoU per ground-truth ad.
  3. Per-second classification (ad / non-ad) precision/recall/F1.
  4. Detection of intro/outro segments where ground truth provides them.

Ground-truth schema notes:
  - test_009 / test_010 use `inserted_segments` and `natural_segments`.
  - test_003 / test_008 use `inserted_ads` (no intro/outro listed).
  - All use `final_video_*_seconds` for absolute timeline.
  - Field names differ: `final_video_start_seconds` vs `final_video_ad_start_seconds`.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import List, Tuple

REPO = Path(__file__).resolve().parents[1]
WS_ROOT = REPO / "workspace——ocr"
GT_ROOT = REPO / "ground_truth"


def gt_ad_windows(gt: dict) -> List[Tuple[float, float, str]]:
    """Return list of (start, end, source_filename) ad windows from ground truth."""
    windows = []
    if "inserted_segments" in gt:
        for seg in gt["inserted_segments"]:
            if seg.get("segment_type") == "ad":
                windows.append((
                    float(seg["final_video_start_seconds"]),
                    float(seg["final_video_end_seconds"]),
                    seg.get("segment_filename", "?"),
                ))
    elif "inserted_ads" in gt:
        for ad in gt["inserted_ads"]:
            windows.append((
                float(ad["final_video_ad_start_seconds"]),
                float(ad["final_video_ad_end_seconds"]),
                ad.get("ad_filename", "?"),
            ))
    return windows


def gt_natural(gt: dict) -> List[Tuple[float, float, str]]:
    """intro/outro spans from ground truth (only test_009/test_010 list them)."""
    out = []
    for nat in gt.get("natural_segments", []):
        out.append((float(nat["start"]), float(nat["end"]), nat["type"]))
    return out


def merge_ad_segments(meta: dict) -> List[Tuple[float, float]]:
    """Coalesce adjacent ad-labeled segments into contiguous ad windows."""
    ads = sorted(
        (s["start_sec"], s["end_sec"])
        for s in meta["segments"] if s["label"] == "ad"
    )
    merged: List[Tuple[float, float]] = []
    for a, b in ads:
        if merged and a <= merged[-1][1] + 1.0:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def best_match(gt_w: Tuple[float, float], pred_ws: List[Tuple[float, float]]):
    if not pred_ws:
        return None, 0.0
    best = max(pred_ws, key=lambda p: iou(gt_w, p))
    return best, iou(gt_w, best)


def per_second_classification(meta: dict, gt_ads: List[Tuple[float, float, str]],
                              total_sec: float) -> dict:
    T = int(total_sec) + 1
    pred = [0] * T
    truth = [0] * T
    for s in meta["segments"]:
        if s["label"] == "ad":
            for t in range(int(s["start_sec"]), min(int(s["end_sec"]) + 1, T)):
                pred[t] = 1
    for a, b, _ in gt_ads:
        for t in range(int(a), min(int(b) + 1, T)):
            truth[t] = 1
    tp = sum(1 for t in range(T) if pred[t] and truth[t])
    fp = sum(1 for t in range(T) if pred[t] and not truth[t])
    fn = sum(1 for t in range(T) if not pred[t] and truth[t])
    tn = sum(1 for t in range(T) if not pred[t] and not truth[t])
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": p, "recall": r, "f1": f1}


def report_one(stem: str) -> dict:
    gt_path = GT_ROOT / f"{stem}.json"
    md_path = WS_ROOT / stem / "metadata.json"
    if not gt_path.exists() or not md_path.exists():
        return {"stem": stem, "error": "missing files"}

    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    meta = json.loads(md_path.read_text(encoding="utf-8"))

    gt_ads = gt_ad_windows(gt)
    gt_nat = gt_natural(gt)
    pred_ad_windows = merge_ad_segments(meta)
    pred_anchors = [c["timestamp_sec"] for c in meta.get("ad_insertion_candidates", [])]

    total = float(meta["video_info"]["duration_sec"])
    cls = per_second_classification(meta, gt_ads, total)

    print(f"\n{'='*72}")
    print(f"  {stem}    duration={total:.1f}s   #gt_ads={len(gt_ads)}   "
          f"#pred_ad_windows={len(pred_ad_windows)}   #anchors={len(pred_anchors)}")
    print(f"{'='*72}")

    print("  Anchor-vs-GT-start (tolerance):")
    starts_gt = [w[0] for w in gt_ads]
    bucket = {"≤2s": 0, "≤5s": 0, "≤10s": 0, ">10s": 0, "missing": 0}
    for s in starts_gt:
        if not pred_anchors:
            bucket["missing"] += 1
            continue
        d = min(abs(s - a) for a in pred_anchors)
        if d <= 2: bucket["≤2s"] += 1
        elif d <= 5: bucket["≤5s"] += 1
        elif d <= 10: bucket["≤10s"] += 1
        else: bucket[">10s"] += 1
    print(f"    {bucket}")

    print("  Per-ad IoU (gt window vs best predicted ad window):")
    for (a, b, fn) in gt_ads:
        best, j = best_match((a, b), pred_ad_windows)
        bs = f"{best[0]:.1f}-{best[1]:.1f}" if best else "—"
        print(f"    gt {a:7.1f}-{b:7.1f}s  ({fn:25s})  →  pred {bs:>17s}  IoU={j:.3f}")

    if gt_nat:
        print("  Natural segments (ground truth):")
        for a, b, t in gt_nat:
            # quick check: any predicted segment overlapping has matching label?
            overlapping = [
                s for s in meta["segments"]
                if not (s["end_sec"] <= a or s["start_sec"] >= b)
            ]
            labels = sorted({s["label"] for s in overlapping})
            print(f"    gt {t:8s} {a:6.1f}-{b:6.1f}s  →  pred labels overlapping: {labels}")

    print("  Per-second AD classification:")
    print(f"    TP={cls['tp']}  FP={cls['fp']}  FN={cls['fn']}  TN={cls['tn']}")
    print(f"    precision={cls['precision']:.3f}  recall={cls['recall']:.3f}  "
          f"f1={cls['f1']:.3f}")

    return {"stem": stem, "cls": cls, "anchor_buckets": bucket,
            "gt_ads": gt_ads, "pred_ad_windows": pred_ad_windows,
            "pred_anchors": pred_anchors}


def main() -> int:
    stems = sorted(p.stem for p in GT_ROOT.glob("test_*.json"))
    stems = [s for s in stems if (WS_ROOT / s / "metadata.json").exists()]
    if not stems:
        print("No matching workspaces found.", file=sys.stderr)
        return 1

    summaries = []
    for s in stems:
        summaries.append(report_one(s))

    # Aggregate
    print(f"\n{'='*72}\n  AGGREGATE\n{'='*72}")
    print(f"  {'stem':<10} {'P':>6} {'R':>6} {'F1':>6}  anchors_within_5s/total")
    for r in summaries:
        if "error" in r:
            print(f"  {r['stem']:<10}  ERROR")
            continue
        c = r["cls"]; b = r["anchor_buckets"]
        within5 = b["≤2s"] + b["≤5s"]
        total_gt = sum(b.values())
        print(f"  {r['stem']:<10} {c['precision']:>6.3f} {c['recall']:>6.3f} "
              f"{c['f1']:>6.3f}  {within5}/{total_gt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
