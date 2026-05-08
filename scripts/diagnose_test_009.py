"""
scripts/diagnose_test_009.py

Read-only diagnostic for test_009 (vlog with sparse dialogue + heavy music
B-roll). Determines whether the high error rate is driven by a single signal
(audio drift) or by multiple rules collectively misfiring.

Outputs:
  docs/diagnostics/2026-05-07-test_009-diagnosis.md   (human-readable)
  docs/diagnostics/2026-05-07-test_009-diagnosis.json (raw numbers)

Does NOT modify any backend code or rerun the pipeline. Reuses
align.build_per_second_grid to recover audio_block_drift / style_block_drift
from Phase-2 features.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Force stdout to UTF-8 on Windows so unicode chars in keywords don't crash.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.pipeline.workspace import Workspace
from backend.pipeline.fusion.align import load_phase2_features, build_per_second_grid

VIDEO_ID = "test_009"
GT_PATH = ROOT / "ground_truth" / f"{VIDEO_ID}.json"
WORKSPACE = ROOT / "workspace" / VIDEO_ID
METADATA_PATH = WORKSPACE / "metadata.json"
OUT_DIR = ROOT / "docs" / "diagnostics"
REPORT_PATH = OUT_DIR / "2026-05-07-test_009-diagnosis.md"
DUMP_PATH = OUT_DIR / "2026-05-07-test_009-diagnosis.json"


# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #

def load_ground_truth() -> List[Dict]:
    with open(GT_PATH, encoding="utf-8") as f:
        gt = json.load(f)
    return gt["timeline_segments"]


def load_metadata() -> Tuple[List[Dict], float]:
    with open(METADATA_PATH, encoding="utf-8") as f:
        m = json.load(f)
    return m["segments"], m["video_info"]["duration_sec"]


def build_grid_for_video() -> Dict[str, np.ndarray]:
    video_path = ROOT / "videos" / f"{VIDEO_ID}.mp4"
    ws = Workspace.for_video(video_path, ROOT / "workspace")
    visual, audio, text = load_phase2_features(ws)
    with open(METADATA_PATH, encoding="utf-8") as f:
        duration = json.load(f)["video_info"]["duration_sec"]
    grid = build_per_second_grid(visual, audio, text, duration)
    return grid, text


# --------------------------------------------------------------------------- #
# Per-second timelines                                                        #
# --------------------------------------------------------------------------- #

def build_gt_timeline(timeline_segments: List[Dict], T: int) -> np.ndarray:
    """Per-second GT label: 'ad' / 'intro' / 'outro' / 'content' / 'unknown'."""
    out = np.array(["content"] * T, dtype=object)
    for seg in timeline_segments:
        s = int(np.floor(seg["final_video_start_seconds"]))
        e = int(np.ceil(seg["final_video_end_seconds"]))
        e = min(e, T)
        kind = seg["type"]
        if kind == "ad":
            out[s:e] = "ad"
        elif kind == "intro":
            out[s:e] = "intro"
        elif kind == "outro":
            out[s:e] = "outro"
        elif kind == "video_content":
            out[s:e] = "content"
    return out


def build_pred_timeline(segments: List[Dict], T: int) -> Tuple[np.ndarray, List[List[str]]]:
    """Per-second predicted label and per-second list of triggered rule names."""
    pred = np.array(["unlabeled"] * T, dtype=object)
    rules: List[List[str]] = [[] for _ in range(T)]
    for seg in segments:
        s = int(np.floor(seg["start_sec"]))
        e = int(np.ceil(seg["end_sec"]))
        e = min(e, T)
        pred[s:e] = seg["label"]
        triggered = seg.get("evidence", {}).get("triggered_rules", []) or []
        for t in range(s, e):
            rules[t] = list(triggered)
    return pred, rules


# --------------------------------------------------------------------------- #
# Q1: rule attribution                                                        #
# --------------------------------------------------------------------------- #

def q1_rule_attribution(gt: np.ndarray, pred: np.ndarray, rules: List[List[str]]) -> Dict:
    """Per-rule contribution to FP/FN.

    For each distinct rule name in the metadata, build a mask of seconds where
    that rule fired. Cross with GT to attribute FP (rule fired on non-ad) and
    'rescue rate' (rule fired on a true ad).

    Also compute the global FN_set = seconds where GT=ad but pred != sponsorship,
    and report which rules (if any) fired there. Most interesting case: the
    kelloggs miss, where we expect *no* rule to have fired.
    """
    T = len(gt)
    is_ad = (gt == "ad")
    is_sponsorship_pred = (pred == "sponsorship")

    rule_names = sorted({r for rs in rules for r in rs})

    rule_stats = {}
    for name in rule_names:
        mask = np.array([name in rules[t] for t in range(T)], dtype=bool)
        rule_fp = int(np.sum(mask & ~is_ad))                 # GT != ad, rule fired
        rule_tp = int(np.sum(mask & is_ad))                  # GT == ad, rule fired
        rule_fired_total = int(np.sum(mask))
        # FP regions: contiguous runs of mask & ~is_ad
        fp_runs = _find_runs_bool(mask & ~is_ad)
        fp_top = sorted(fp_runs, key=lambda r: r[1] - r[0], reverse=True)[:3]
        rule_stats[name] = {
            "fired_total_sec": rule_fired_total,
            "tp_sec": rule_tp,
            "fp_sec": rule_fp,
            "top_fp_runs": [{"start": s, "end": e, "len": e - s} for s, e in fp_top],
        }

    # Global FP/FN at the predicted-label level (sponsorship vs ad)
    fp_pred_sec = int(np.sum(is_sponsorship_pred & ~is_ad))
    fn_pred_sec = int(np.sum(is_ad & ~is_sponsorship_pred))
    tp_pred_sec = int(np.sum(is_ad & is_sponsorship_pred))
    gt_ad_total = int(np.sum(is_ad))
    pred_sponsorship_total = int(np.sum(is_sponsorship_pred))

    # FN attribution: of the GT-ad seconds we missed, what got fired?
    fn_mask = is_ad & ~is_sponsorship_pred
    fn_rule_counts: Dict[str, int] = {}
    for t in np.where(fn_mask)[0]:
        for r in rules[t]:
            fn_rule_counts[r] = fn_rule_counts.get(r, 0) + 1

    return {
        "rule_stats": rule_stats,
        "global": {
            "gt_ad_total_sec": gt_ad_total,
            "pred_sponsorship_total_sec": pred_sponsorship_total,
            "tp_sec": tp_pred_sec,
            "fp_sec": fp_pred_sec,
            "fn_sec": fn_pred_sec,
            "precision": tp_pred_sec / max(1, pred_sponsorship_total),
            "recall": tp_pred_sec / max(1, gt_ad_total),
        },
        "fn_rule_counts": fn_rule_counts,
    }


def _find_runs_bool(mask: np.ndarray) -> List[Tuple[int, int]]:
    runs = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


# --------------------------------------------------------------------------- #
# Q2: signal distributions over four populations                              #
# --------------------------------------------------------------------------- #

def q2_drift_distribution(grid: Dict[str, np.ndarray], gt: np.ndarray, pred: np.ndarray) -> Dict:
    """Compare audio_block_drift / style_block_drift / rms / is_speech across:
        A: TP   (gt=ad,      pred=sponsorship)
        B: FN   (gt=ad,      pred!=sponsorship)
        C: FP   (gt!=ad,     pred=sponsorship)
        D: TN   (gt=content, pred=core_content)   [reference baseline]
    """
    is_ad = (gt == "ad")
    is_content = (gt == "content")
    is_sp = (pred == "sponsorship")
    is_core = (pred == "core_content")

    masks = {
        "A_TP_ad_correct": is_ad & is_sp,
        "B_FN_ad_missed": is_ad & ~is_sp,
        "C_FP_content_painted_sp": is_content & is_sp,
        "D_TN_content_correct": is_content & is_core,
    }

    cols = ["audio_block_drift", "style_block_drift", "rms_energy", "is_speech"]
    out = {}
    for name, m in masks.items():
        if not m.any():
            out[name] = {"n_sec": 0}
            continue
        row = {"n_sec": int(m.sum())}
        for c in cols:
            arr = grid[c][m].astype(np.float64)
            row[c] = {
                "p15": float(np.percentile(arr, 15)),
                "p50": float(np.percentile(arr, 50)),
                "p85": float(np.percentile(arr, 85)),
                "mean": float(arr.mean()),
            }
        out[name] = row
    return out


# --------------------------------------------------------------------------- #
# Q3: kelloggs per-second table (60–90s)                                      #
# --------------------------------------------------------------------------- #

def q3_kelloggs_table(grid: Dict[str, np.ndarray], gt: np.ndarray, pred: np.ndarray,
                     rules: List[List[str]], text_features) -> List[Dict]:
    """Per-second snapshot of every relevant signal between 60 and 90 inclusive."""
    rows = []
    # Build a per-second transcript text lookup.
    transcript_at = [""] * len(gt)
    for seg in text_features.segments:
        s = int(np.floor(seg.start))
        e = int(np.ceil(seg.end))
        for t in range(s, min(e, len(gt))):
            transcript_at[t] = seg.text[:60]

    ocr_cols = ["has_url", "has_price", "has_phone", "has_cta", "has_brand_lockup"]
    for t in range(58, 93):  # 2s padding either side
        ocr_flags = []
        for c in ocr_cols:
            if c in grid and grid[c][t]:
                ocr_flags.append(c.replace("has_", ""))
        rows.append({
            "t": t,
            "gt": gt[t],
            "pred": pred[t],
            "rules": rules[t],
            "is_speech": int(grid["is_speech"][t]),
            "rms": round(float(grid["rms_energy"][t]), 4),
            "bw": round(float(grid["spectral_bandwidth"][t]), 1),
            "audio_drift": round(float(grid["audio_block_drift"][t]), 3),
            "visual_drift": round(float(grid["style_block_drift"][t]), 3),
            "hard_cut": int(grid["is_hard_cut"][t]),
            "ocr_comm": ",".join(ocr_flags),
            "transcript": transcript_at[t],
        })
    return rows


# --------------------------------------------------------------------------- #
# Q4: paired region comparisons                                               #
# --------------------------------------------------------------------------- #

def q4_region_compare(grid: Dict[str, np.ndarray]) -> Dict:
    """Aggregate stats over four regions for paired comparison."""
    regions = {
        "ramp_ad_TP_330_361":          (330, 361),  # GT ad, partially captured
        "post_ramp_FP_363_456":        (363, 456),  # GT content, painted sponsorship
        "kelloggs_ad_FN_60_90":        (60, 90),    # GT ad, completely missed
        "post_kelloggs_FP_90_200":     (90, 200),   # GT content, painted sponsorship
    }
    cols = ["audio_block_drift", "style_block_drift", "rms_energy",
            "is_speech", "is_hard_cut",
            "has_url", "has_price", "has_phone", "has_cta", "has_brand_lockup"]
    out = {}
    for name, (s, e) in regions.items():
        e = min(e, len(grid["is_speech"]))
        row = {"start": s, "end": e, "len_sec": e - s}
        for c in cols:
            if c not in grid:
                continue
            arr = grid[c][s:e].astype(np.float64)
            if c.startswith("is_") or c.startswith("has_"):
                row[c] = {"sum": int(arr.sum()), "frac": float(arr.mean())}
            else:
                row[c] = {
                    "p15": float(np.percentile(arr, 15)),
                    "p50": float(np.percentile(arr, 50)),
                    "p85": float(np.percentile(arr, 85)),
                    "mean": float(arr.mean()),
                }
        out[name] = row
    return out


# --------------------------------------------------------------------------- #
# Q5: "Music" placeholder vs sponsor_keyword chain                            #
# --------------------------------------------------------------------------- #

def q5_music_keyword(text_features) -> Dict:
    music_segs = []
    sponsor_keyword_hits = 0
    music_with_sponsor = 0
    music_total_sec = 0.0
    for seg in text_features.segments:
        text_norm = (seg.text or "").strip().lower()
        if text_norm in {"music", "[music]", "(music)"} or text_norm.startswith("music"):
            dur = max(0.0, seg.end - seg.start)
            music_total_sec += dur
            sponsor_kw = [k for k in seg.matched_keywords if k.startswith("sponsor:")]
            if sponsor_kw:
                music_with_sponsor += 1
            music_segs.append({
                "id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "matched_keywords": list(seg.matched_keywords),
            })
        if any(k.startswith("sponsor:") for k in seg.matched_keywords):
            sponsor_keyword_hits += 1
    return {
        "n_music_segments": len(music_segs),
        "music_total_sec": music_total_sec,
        "music_with_sponsor_match": music_with_sponsor,
        "global_sponsor_keyword_hits": sponsor_keyword_hits,
        "music_segments_sample": music_segs[:8],
    }


# --------------------------------------------------------------------------- #
# Report rendering                                                            #
# --------------------------------------------------------------------------- #

def render_markdown(q1, q2, q3, q4, q5) -> str:
    lines = []
    lines.append(f"# test_009 fusion diagnosis (2026-05-07)\n")
    lines.append("Read-only diagnostic. Determines whether the high error rate on test_009 "
                 "(vlog with music B-roll) stems from one dominant signal or several rules "
                 "collectively misfiring. No backend code modified; grid recomputed from "
                 "phase-2 features via `align.build_per_second_grid`.\n")

    g = q1["global"]
    lines.append("## Headline\n")
    lines.append(f"- GT ad seconds: **{g['gt_ad_total_sec']}**")
    lines.append(f"- Predicted sponsorship seconds: **{g['pred_sponsorship_total_sec']}**")
    lines.append(f"- TP={g['tp_sec']}  FP={g['fp_sec']}  FN={g['fn_sec']}")
    lines.append(f"- Precision={g['precision']:.3f}  Recall={g['recall']:.3f}\n")

    # Q1
    lines.append("## Q1 — Rule attribution\n")
    lines.append("| rule | fired_total | TP | FP | top FP runs |")
    lines.append("|---|---:|---:|---:|---|")
    rule_stats = q1["rule_stats"]
    rows_sorted = sorted(rule_stats.items(), key=lambda kv: kv[1]["fp_sec"], reverse=True)
    total_rule_fp = sum(s["fp_sec"] for s in rule_stats.values())
    for name, s in rows_sorted:
        runs = ", ".join(f"{r['start']}–{r['end']}({r['len']}s)" for r in s["top_fp_runs"])
        lines.append(f"| {name} | {s['fired_total_sec']} | {s['tp_sec']} | {s['fp_sec']} | {runs} |")
    lines.append("")
    lines.append(f"FN attribution (GT=ad but predicted!=sponsorship — {g['fn_sec']} sec total):")
    if q1["fn_rule_counts"]:
        for r, c in sorted(q1["fn_rule_counts"].items(), key=lambda kv: -kv[1]):
            lines.append(f"- {r}: {c} sec fired inside FN region")
    else:
        lines.append("- *no rule fired on any GT-ad second that we missed* — i.e. the missed "
                     "ads (kelloggs) were silent at the rule layer.")
    lines.append("")

    # Q1 conclusion logic
    if total_rule_fp > 0:
        top_name, top_stat = rows_sorted[0]
        share = top_stat["fp_sec"] / total_rule_fp
        verdict = "single-factor" if share >= 0.7 else "multi-factor"
        lines.append(f"**Verdict (mechanical):** top rule `{top_name}` accounts for "
                     f"{share*100:.0f}% of rule-attributed FP-seconds → **{verdict}**.\n")
    else:
        lines.append("**Verdict:** no FP detected (unlikely — check timeline).\n")

    # Q2
    lines.append("## Q2 — Signal distribution across populations\n")
    lines.append("Per-second percentiles of drift / energy / speech over four mutually "
                 "exclusive groups.\n")
    lines.append("| group | n_sec | audio_drift p15/p50/p85 | visual_drift p15/p50/p85 | rms p50 | speech_frac |")
    lines.append("|---|---:|---|---|---:|---:|")
    for name in ["A_TP_ad_correct", "B_FN_ad_missed", "C_FP_content_painted_sp", "D_TN_content_correct"]:
        row = q2[name]
        if row["n_sec"] == 0:
            lines.append(f"| {name} | 0 | — | — | — | — |")
            continue
        ad = row["audio_block_drift"]
        vd = row["style_block_drift"]
        lines.append(
            f"| {name} | {row['n_sec']} | "
            f"{ad['p15']:.2f}/{ad['p50']:.2f}/{ad['p85']:.2f} | "
            f"{vd['p15']:.2f}/{vd['p50']:.2f}/{vd['p85']:.2f} | "
            f"{row['rms_energy']['p50']:.3f} | "
            f"{row['is_speech']['mean']:.2f} |"
        )
    lines.append("")

    # Q3
    lines.append("## Q3 — Kelloggs ad per-second snapshot (58–92s)\n")
    lines.append("GT ad is 60.0–90.016s. Looking for the signals that should but don't fire.\n")
    lines.append("| t | gt | pred | rules | spk | rms | audio_drift | vis_drift | hard_cut | OCR | transcript |")
    lines.append("|---:|---|---|---|---:|---:|---:|---:|---:|---|---|")
    for r in q3:
        rules_str = ",".join(r["rules"]) if r["rules"] else "—"
        lines.append(
            f"| {r['t']} | {r['gt']} | {r['pred']} | {rules_str} | "
            f"{r['is_speech']} | {r['rms']:.3f} | "
            f"{r['audio_drift']:.2f} | {r['visual_drift']:.2f} | "
            f"{r['hard_cut']} | {r['ocr_comm'] or '—'} | "
            f"{r['transcript'] or '—'} |"
        )
    lines.append("")

    # Q4
    lines.append("## Q4 — Paired region compare\n")
    lines.append("True ad regions vs adjacent FP regions. Same column = same statistic; "
                 "compare row-pairs to see whether GT-ad and FP-content separate in any "
                 "feature dimension.\n")
    region_order = [
        "ramp_ad_TP_330_361", "post_ramp_FP_363_456",
        "kelloggs_ad_FN_60_90", "post_kelloggs_FP_90_200",
    ]
    lines.append("| region | len | audio_drift mean | visual_drift mean | rms p50 | speech_frac | hard_cut/sec | OCR_comm hits |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name in region_order:
        r = q4[name]
        ad = r["audio_block_drift"]["mean"]
        vd = r["style_block_drift"]["mean"]
        rms = r["rms_energy"]["p50"]
        spk = r["is_speech"]["frac"]
        cut_rate = r["is_hard_cut"]["sum"] / max(1, r["len_sec"])
        ocr_total = sum(r[c]["sum"] for c in ["has_url", "has_price", "has_phone", "has_cta", "has_brand_lockup"])
        lines.append(
            f"| {name} | {r['len_sec']} | {ad:.2f} | {vd:.2f} | "
            f"{rms:.3f} | {spk:.2f} | {cut_rate:.2f} | {ocr_total} |"
        )
    lines.append("")

    # Q5
    lines.append("## Q5 — \"Music\" transcript placeholder vs sponsor_keyword\n")
    lines.append(f"- Segments whose text starts with \"Music\": **{q5['n_music_segments']}** "
                 f"({q5['music_total_sec']:.0f} sec total)")
    lines.append(f"- Of those, segments with a `sponsor:*` matched keyword: "
                 f"**{q5['music_with_sponsor_match']}**")
    lines.append(f"- Total `sponsor:*` keyword hits across the entire transcript: "
                 f"**{q5['global_sponsor_keyword_hits']}**")
    if q5["music_with_sponsor_match"] == 0 and q5["global_sponsor_keyword_hits"] == 0:
        lines.append("\n**Verdict:** the \"Music\" placeholder does NOT feed `rule_sponsor_keyword` "
                     "in this video. The high text_score on FP segments must come from another "
                     "channel (probably the keyword-free 'has_text' axis or a different rule).\n")
    elif q5["music_with_sponsor_match"] > 0:
        lines.append("\n**Verdict:** \"Music\" segments ARE matching sponsor keywords — fix the "
                     "keyword table or skip placeholder transcripts.\n")
    else:
        lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/6] loading inputs ...")
    timeline = load_ground_truth()
    segments, duration = load_metadata()

    print("[2/6] building per-second grid (recomputes block_drift) ...")
    grid, text_features = build_grid_for_video()
    T = len(grid["is_speech"])

    print("[3/6] building timelines ...")
    gt = build_gt_timeline(timeline, T)
    pred, rules = build_pred_timeline(segments, T)

    print("[4/6] running Q1 rule attribution ...")
    q1 = q1_rule_attribution(gt, pred, rules)

    print("[5/6] running Q2-Q5 ...")
    q2 = q2_drift_distribution(grid, gt, pred)
    q3 = q3_kelloggs_table(grid, gt, pred, rules, text_features)
    q4 = q4_region_compare(grid)
    q5 = q5_music_keyword(text_features)

    print("[6/6] writing report ...")
    md = render_markdown(q1, q2, q3, q4, q5)
    REPORT_PATH.write_text(md, encoding="utf-8")
    DUMP_PATH.write_text(json.dumps({
        "q1": q1, "q2": q2, "q3": q3, "q4": q4, "q5": q5,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nReport: {REPORT_PATH}")
    print(f"Data:   {DUMP_PATH}")


if __name__ == "__main__":
    main()
