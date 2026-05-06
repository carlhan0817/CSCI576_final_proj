"""
Phase 3 Step 1: Time-grid alignment.

Resamples the three modality features (visual @ 1 FPS, audio @ Silero VAD intervals,
text @ sentence-level) onto a unified per-second grid. Output is a numpy array where
each row = one second, columns = aggregated features from all three modalities.
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from backend.pipeline.schemas import (
    AudioFeatures,
    VisualFeatures,
    TextFeatures,
)
from backend.pipeline.workspace import Workspace


def load_phase2_features(workspace: Workspace) -> Tuple[VisualFeatures, AudioFeatures, TextFeatures]:
    """Read the three Phase 2 JSON artifacts."""
    with open(workspace.visual_features_path, encoding="utf-8") as f:
        visual = VisualFeatures.model_validate(json.load(f))
    with open(workspace.audio_features_path, encoding="utf-8") as f:
        audio = AudioFeatures.model_validate(json.load(f))
    with open(workspace.text_features_path, encoding="utf-8") as f:
        text = TextFeatures.model_validate(json.load(f))
    return visual, audio, text


def build_per_second_grid(
    visual: VisualFeatures,
    audio: AudioFeatures,
    text: TextFeatures,
    duration_sec: float,
) -> Dict[str, np.ndarray]:
    """
    Build a per-second feature grid covering [0, duration_sec).
    
    Returns a dict of numpy arrays, all of length T = ceil(duration_sec):
      - is_speech[t]         : 1 if second t is inside any speech VAD segment
      - hist_diff[t]          : visual histogram diff at second t (0.0 if no frame)
      - is_hard_cut[t]        : 1 if visual hard cut at second t
      - clip_<label>[t]       : CLIP zero-shot probability for each scene label
      - text_sim_to_next[t]   : text cosine sim to next sentence (NaN if no sentence here)
      - has_text[t]           : 1 if any transcript segment overlaps second t
      - text_sentence_id[t]   : id of transcript segment overlapping second t (-1 if none)
    """
    T = int(np.ceil(duration_sec))
    grid: Dict[str, np.ndarray] = {}

    # --- Audio: per-second grid (one segment per second in new schema)
    is_speech = np.zeros(T, dtype=np.int8)
    rms_energy = np.zeros(T, dtype=np.float32)
    spectral_centroid = np.zeros(T, dtype=np.float32)
    spectral_bandwidth = np.zeros(T, dtype=np.float32)
    zero_crossing_rate = np.zeros(T, dtype=np.float32)
    spectral_entropy = np.zeros(T, dtype=np.float32)
    is_music = np.zeros(T, dtype=np.int8)

    for seg in audio.segments:
        t = int(np.floor(seg.start))
        if t >= T:
            continue
        is_speech[t] = int(seg.is_speech)
        rms_energy[t] = seg.rms_energy
        spectral_centroid[t] = seg.spectral_centroid
        spectral_bandwidth[t] = seg.spectral_bandwidth
        zero_crossing_rate[t] = seg.zero_crossing_rate
        spectral_entropy[t] = seg.spectral_entropy
        is_music[t] = int(seg.audio_class == "music")

    grid["is_speech"] = is_speech
    grid["rms_energy"] = rms_energy
    grid["spectral_centroid"] = spectral_centroid
    grid["spectral_bandwidth"] = spectral_bandwidth
    grid["zero_crossing_rate"] = zero_crossing_rate
    grid["spectral_entropy"] = spectral_entropy
    grid["is_music"] = is_music

    # ── Audio: MFCC matrix + per-second audio style drift ─────────────────────
    mfcc_dim = (
        len(audio.segments[0].mfcc) if audio.segments and audio.segments[0].mfcc else 0
    )
    if mfcc_dim > 0:
        mfcc_matrix = np.zeros((T, mfcc_dim), dtype=np.float32)
        for seg in audio.segments:
            t = int(np.floor(seg.start))
            if t >= T or not seg.mfcc:
                continue
            mfcc_matrix[t] = np.array(seg.mfcc, dtype=np.float32)
        from backend.pipeline.fusion.style_drift import compute_drift
        grid["audio_drift"] = compute_drift(mfcc_matrix, window_sec=15)
    else:
        grid["audio_drift"] = np.zeros(T, dtype=np.float32)

    # --- Visual: 1 FPS sampling → already per-second
    hist_diff = np.zeros(T, dtype=np.float32)
    is_hard_cut = np.zeros(T, dtype=np.int8)
    
    # Collect all CLIP labels (assumes all frames share the same label set)
    clip_labels = list(visual.frames[0].clip_labels.keys()) if visual.frames else []
    clip_per_label: Dict[str, np.ndarray] = {
        lbl: np.zeros(T, dtype=np.float32) for lbl in clip_labels
    }
    
    for f in visual.frames:
        t = int(f.timestamp_sec)
        if t >= T:
            continue
        hist_diff[t] = f.hist_diff_to_previous
        is_hard_cut[t] = 1 if f.is_hard_cut else 0
        for lbl, prob in f.clip_labels.items():
            clip_per_label[lbl][t] = prob

    grid["hist_diff"] = hist_diff
    grid["is_hard_cut"] = is_hard_cut
    for lbl, arr in clip_per_label.items():
        grid[f"clip_{lbl}"] = arr

    # ── Visual: pooled CLIP embedding + per-second style drift ────────────────
    embed_dim = (
        len(visual.frames[0].clip_embedding) if visual.frames and visual.frames[0].clip_embedding else 0
    )
    if embed_dim > 0:
        embeddings = np.zeros((T, embed_dim), dtype=np.float32)
        for f in visual.frames:
            t = int(f.timestamp_sec)
            if t >= T or not f.clip_embedding:
                continue
            embeddings[t] = np.array(f.clip_embedding, dtype=np.float32)
        from backend.pipeline.fusion.style_drift import compute_drift
        grid["style_drift"] = compute_drift(embeddings, window_sec=15)
    else:
        grid["style_drift"] = np.zeros(T, dtype=np.float32)

    # ── Visual: OCR commercial signals ───────────────────────────────────────
    has_url = np.zeros(T, dtype=np.int8)
    has_price = np.zeros(T, dtype=np.int8)
    has_phone = np.zeros(T, dtype=np.int8)
    has_cta = np.zeros(T, dtype=np.int8)
    has_brand_lockup = np.zeros(T, dtype=np.int8)

    for f in visual.frames:
        t = int(f.timestamp_sec)
        if t >= T:
            continue
        has_url[t] = int(f.has_url)
        has_price[t] = int(f.has_price)
        has_phone[t] = int(f.has_phone)
        has_cta[t] = int(f.has_cta)
        has_brand_lockup[t] = int(f.has_brand_lockup)

    grid["has_url"] = has_url
    grid["has_price"] = has_price
    grid["has_phone"] = has_phone
    grid["has_cta"] = has_cta
    grid["has_brand_lockup"] = has_brand_lockup

    # --- Text: sentence-level → per-second tags
    text_sim = np.full(T, np.nan, dtype=np.float32)
    has_text = np.zeros(T, dtype=np.int8)
    text_sentence_id = np.full(T, -1, dtype=np.int32)
    text_has_cta = np.zeros(T, dtype=np.int8)

    for seg in text.segments:
        s = max(0, int(np.floor(seg.start)))
        e = min(T, int(np.ceil(seg.end)))
        if any(kw.startswith("cta:") for kw in seg.matched_keywords):
            text_has_cta[s:e] = 1
        for t in range(s, e):
            has_text[t] = 1
            text_sentence_id[t] = seg.id
            if seg.similarity_to_next is not None:
                text_sim[t] = seg.similarity_to_next

    grid["text_sim_to_next"] = text_sim
    grid["has_text"] = has_text
    grid["text_sentence_id"] = text_sentence_id
    grid["text_has_cta"] = text_has_cta

    return grid