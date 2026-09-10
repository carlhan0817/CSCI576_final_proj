from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Import keyword groups from the rules module (single source of truth).
# text.py detects matches; rules.py still uses the same lists for rule-hit
# intervals — they no longer re-scan the text, they read matched_keywords.
from backend.pipeline.fusion.rules import (
    INTRO_KEYWORDS,
    OUTRO_KEYWORDS,
    RECAP_KEYWORDS,
    SELF_PROMO_KEYWORDS,
    SPONSOR_KEYWORDS,
)
from backend.pipeline.logging_setup import get_logger
from backend.pipeline.schemas import TextFeatures, TextFeatureSegment, Transcript
from backend.pipeline.workspace import Workspace

DEFAULT_SIMILARITY_THRESHOLD = 0.35

_KEYWORD_GROUPS = {
    "sponsor": SPONSOR_KEYWORDS,
    "intro": INTRO_KEYWORDS,
    "outro": OUTRO_KEYWORDS,
    "self_promo": SELF_PROMO_KEYWORDS,
    "recap": RECAP_KEYWORDS,
}


def _match_keywords(text: str) -> list[str]:
    """
    Return a list of '<group>:<keyword>' strings for every keyword group that
    matches anywhere in `text` (case-insensitive). At most one match per group.
    """
    text_lower = text.lower()
    matched = []
    for group, keywords in _KEYWORD_GROUPS.items():
        for kw in keywords:
            if kw in text_lower:
                matched.append(f"{group}:{kw}")
                break  # one hit per group is enough
    return matched


def _load_minilm_model(device: str = "cpu") -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2", device=device)


def extract_text_features(
    workspace: Workspace,
    device: str = "cpu",
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> Path:
    """
    Reads transcript.json, generates MiniLM embeddings, computes per-sentence
    cosine similarity to detect topic boundaries, matches keyword groups, and
    writes:
      - features/text_features.json   (TextFeatures schema)
      - features/text_embeddings.npy  (float32 array, shape [N, 384])
    """
    log = get_logger(workspace.log_path)
    out_path = workspace.text_features_path
    embeddings_path = workspace.text_embeddings_path

    if out_path.exists() and embeddings_path.exists():
        log.info("Text features already exist. Skipping.")
        return out_path

    log.info("Stage 2 (Text): Loading transcript...")
    with open(workspace.transcript_path, "r", encoding="utf-8") as f:
        transcript = Transcript.model_validate(json.load(f))

    if not transcript.segments:
        log.warning("Transcript is empty. Writing empty text features.")
        empty = TextFeatures(segments=[])
        out_path.write_text(empty.model_dump_json(indent=2))
        np.save(str(embeddings_path), np.empty((0, 384), dtype=np.float32))
        return out_path

    # ── Encode all sentences ──────────────────────────────────────────────────
    log.info("Stage 2 (Text): Encoding %d segments with MiniLM...", len(transcript.segments))
    model = _load_minilm_model(device)
    sentences = [seg.text for seg in transcript.segments]
    embeddings: np.ndarray = model.encode(sentences, convert_to_numpy=True)  # (N, 384)

    # Persist embeddings for supervised classifier and cross-episode matching
    np.save(str(embeddings_path), embeddings.astype(np.float32))
    log.info("Saved MiniLM embeddings to %s", embeddings_path.name)

    # ── Per-sentence features ─────────────────────────────────────────────────
    text_feature_segments = []

    for i, seg in enumerate(transcript.segments):
        sim_to_next = None
        is_boundary = False

        if i < len(transcript.segments) - 1:
            vec_curr = embeddings[i].reshape(1, -1)
            vec_next = embeddings[i + 1].reshape(1, -1)
            sim_to_next = float(cosine_similarity(vec_curr, vec_next)[0][0])
            is_boundary = sim_to_next < threshold

        matched_kws = _match_keywords(seg.text)

        text_feature_segments.append(
            TextFeatureSegment(
                id=seg.id,
                start=seg.start,
                end=seg.end,
                text=seg.text,
                similarity_to_next=sim_to_next,
                is_potential_boundary=is_boundary,
                matched_keywords=matched_kws,
            )
        )

    result = TextFeatures(segments=text_feature_segments)
    out_path.write_text(result.model_dump_json(indent=2))
    log.info("Text features saved to %s (%d segments).", out_path.name, len(text_feature_segments))
    return out_path
