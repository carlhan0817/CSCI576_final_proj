"""
Phase 4: Metadata Generation & Persistence.

Per PDD V2.1 §4.4, this stage's five sub-tasks are:

1. **Schema validation** — every field is validated by the Pydantic V2 models
    defined in ``backend.pipeline.schemas`` (the ``Metadata`` model with
    ``extra="forbid"`` rejects malformed data). We surface validation errors
    explicitly so a misconfigured upstream stage fails loudly rather than
    silently producing a broken JSON.

2. **Chapter generation** — adjacent ``core_content`` segments are merged into
    chapter ranges. The chapter title is *not* a substring of the first
    sentence; instead we pick the **most semantically representative sentence**
    inside the chapter using MiniLM embeddings (already cached at
    ``features/text_embeddings.npy`` from Phase 2). If embeddings are missing
    we degrade gracefully to a longest-sentence heuristic.

3. **Skip suggestions** — every segment whose ``label != "core_content"`` is appended to ``skip_suggestions[]``.

4. **Per-segment summary** — each segment gets a one-sentence summary picked
    the same way as chapter titles (representative sentence by embedding
    similarity to the segment's centroid). This is computed *here*, not in
    ``classify.py``, because it is part of the export contract.

5. **Version stamping** — ``video_info.analysis_version`` is set from the
    module-level ``ANALYSIS_VERSION`` constant. Bump this whenever the export
    contract changes so the front-end can detect stale metadata.

The output is a single ``metadata.json`` written under the workspace root and
already conforming to ``§2.2`` of the PDD.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from pydantic import ValidationError

from backend.pipeline.schemas import (
    MetaRaw,
    Metadata,
    VideoInfo,
    Segment,
    Chapter,
    Transcript,
    TranscriptSegment,
)
from backend.pipeline.workspace import Workspace


# Bumped to 0.4.0 for the Phase 4 export contract:
# - real semantic chapter titles
# - representative-sentence summaries
# - explicit pydantic validation step
ANALYSIS_VERSION = "0.4.0"


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-task 4: representative-sentence picker (used by both summary and chapter)
# ---------------------------------------------------------------------------

def _sentences_in_window(
    transcript_segments: Sequence[TranscriptSegment],
    start_sec: float,
    end_sec: float,
) -> List[TranscriptSegment]:
    """
    Return every transcript sentence whose [start, end] interval intersects
    [start_sec, end_sec). We use intersection rather than full containment so
    sentences spanning the boundary aren't dropped.
    """
    out: List[TranscriptSegment] = []
    for s in transcript_segments:
        if s.end <= start_sec or s.start >= end_sec:
            continue
        out.append(s)
    return out


def _pick_representative_sentence(
    sentences: Sequence[TranscriptSegment],
    embeddings: Optional[np.ndarray],
    max_chars: int = 200,
) -> str:
    """
    Pick the sentence that best represents this group.

    Strategy:
    1. If embeddings are available, compute the centroid of the group's
        embeddings and return the sentence whose embedding has the highest
        cosine similarity to the centroid. This is a tiny extractive summary —
        the sentence "in the middle" of the semantic cloud.
    2. Otherwise (e.g. embeddings file missing), fall back to the longest
        sentence — a reasonable proxy for "informative" and what the original
        classify.py used.

    Returns "" if there are no sentences. The result is stripped and truncated
    to ``max_chars``.
    """
    if not sentences:
        return ""

    if embeddings is not None and embeddings.shape[0] > 0:
        ids = [s.id for s in sentences if 0 <= s.id < embeddings.shape[0]]
        if ids:
            sub = embeddings[ids]  # (k, d)
            # L2-normalise for cosine similarity
            norms = np.linalg.norm(sub, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            unit = sub / norms
            centroid = unit.mean(axis=0)
            cnorm = np.linalg.norm(centroid)
            if cnorm > 0:
                centroid = centroid / cnorm
                sims = unit @ centroid  # (k,)
                best_local = int(np.argmax(sims))
                best_id = ids[best_local]
                for s in sentences:
                    if s.id == best_id:
                        return s.text.strip()[:max_chars]

    # Fallback: longest sentence.
    longest = max(sentences, key=lambda s: len(s.text))
    return longest.text.strip()[:max_chars]


def _load_embeddings(workspace: Workspace) -> Optional[np.ndarray]:
    """
    Try to load the cached MiniLM sentence embeddings from Phase 2.
    Returns None if the file is missing or unreadable so the caller can fall
    back to the heuristic.
    """
    path = workspace.text_embeddings_path
    if not path.exists():
        log.warning("Phase 2 embeddings missing at %s; using fallback summarisation.", path)
        return None
    try:
        emb = np.load(str(path))
        if emb.ndim != 2:
            log.warning("Embeddings have unexpected shape %s; ignoring.", emb.shape)
            return None
        return emb
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("Failed to load embeddings (%s); using fallback.", exc)
        return None


def _load_transcript(workspace: Workspace) -> List[TranscriptSegment]:
    path = workspace.transcript_path
    if not path.exists():
        log.warning("Transcript missing at %s; segment summaries will be empty.", path)
        return []
    transcript = Transcript.model_validate(json.loads(path.read_text()))
    return list(transcript.segments)


# ---------------------------------------------------------------------------
# Sub-task 4 wrapper: regenerate per-segment summaries
# ---------------------------------------------------------------------------

def attach_summaries(
    segments: List[Segment],
    transcript_segments: Sequence[TranscriptSegment],
    embeddings: Optional[np.ndarray],
) -> List[Segment]:
    """
    Replace each segment's ``summary`` with a representative sentence picked
    via embedding-centroid similarity (or longest-sentence fallback).

    A new list is returned; the input is not mutated.
    """
    out: List[Segment] = []
    for seg in segments:
        sents = _sentences_in_window(transcript_segments, seg.start_sec, seg.end_sec)
        summary = _pick_representative_sentence(sents, embeddings)
        out.append(Segment(
            segment_id=seg.segment_id,
            start_sec=seg.start_sec,
            end_sec=seg.end_sec,
            label=seg.label,
            confidence=seg.confidence,
            evidence=seg.evidence,
            summary=summary,
            user_corrected=seg.user_corrected,
        ))
    return out


# ---------------------------------------------------------------------------
# Sub-task 2: chapters
# ---------------------------------------------------------------------------

def _make_chapter_title(
    chapter_index: int,
    sentences: Sequence[TranscriptSegment],
    embeddings: Optional[np.ndarray],
) -> str:
    """Build a human-readable chapter title."""
    if not sentences:
        return f"Chapter {chapter_index + 1}"
    rep = _pick_representative_sentence(sentences, embeddings, max_chars=80)
    if not rep:
        return f"Chapter {chapter_index + 1}"
    title = " ".join(rep.split())
    if len(title) >= 80:
        title = title[:77].rsplit(" ", 1)[0] + "..."
    return title


def build_chapters(
    segments: Sequence[Segment],
    transcript_segments: Sequence[TranscriptSegment],
    embeddings: Optional[np.ndarray],
) -> List[Chapter]:
    """
    Group consecutive ``core_content`` segments into chapters and assign each
    a representative-sentence title.

    Two adjacent core_content segments → one chapter spanning both.
    A non-core_content segment → ends the current chapter (if any).
    """
    chapters: List[Chapter] = []
    chapter_id = 0
    current_start: Optional[float] = None
    current_end: Optional[float] = None

    def _flush() -> None:
        nonlocal chapter_id, current_start, current_end
        if current_start is None or current_end is None:
            return
        sents = _sentences_in_window(transcript_segments, current_start, current_end)
        title = _make_chapter_title(chapter_id, sents, embeddings)
        chapters.append(Chapter(
            chapter_id=chapter_id,
            start_sec=current_start,
            end_sec=current_end,
            title=title,
        ))
        chapter_id += 1
        current_start = None
        current_end = None

    for seg in segments:
        if seg.label == "core_content":
            if current_start is None:
                current_start = seg.start_sec
            current_end = seg.end_sec
        else:
            _flush()

    _flush()
    return chapters


# ---------------------------------------------------------------------------
# Sub-task 3: skip suggestions
# ---------------------------------------------------------------------------

def build_skip_suggestions(segments: Sequence[Segment]) -> List[int]:
    """All non-core_content segment_ids, in order."""
    return [s.segment_id for s in segments if s.label != "core_content"]


# ---------------------------------------------------------------------------
# Sub-task 1 + 5: validate + version-stamp + write
# ---------------------------------------------------------------------------

def build_metadata(
    meta_raw: MetaRaw,
    segments: Sequence[Segment],
    transcript_segments: Sequence[TranscriptSegment],
    embeddings: Optional[np.ndarray],
    *,
    analysis_version: str = ANALYSIS_VERSION,
    verified_by_human: bool = False,
) -> Metadata:
    """
    Build the in-memory Metadata object. Useful for testing without touching
    the filesystem.

    Raises ``pydantic.ValidationError`` if any field violates the schema.
    """
    # Step (4): regenerate summaries so the export has the canonical
    # representative-sentence form, regardless of what classify.py wrote.
    segments_with_summary = attach_summaries(list(segments), transcript_segments, embeddings)

    chapters = build_chapters(segments_with_summary, transcript_segments, embeddings)
    skip_suggestions = build_skip_suggestions(segments_with_summary)

    video_info = VideoInfo(
        filename=meta_raw.filename,
        duration_sec=meta_raw.duration_sec,
        fps=meta_raw.fps,
        width=meta_raw.width,
        height=meta_raw.height,
        analysis_version=analysis_version,
        verified_by_human=verified_by_human,
    )

    # Constructing the Metadata object validates the entire structure
    # (extra="forbid", numeric bounds, label literals, …). We catch
    # ValidationError to add context before re-raising.
    try:
        return Metadata(
            video_info=video_info,
            segments=list(segments_with_summary),
            chapters=chapters,
            skip_suggestions=skip_suggestions,
        )
    except ValidationError as exc:
        log.error("Phase 4 schema validation failed:\n%s", exc)
        raise


def export_metadata(
    workspace: Workspace,
    meta_raw: MetaRaw,
    segments: Sequence[Segment],
    *,
    analysis_version: str = ANALYSIS_VERSION,
) -> Path:
    """
    Build, validate, and write ``metadata.json``.

    Returns the path to the written file. Raises ``pydantic.ValidationError``
    on any schema violation.
    """
    transcript_segments = _load_transcript(workspace)
    embeddings = _load_embeddings(workspace)

    metadata = build_metadata(
        meta_raw,
        segments,
        transcript_segments,
        embeddings,
        analysis_version=analysis_version,
    )

    out_path = workspace.metadata_path
    out_path.write_text(metadata.model_dump_json(indent=2))
    log.info(
        "Wrote metadata.json to %s (segments=%d, chapters=%d, skips=%d, version=%s)",
        out_path,
        len(metadata.segments),
        len(metadata.chapters),
        len(metadata.skip_suggestions),
        metadata.video_info.analysis_version,
    )
    return out_path