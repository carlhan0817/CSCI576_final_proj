"""
Phase 3 Step 6: Generate the final metadata.json artifact.

- Build VideoInfo from meta_raw.json.
- Build chapters by merging adjacent core_content segments.
- Build skip_suggestions list of all non-content segment_ids.
- Validate the whole structure with Pydantic.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional

from backend.pipeline.schemas import (
    MetaRaw, Metadata, VideoInfo, Segment, Chapter,
)
from backend.pipeline.workspace import Workspace


def build_chapters(segments: List[Segment]) -> List[Chapter]:
    """
    Group consecutive core_content segments into chapters.
    Title = first 60 chars of the first segment's summary (or "Chapter N" if empty).
    """
    chapters: List[Chapter] = []
    chapter_id = 0
    current_start = None
    current_summary = ""
    
    for seg in segments:
        if seg.label == "core_content":
            if current_start is None:
                current_start = seg.start_sec
                current_summary = seg.summary
            current_end = seg.end_sec
        else:
            if current_start is not None:
                title = (current_summary[:60].rsplit(" ", 1)[0] + "...") if current_summary else f"Chapter {chapter_id + 1}"
                chapters.append(Chapter(
                    chapter_id=chapter_id,
                    start_sec=current_start,
                    end_sec=current_end,
                    title=title.strip(),
                ))
                chapter_id += 1
                current_start = None
                current_summary = ""
    
    # Tail
    if current_start is not None:
        title = (current_summary[:60].rsplit(" ", 1)[0] + "...") if current_summary else f"Chapter {chapter_id + 1}"
        chapters.append(Chapter(
            chapter_id=chapter_id,
            start_sec=current_start,
            end_sec=current_end,
            title=title.strip(),
        ))
    
    return chapters


def build_skip_suggestions(segments: List[Segment]) -> List[int]:
    """All non-core_content segment_ids."""
    return [s.segment_id for s in segments if s.label != "core_content"]


def export_metadata(
    workspace: Workspace,
    meta_raw: MetaRaw,
    segments: List[Segment],
    raw_boundaries: Optional[List[int]] = None,
) -> Path:
    """Write metadata.json and return its path.

    raw_boundaries: the full boundary list from find_boundaries() before smoothing.
        Timestamps at 0 and video-end are excluded from natural_break_candidates
        since they are trivial endpoints, not content transitions.
    """
    video_info = VideoInfo(
        filename=meta_raw.filename,
        duration_sec=meta_raw.duration_sec,
        fps=meta_raw.fps,
        width=meta_raw.width,
        height=meta_raw.height,
        analysis_version="0.3.0",
        verified_by_human=False,
    )

    chapters = build_chapters(segments)
    skip_suggestions = build_skip_suggestions(segments)

    # Build natural_break_candidates: every pre-smooth boundary except 0 and T.
    if raw_boundaries:
        T = int(meta_raw.duration_sec)
        break_candidates = sorted(
            float(b) for b in raw_boundaries if b != 0 and b != T
        )
    else:
        break_candidates = []

    metadata = Metadata(
        video_info=video_info,
        segments=segments,
        chapters=chapters,
        skip_suggestions=skip_suggestions,
        natural_break_candidates=break_candidates,
    )

    workspace.metadata_path.write_text(metadata.model_dump_json(indent=2))
    return workspace.metadata_path