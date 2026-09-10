from __future__ import annotations

import os
from pathlib import Path

from backend.pipeline.schemas import Transcript, TranscriptSegment
from backend.pipeline.workspace import Workspace

_MODEL_CACHE = Path(__file__).resolve().parents[2] / "workspace" / "models"
_MODEL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_MODEL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(_MODEL_CACHE))


def _load_model(model_name: str, device: str):
    from faster_whisper import WhisperModel

    compute_type = "int8" if device == "cpu" else "float16"
    return WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        download_root=str(_MODEL_CACHE),
    )


def transcribe(
    wav_path: Path,
    workspace: Workspace,
    model_name: str = "base",
    device: str = "cpu",
) -> Transcript:
    model = _load_model(model_name, device)
    segments_iter, info = model.transcribe(str(wav_path), beam_size=1)

    segments: list[TranscriptSegment] = []
    parts: list[str] = []
    for i, seg in enumerate(segments_iter):
        text = seg.text.strip()
        segments.append(TranscriptSegment(
            id=i,
            start=float(seg.start),
            end=float(seg.end),
            text=text,
        ))
        parts.append(text)

    transcript = Transcript(
        language=info.language,
        duration_sec=float(info.duration),
        model=model_name,
        segments=segments,
        full_text=" ".join(parts),
    )
    workspace.transcript_path.write_text(transcript.model_dump_json(indent=2))
    return transcript
