from __future__ import annotations
from pydantic import BaseModel, Field, ConfigDict


class MetaRaw(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filename: str
    duration_sec: float = Field(ge=0.0)
    fps: float = Field(gt=0.0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    video_codec: str
    audio_codec: str
    has_audio: bool
    ingested_at: str  # ISO-8601 UTC string


class TranscriptSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: int = Field(ge=0)
    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)
    text: str


class Transcript(BaseModel):
    model_config = ConfigDict(extra="forbid")
    language: str
    duration_sec: float = Field(ge=0.0)
    model: str
    segments: list[TranscriptSegment]
    full_text: str
