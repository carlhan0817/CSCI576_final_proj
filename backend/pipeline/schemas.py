from __future__ import annotations
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from typing import Dict

class AudioFeatureSegment(BaseModel):
    start: float = Field(..., ge=0.0)
    end: float = Field(..., ge=0.0)
    is_speech: bool = Field(True, description="True if human speech is detected in this window.")

class AudioFeatures(BaseModel):
    model_used: str = "silero_vad"
    segments: List[AudioFeatureSegment]

class VisualFrameFeature(BaseModel):
    frame_index: int = Field(..., ge=0)
    timestamp_sec: float = Field(..., ge=0.0)
    hist_diff_to_previous: float = Field(0.0, description="Color histogram difference from the previous frame. High value = likely cut.")
    is_hard_cut: bool = Field(False, description="True if hist_diff exceeds the strict baseline threshold.")
    clip_labels: Dict[str, float] = Field(default_factory=dict, description="Probabilities of predefined scene classes.")

class VisualFeatures(BaseModel):
    model_used: str = "openai/clip-vit-base-patch32"
    frames: List[VisualFrameFeature]

class TextFeatureSegment(BaseModel):
    id: int = Field(..., ge=0)
    start: float = Field(..., ge=0.0)
    end: float = Field(..., ge=0.0)
    text: str
    similarity_to_next: Optional[float] = Field(None, description="Cosine similarity to the sequential segment. None for the last segment.")
    is_potential_boundary: bool = Field(False, description="True if similarity dips below a strict baseline threshold.")

class TextFeatures(BaseModel):
    model_used: str = "all-MiniLM-L6-v2"
    segments: List[TextFeatureSegment]

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
