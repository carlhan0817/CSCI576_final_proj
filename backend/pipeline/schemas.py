from __future__ import annotations
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, List, Literal, Optional


class AudioFeatureSegment(BaseModel):
    start: float = Field(..., ge=0.0)
    end: float = Field(..., ge=0.0)
    is_speech: bool = Field(False, description="True if Silero VAD detects speech in this second.")
    rms_energy: float = Field(0.0, ge=0.0, description="Root-mean-square amplitude of the audio chunk.")
    spectral_centroid: float = Field(0.0, ge=0.0, description="Frequency centroid of the power spectrum (Hz).")
    spectral_bandwidth: float = Field(0.0, ge=0.0, description="Spectral bandwidth around the centroid (Hz).")
    zero_crossing_rate: float = Field(0.0, ge=0.0, description="Fraction of samples where the waveform crosses zero.")
    spectral_entropy: float = Field(0.0, ge=0.0, description="[V2.1] Shannon entropy of the normalized power spectrum.")
    audio_class: Literal["speech", "music", "silence", "noise"] = Field(
        "silence", description="Heuristic audio class for this second."
    )


class AudioFeatures(BaseModel):
    model_used: str = "silero_vad+librosa"
    segments: List[AudioFeatureSegment]


class VisualFrameFeature(BaseModel):
    frame_index: int = Field(..., ge=0)
    timestamp_sec: float = Field(..., ge=0.0)
    hist_diff_to_previous: float = Field(0.0, description="HSV histogram correlation distance from the previous frame.")
    is_hard_cut: bool = Field(False, description="True if hist_diff exceeds the cut threshold.")
    clip_labels: Dict[str, float] = Field(default_factory=dict, description="Zero-shot CLIP probabilities per scene prompt.")
    mean_luminance: float = Field(0.0, ge=0.0, description="Mean pixel brightness in [0, 255] (grayscale).")
    luminance_variance: float = Field(0.0, ge=0.0, description="Variance of pixel brightness.")
    is_black_frame: bool = Field(False, description="True if mean_luminance < 10 and luminance_variance < 50.")
    motion_intensity: float = Field(0.0, ge=0.0, description="Mean absolute pixel difference from the previous frame.")
    chroma_diff: float = Field(0.0, ge=0.0, description="[V2.1] Mean chroma channel difference after 4:2:0 downsampling.")
    dct_hf_energy: float = Field(0.0, ge=0.0, description="[V2.1] Normalized high-frequency DCT energy of the frame.")


class VisualFeatures(BaseModel):
    model_used: str = "openai/clip-vit-base-patch32"
    frames: List[VisualFrameFeature]


class TextFeatureSegment(BaseModel):
    id: int = Field(..., ge=0)
    start: float = Field(..., ge=0.0)
    end: float = Field(..., ge=0.0)
    text: str
    similarity_to_next: Optional[float] = Field(None, description="Cosine similarity to the next segment. None for the last.")
    is_potential_boundary: bool = Field(False, description="True if similarity_to_next is below threshold.")
    matched_keywords: List[str] = Field(
        default_factory=list,
        description="Keyword group matches found in this segment's text (e.g. 'sponsor:sponsored by')."
    )


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


# ============================================================
# Phase 3 Schemas: Final Metadata
# ============================================================

# 10-class taxonomy from PDD §2.3
SegmentLabel = Literal[
    "core_content",
    "intro",
    "outro",
    "sponsorship",
    "self_promotion",
    "recap",
    "transition",
    "dead_air",
    "holding_screen",
    "filler",
]


class SegmentEvidence(BaseModel):
    """Why a segment got its label. Filled in by classify.py."""
    visual_score: float = Field(0.0, description="Visual modality contribution to chosen label.")
    audio_score: float = Field(0.0, description="Audio modality contribution.")
    text_score: float = Field(0.0, description="Text modality contribution.")
    triggered_rules: List[str] = Field(default_factory=list, description="Hard-rule names that fired in this segment.")


class Segment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    segment_id: int = Field(..., ge=0)
    start_sec: float = Field(..., ge=0.0)
    end_sec: float = Field(..., ge=0.0)
    label: SegmentLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: SegmentEvidence
    summary: str = ""
    user_corrected: bool = False


class Chapter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chapter_id: int = Field(..., ge=0)
    start_sec: float = Field(..., ge=0.0)
    end_sec: float = Field(..., ge=0.0)
    title: str


class VideoInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filename: str
    duration_sec: float = Field(..., ge=0.0)
    fps: float = Field(..., gt=0.0)
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    analysis_version: str = "0.3.0"
    verified_by_human: bool = False


class Metadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    video_info: VideoInfo
    segments: List[Segment]
    chapters: List[Chapter]
    skip_suggestions: List[int] = Field(default_factory=list, description="segment_ids that should be skipped")
