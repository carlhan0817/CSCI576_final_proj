import torch
import torchaudio
from pathlib import Path

# Assuming these are imported from your existing architecture
from backend.pipeline.workspace import Workspace
from backend.pipeline.schemas import AudioFeatures, AudioFeatureSegment
from backend.pipeline.logging_setup import get_logger

def _load_silero_vad(device: str = "cpu"):
    """Loads the Silero VAD model from PyTorch Hub. Caches locally after first run."""
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        trust_repo=True
    )
    return model.to(device), utils

def extract_audio_features(workspace: Workspace, device: str = "cpu") -> Path:
    """
    Reads audio_processed.wav, runs Silero Voice Activity Detection, 
    and writes to features/audio_features.json.
    """
    log = get_logger(workspace.log_path)
    
    features_dir = workspace.root / "features"
    features_dir.mkdir(exist_ok=True)
    out_path = features_dir / "audio_features.json"
    
    if out_path.exists():
        log.info("Audio features already exist. Skipping.")
        return out_path
        
    log.info("Stage 2 (Audio): Loading Silero VAD model...")
    model, utils = _load_silero_vad(device)
    (get_speech_timestamps, _, read_audio, _, _) = utils
    
    log.info("Reading Phase 1 processed audio...")
    # Silero's read_audio handles the WAV loading seamlessly
    wav = read_audio(str(workspace.audio_path)).to(device)
    
    log.info("Detecting speech boundaries...")
    # Get timestamps (returns list of dicts with 'start' and 'end' in audio frames)
    speech_timestamps = get_speech_timestamps(
        wav, 
        model, 
        sampling_rate=16000,
        return_seconds=True # This converts the frame counts directly to seconds!
    )
    
    if not speech_timestamps:
        log.warning("No speech detected in the entire audio file.")
    
    # Format the results into our schema
    audio_segments = []
    for ts in speech_timestamps:
        audio_segments.append(
            AudioFeatureSegment(
                start=round(ts['start'], 3),
                end=round(ts['end'], 3),
                is_speech=True
            )
        )
        
    # Save Artifact
    result = AudioFeatures(segments=audio_segments)
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=2))
        
    log.info(f"Audio features saved to {out_path.name}")
    return out_path