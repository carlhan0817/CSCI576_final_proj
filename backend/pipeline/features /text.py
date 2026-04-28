import json
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Assuming these are imported from your existing architecture
from backend.pipeline.workspace import Workspace
from backend.pipeline.schemas import Transcript, TextFeatures, TextFeatureSegment
from backend.pipeline.logging_setup import get_logger

# We use a default baseline threshold here, but Phase 3 will override this via TOML!
DEFAULT_SIMILARITY_THRESHOLD = 0.35 

def _load_minilm_model(device: str = "cpu") -> SentenceTransformer:
    """
    Loads the MiniLM model. Like Whisper, it will download to HuggingFace cache 
    on the first run and be offline thereafter.
    """
    return SentenceTransformer("all-MiniLM-L6-v2", device=device)

def extract_text_features(workspace: Workspace, device: str = "cpu", threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> Path:
    """
    Reads transcript.json, generates embeddings, calculates sequential similarity, 
    and writes to features/text_features.json.
    """
    log = get_logger(workspace.log_path)
    
    # Define output path (you may need to add features_dir to your Workspace dataclass)
    features_dir = workspace.root / "features"
    features_dir.mkdir(exist_ok=True)
    out_path = features_dir / "text_features.json"
    
    if out_path.exists():
        log.info("Text features already exist. Skipping.")
        return out_path
        
    log.info("Stage 2 (Text): Loading transcript and MiniLM model...")
    
    # 1. Load Transcript
    with open(workspace.transcript_path, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)
        transcript = Transcript.model_validate(transcript_data)
        
    if not transcript.segments:
        log.warning("Transcript is empty. Skipping text features.")
        return out_path

    # 2. Generate Embeddings
    model = _load_minilm_model(device)
    sentences = [seg.text for seg in transcript.segments]
    log.info(f"Generating embeddings for {len(sentences)} segments...")
    
    # Output is an (N, 384) numpy array
    embeddings = model.encode(sentences, convert_to_numpy=True)
    
    # 3. Calculate Sequential Similarity
    log.info("Calculating cosine similarities between adjacent segments...")
    text_feature_segments = []
    
    for i, seg in enumerate(transcript.segments):
        sim_to_next = None
        is_boundary = False
        
        # If there is a next sentence, calculate the distance
        if i < len(transcript.segments) - 1:
            # reshape(1, -1) is required by sklearn for single vector comparisons
            vec_current = embeddings[i].reshape(1, -1)
            vec_next = embeddings[i+1].reshape(1, -1)
            
            # Extract the float from the [[1.0]] result array
            sim_to_next = float(cosine_similarity(vec_current, vec_next)[0][0])
            is_boundary = sim_to_next < threshold
            
        text_feature_segments.append(
            TextFeatureSegment(
                id=seg.id,
                start=seg.start,
                end=seg.end,
                text=seg.text,
                similarity_to_next=sim_to_next,
                is_potential_boundary=is_boundary
            )
        )
        
    # 4. Save Artifact
    result = TextFeatures(segments=text_feature_segments)
    
    with open(out_path, "w", encoding="utf-8") as f:
        # Use model_dump_json for clean Pydantic v2 serialization
        f.write(result.model_dump_json(indent=2))
        
    log.info(f"Text features saved to {out_path.name}")
    return out_path