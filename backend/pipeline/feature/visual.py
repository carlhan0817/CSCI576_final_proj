import cv2
import json
import torch
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Assuming these are imported from your existing architecture
from backend.pipeline.workspace import Workspace
from backend.pipeline.schemas import VisualFeatures, VisualFrameFeature
from backend.pipeline.logging_setup import get_logger

# Baseline threshold for OpenCV histogram differences. Will be overridden by TOML later.
DEFAULT_CUT_THRESHOLD = 0.6 

# The classes we want CLIP to look for in educational/long-form video
SCENE_LABELS = [
    "a presentation slide",
    "a person talking to a camera",
    "a screen recording of software",
    "a blank screen"
]

def _load_clip_model(device: str):
    """Loads the CLIP model and processor. Caches locally after first run."""
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)
    return processor, model

def _calculate_histogram(image_path: Path):
    """Reads an image and calculates its normalized 3D color histogram."""
    img = cv2.imread(str(image_path))
    # Convert to HSV for better color comparison resilience to lighting changes
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Calculate 3D histogram (Hue, Saturation, Value)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist

def extract_visual_features(workspace: Workspace, device: str = "cpu", cut_threshold: float = DEFAULT_CUT_THRESHOLD) -> Path:
    """
    Reads 1 FPS JPEGs, calculates frame-to-frame histogram diffs, runs CLIP zero-shot classification,
    and writes to features/visual_features.json.
    """
    log = get_logger(workspace.log_path)
    
    features_dir = workspace.root / "features"
    features_dir.mkdir(exist_ok=True)
    out_path = features_dir / "visual_features.json"
    
    if out_path.exists():
        log.info("Visual features already exist. Skipping.")
        return out_path
        
    log.info("Stage 2 (Visual): Loading CLIP model...")
    processor, model = _load_clip_model(device)
    
    # Grab all frames and sort them numerically based on the filename (frame_000000.jpg)
    frames_dir = workspace.root / "frames_cache"
    frame_files = sorted(list(frames_dir.glob("frame_*.jpg")))
    
    if not frame_files:
        log.warning("No frames found in frames_cache. Skipping visual features.")
        return out_path

    log.info(f"Processing {len(frame_files)} frames for visual features...")
    
    frame_features = []
    prev_hist = None
    
    # Process each frame
    for frame_path in frame_files:
        # Extract the second index directly from the filename
        frame_idx = int(frame_path.stem.split("_")[1])
        timestamp_sec = float(frame_idx) # Because it's exactly 1 FPS
        
        # 1. OpenCV Histogram Difference
        curr_hist = _calculate_histogram(frame_path)
        hist_diff = 0.0
        is_cut = False
        
        if prev_hist is not None:
            # cv2.compareHist returns 1.0 for identical, lower for different
            similarity = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
            # Convert similarity to a "difference" score (0.0 = identical, higher = different)
            hist_diff = max(0.0, 1.0 - similarity)
            is_cut = hist_diff > cut_threshold
            
        prev_hist = curr_hist
        
        # 2. CLIP Zero-Shot Classification
        image = Image.open(frame_path)
        inputs = processor(text=SCENE_LABELS, images=image, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Apply softmax to get probabilities across our specific labels
            probs = outputs.logits_per_image.softmax(dim=1)[0].cpu().numpy()
            
        clip_results = {label: float(prob) for label, prob in zip(SCENE_LABELS, probs)}
        
        # 3. Append to Results
        frame_features.append(
            VisualFrameFeature(
                frame_index=frame_idx,
                timestamp_sec=timestamp_sec,
                hist_diff_to_previous=hist_diff,
                is_hard_cut=is_cut,
                clip_labels=clip_results
            )
        )
        
    # 4. Save Artifact
    result = VisualFeatures(frames=frame_features)
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=2))
        
    log.info(f"Visual features saved to {out_path.name}")
    return out_path