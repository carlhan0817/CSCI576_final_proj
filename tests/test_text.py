from pathlib import Path
from backend.pipeline.workspace import Workspace
from CSCI576_final_proj.backend.pipeline.feature.text import extract_text_features

# Point this to whatever mp4 file you ran through Phase 1
mp4_path = Path("path/to/your/test_video.mp4")
ws = Workspace.for_video(mp4_path)

# Run the extraction
out_path = extract_text_features(ws)
print(f"Success! Check the output at: {out_path}")