from pathlib import Path
from backend.pipeline.workspace import Workspace
from backend.pipeline.features.visual import extract_visual_features

# Point this to the same mp4 file you ran through Phase 1
mp4_path = Path("path/to/your/test_video.mp4")
ws = Workspace.for_video(mp4_path)

out_path = extract_visual_features(ws)
print(f"Success! Check the output at: {out_path}")