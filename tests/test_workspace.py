from pathlib import Path
from backend.pipeline.workspace import Workspace


def test_workspace_paths_from_mp4(tmp_path):
    video = tmp_path / "my_video.mp4"
    video.write_bytes(b"fake")
    ws = Workspace.for_video(video, root=tmp_path / "workspace")

    assert ws.root == tmp_path / "workspace" / "my_video"
    assert ws.meta_raw_path == ws.root / "meta_raw.json"
    assert ws.audio_path == ws.root / "audio_processed.wav"
    assert ws.transcript_path == ws.root / "transcript.json"
    assert ws.frames_dir == ws.root / "frames_cache"
    assert ws.log_path == ws.root / "ingest.log"


def test_workspace_ensure_creates_dirs(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"fake")
    ws = Workspace.for_video(video, root=tmp_path / "workspace")
    ws.ensure()
    assert ws.root.is_dir()
    assert ws.frames_dir.is_dir()


def test_synthetic_mp4_fixture_exists(synthetic_mp4):
    assert synthetic_mp4.exists()
    assert synthetic_mp4.suffix == ".mp4"
    assert synthetic_mp4.stat().st_size > 1000
