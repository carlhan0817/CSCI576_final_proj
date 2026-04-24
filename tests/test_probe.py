import json
from backend.pipeline.workspace import Workspace
from backend.pipeline.probe import probe_video


def test_probe_writes_valid_meta_raw(tmp_path, synthetic_mp4):
    ws = Workspace.for_video(synthetic_mp4, root=tmp_path / "workspace")
    ws.ensure()

    meta = probe_video(synthetic_mp4, ws)

    assert ws.meta_raw_path.exists()
    data = json.loads(ws.meta_raw_path.read_text())
    assert data["filename"] == "synth.mp4"
    assert 2.5 <= data["duration_sec"] <= 3.5
    assert data["width"] == 320
    assert data["height"] == 240
    assert data["has_audio"] is True
    assert data["video_codec"] == "h264"
    assert data["ingested_at"].endswith("Z")
    assert meta.width == 320
