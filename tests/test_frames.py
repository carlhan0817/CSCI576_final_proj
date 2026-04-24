import cv2
from backend.pipeline.workspace import Workspace
from backend.pipeline.frames import sample_frames


def test_sample_frames_writes_one_jpeg_per_second(tmp_path, synthetic_mp4):
    ws = Workspace.for_video(synthetic_mp4, root=tmp_path / "workspace")
    ws.ensure()

    paths = sample_frames(synthetic_mp4, ws, fps=1, longest_side=512, jpeg_quality=85)

    assert 2 <= len(paths) <= 4
    for p in paths:
        assert p.exists()
        assert p.suffix == ".jpg"
        assert p.name.startswith("frame_")
        img = cv2.imread(str(p))
        assert img is not None
        h, w = img.shape[:2]
        assert max(h, w) <= 512
        assert abs((w / h) - (320 / 240)) < 0.05


def test_sample_frames_filename_encodes_index(tmp_path, synthetic_mp4):
    ws = Workspace.for_video(synthetic_mp4, root=tmp_path / "workspace")
    ws.ensure()
    paths = sample_frames(synthetic_mp4, ws, fps=1, longest_side=512, jpeg_quality=85)
    names = sorted(p.name for p in paths)
    assert names[0] == "frame_000000.jpg"
    assert names[1] == "frame_000001.jpg"
