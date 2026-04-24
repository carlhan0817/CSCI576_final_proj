import subprocess
from pathlib import Path
import pytest
import imageio_ffmpeg


FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


@pytest.fixture(scope="session")
def synthetic_mp4(tmp_path_factory) -> Path:
    """Deterministic 3-second MP4: 320x240 color-pattern video + 440Hz sine audio."""
    out_dir = tmp_path_factory.mktemp("fixtures")
    out = out_dir / "synth.mp4"
    cmd = [
        FFMPEG, "-y",
        "-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=10",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3:sample_rate=44100",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    assert out.exists() and out.stat().st_size > 0
    return out
