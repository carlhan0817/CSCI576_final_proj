import wave
from backend.pipeline.workspace import Workspace
from backend.pipeline.audio import extract_audio


def test_extract_audio_produces_mono_16k_wav(tmp_path, synthetic_mp4):
    ws = Workspace.for_video(synthetic_mp4, root=tmp_path / "workspace")
    ws.ensure()

    out = extract_audio(synthetic_mp4, ws)

    assert out == ws.audio_path
    assert out.exists()
    with wave.open(str(out), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == 16000
        assert w.getsampwidth() == 2
        duration = w.getnframes() / w.getframerate()
        assert 2.5 <= duration <= 3.5
