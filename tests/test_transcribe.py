import json
from types import SimpleNamespace
import pytest

from backend.pipeline.workspace import Workspace
from backend.pipeline.schemas import Transcript
from backend.pipeline import transcribe as transcribe_mod


def _fake_segment(id_, start, end, text):
    return SimpleNamespace(id=id_, start=start, end=end, text=text)


def test_transcribe_writes_valid_transcript_json_mocked(tmp_path, synthetic_mp4, monkeypatch):
    ws = Workspace.for_video(synthetic_mp4, root=tmp_path / "workspace")
    ws.ensure()
    ws.audio_path.write_bytes(b"RIFF0000WAVEfmt ")

    class FakeModel:
        def transcribe(self, wav_path, **kwargs):
            segs = [
                _fake_segment(0, 0.0, 1.2, "hello"),
                _fake_segment(1, 1.2, 2.5, "world"),
            ]
            info = SimpleNamespace(language="en", duration=2.5)
            return iter(segs), info

    monkeypatch.setattr(transcribe_mod, "_load_model", lambda name, device: FakeModel())

    t = transcribe_mod.transcribe(ws.audio_path, ws, model_name="base", device="cpu")

    assert ws.transcript_path.exists()
    data = json.loads(ws.transcript_path.read_text())
    assert data["language"] == "en"
    assert data["model"] == "base"
    assert len(data["segments"]) == 2
    assert data["segments"][0]["text"] == "hello"
    assert data["full_text"] == "hello world"
    assert isinstance(t, Transcript)


@pytest.mark.slow
def test_transcribe_real_whisper_on_synthetic(tmp_path, synthetic_mp4):
    """Real faster-whisper run — structural validity only."""
    from backend.pipeline.audio import extract_audio

    ws = Workspace.for_video(synthetic_mp4, root=tmp_path / "workspace")
    ws.ensure()
    extract_audio(synthetic_mp4, ws)

    t = transcribe_mod.transcribe(ws.audio_path, ws, model_name="base", device="cpu")

    assert ws.transcript_path.exists()
    assert t.model == "base"
    assert t.language
