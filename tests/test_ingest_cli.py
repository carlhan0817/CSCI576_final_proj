import json
from types import SimpleNamespace

from backend.pipeline import audio
from backend.pipeline import ingest as ingest_mod
from backend.pipeline import transcribe as transcribe_mod


def _fake_model():
    class FakeModel:
        def transcribe(self, wav_path, **kwargs):
            info = SimpleNamespace(language="en", duration=3.0)
            segs = [SimpleNamespace(id=0, start=0.0, end=3.0, text="test")]
            return iter(segs), info
    return FakeModel()


def test_end_to_end_pipeline_creates_all_artifacts(tmp_path, synthetic_mp4, monkeypatch):
    monkeypatch.setattr(transcribe_mod, "_load_model", lambda n, d: _fake_model())

    workspace_root = tmp_path / "workspace"
    ws = ingest_mod.run_pipeline(
        synthetic_mp4,
        workspace_root=workspace_root,
        model_name="base",
        device="cpu",
        force=False,
    )

    assert ws.meta_raw_path.exists()
    assert ws.audio_path.exists()
    assert ws.transcript_path.exists()
    assert ws.frames_dir.is_dir()
    assert len(list(ws.frames_dir.glob("frame_*.jpg"))) >= 2
    assert ws.log_path.exists()

    meta = json.loads(ws.meta_raw_path.read_text())
    assert meta["filename"] == "synth.mp4"


def test_cache_skip_reruns_nothing_when_complete(tmp_path, synthetic_mp4, monkeypatch):
    monkeypatch.setattr(transcribe_mod, "_load_model", lambda n, d: _fake_model())
    workspace_root = tmp_path / "workspace"

    ingest_mod.run_pipeline(synthetic_mp4, workspace_root=workspace_root,
                            model_name="base", device="cpu", force=False)

    calls = {"probe": 0, "audio": 0, "frames": 0, "transcribe": 0}

    from backend.pipeline import frames, probe
    monkeypatch.setattr(probe, "probe_video",
                        lambda *a, **kw: calls.__setitem__("probe", calls["probe"] + 1))
    monkeypatch.setattr(audio, "extract_audio",
                        lambda *a, **kw: calls.__setitem__("audio", calls["audio"] + 1))
    monkeypatch.setattr(frames, "sample_frames",
                        lambda *a, **kw: calls.__setitem__("frames", calls["frames"] + 1))
    monkeypatch.setattr(transcribe_mod, "transcribe",
                        lambda *a, **kw: calls.__setitem__("transcribe", calls["transcribe"] + 1))

    ingest_mod.run_pipeline(synthetic_mp4, workspace_root=workspace_root,
                            model_name="base", device="cpu", force=False)

    assert calls == {"probe": 0, "audio": 0, "frames": 0, "transcribe": 0}


def test_force_reruns_all_stages(tmp_path, synthetic_mp4, monkeypatch):
    monkeypatch.setattr(transcribe_mod, "_load_model", lambda n, d: _fake_model())
    workspace_root = tmp_path / "workspace"

    ingest_mod.run_pipeline(synthetic_mp4, workspace_root=workspace_root,
                            model_name="base", device="cpu", force=False)

    calls = {"probe": 0}
    from backend.pipeline import probe
    original_probe = probe.probe_video

    def counting_probe(*a, **kw):
        calls["probe"] += 1
        return original_probe(*a, **kw)

    monkeypatch.setattr(probe, "probe_video", counting_probe)

    ingest_mod.run_pipeline(synthetic_mp4, workspace_root=workspace_root,
                            model_name="base", device="cpu", force=True)
    assert calls["probe"] == 1
