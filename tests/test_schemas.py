import json
from backend.pipeline.schemas import MetaRaw, Transcript, TranscriptSegment


def test_meta_raw_roundtrip():
    payload = {
        "filename": "foo.mp4",
        "duration_sec": 12.5,
        "fps": 30.0,
        "width": 640,
        "height": 480,
        "video_codec": "h264",
        "audio_codec": "aac",
        "has_audio": True,
        "ingested_at": "2026-04-24T10:00:00Z",
    }
    m = MetaRaw.model_validate(payload)
    assert m.duration_sec == 12.5
    assert m.has_audio is True
    assert json.loads(m.model_dump_json()) == payload


def test_transcript_roundtrip():
    payload = {
        "language": "en",
        "duration_sec": 3.5,
        "model": "base",
        "segments": [
            {"id": 0, "start": 0.0, "end": 1.2, "text": "hi"},
            {"id": 1, "start": 1.2, "end": 3.5, "text": "there"},
        ],
        "full_text": "hi there",
    }
    t = Transcript.model_validate(payload)
    assert len(t.segments) == 2
    assert t.segments[0].text == "hi"
    assert json.loads(t.model_dump_json()) == payload


def test_transcript_segment_rejects_negative_start():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        TranscriptSegment(id=0, start=-0.1, end=1.0, text="x")
