"""Speech to text with faster-whisper. Audio is processed and never stored."""
import io
from functools import lru_cache


class SpeechUnavailable(RuntimeError):
    pass


@lru_cache
def _model(name: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise SpeechUnavailable("Install requirements-ml.txt to enable speech") from error
    return WhisperModel(name, device="cpu", compute_type="int8")


def transcribe(audio: bytes, model: str = "tiny.en") -> dict:
    segments, _ = _model(model).transcribe(io.BytesIO(audio), vad_filter=True, beam_size=1)
    parts = list(segments)
    text = " ".join(s.text.strip() for s in parts).strip()
    confidence = sum(s.avg_logprob for s in parts) / len(parts) if parts else None
    return {"text": text, "avg_logprob": confidence}
