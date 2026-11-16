"""Speech and transcription: tp.speak() / transcribe(), and the LiteLLM backend's speech and
transcription."""

import asyncio
import io
import types
import wave

import pytest

from llm_kit.core.media import backends as media_backends
from llm_kit.core.media.types import Media, audio_seconds
from llm_kit.core.touchpoints import touchpoint
from tests.test_media import CALLS, env  # noqa: F401
from tests.test_media import _setup as _media_setup  # noqa: F401


def wav(seconds: float, rate: int = 8000) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as clip:
        clip.setnchannels(1)
        clip.setsampwidth(2)
        clip.setframerate(rate)
        clip.writeframes(b"\x00\x00" * int(seconds * rate))
    return out.getvalue()


def voice(**kw):
    kw.setdefault("default", "openai/tts-1")
    kw.setdefault("fallbacks", ["elevenlabs/eleven_multilingual_v2"])
    return touchpoint("t.voice", **kw)


def ears(**kw):
    kw.setdefault("default", "openai/whisper-1")
    kw.setdefault("fallbacks", ["groq/whisper-large-v3"])
    return touchpoint("t.ears", **kw)


def test_speak_sends_the_text_and_each_models_voice():
    point = voice(model_params={"openai/*": {"voice": "alloy"}, "elevenlabs/*": {"voice": "21m00"}})
    result = point.speak("Welcome back.", format="mp3", env=env(keys={"openai": ["o"], "elevenlabs": ["e"]}))
    ((ref, kind, request, params),) = CALLS
    assert (ref, kind) == ("openai/tts-1", "speech") and request == {"text": "Welcome back.", "format": "mp3"}
    assert params == {"voice": "alloy"} and result.units == {"characters": 13}


def test_transcribe_sends_the_audio_and_measures_wav():
    result = ears().transcribe(wav(2.5), language="en", timestamps=True, env=env(keys={"openai": ["o"]}))
    ((_, kind, request, _),) = CALLS
    assert kind == "transcription" and request["language"] == "en" and request["timestamps"] == ["segment"]
    assert isinstance(request["audio"], Media) and request["duration"] == 2.5
    assert result.text == "hello there" and result.units == {"input_seconds": 12.5}  # the provider's duration wins


def test_audio_seconds():
    assert audio_seconds(Media.of(wav(1.25))) == 1.25
    assert audio_seconds(Media.of(b"ID3...")) is None


# ------------------------------------------------------------------ LiteLLM


class Binary:
    def __init__(self, content):
        self.content = content
        self._hidden_params = {"response_cost": 0.0002}


class Transcript(types.SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


@pytest.fixture
def litellm(monkeypatch, settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "MEDIA_BACKEND": "litellm"}
    sent = {}

    def speech(**kw):
        sent["speech"] = kw
        return Binary(wav(0.5))

    async def aspeech(**kw):
        sent["aspeech"] = kw
        return Binary(b"ID3" + b"\x00" * 20)

    def transcription(**kw):
        sent["transcription"] = kw
        return Transcript(
            text="hello there",
            language="english",
            duration=3.2,
            segments=[{"start": 0.0, "end": 3.2, "text": "hello there"}],
            usage={"type": "duration", "seconds": 4},
        )

    async def atranscription(**kw):
        sent["atranscription"] = kw
        return Transcript(
            text="hi",
            usage={
                "type": "tokens",
                "input_tokens": 120,
                "output_tokens": 3,
                "input_token_details": {"audio_tokens": 110, "text_tokens": 10},
            },
        )

    fake = types.SimpleNamespace(
        speech=speech, aspeech=aspeech, transcription=transcription, atranscription=atranscription
    )
    monkeypatch.setattr(media_backends, "_litellm", lambda: fake)
    return sent


def test_litellm_speech(litellm):
    result = voice(fallbacks=[]).speak("Hi", voice="alloy", format="wav", speed=1.1, env=env())
    assert litellm["speech"] == {
        "model": "openai/tts-1",
        "input": "Hi",
        "response_format": "wav",
        "voice": "alloy",
        "speed": 1.1,
        "api_key": "o",
    }
    assert result.first.mime_type == "audio/wav" and result.duration == 0.5
    assert result.units == {"characters": 2, "output_seconds": 0.5} and str(result.cost) == "0.0002"


def test_litellm_speech_async(litellm):
    result = asyncio.run(voice(fallbacks=[]).aspeak("Hi", env=env()))
    assert result.first.mime_type == "audio/mpeg" and "response_format" not in litellm["aspeech"]


def test_litellm_transcription(litellm):
    result = ears(fallbacks=[]).transcribe(wav(1), language="en", timestamps=["word"], env=env())
    sent = litellm["transcription"]
    assert sent["file"][1] == wav(1) and sent["file"][2] == "audio/wav"
    assert sent["response_format"] == "verbose_json" and sent["timestamp_granularities"] == ["word"]
    assert sent["language"] == "en" and "duration" not in sent
    assert result.text == "hello there" and result.language == "english" and result.duration == 3.2
    assert result.segments == [{"start": 0.0, "end": 3.2, "text": "hello there"}]
    assert result.units == {"input_seconds": 3.2}


def test_litellm_transcription_billed_by_tokens(litellm):
    result = asyncio.run(ears(fallbacks=[]).atranscribe(b"ID3abc", format="text", env=env()))
    assert litellm["atranscription"]["response_format"] == "text"
    assert result.text == "hi" and result.units == {"input_audio_tokens": 110}
    assert (result.input_tokens, result.output_tokens) == (10, 3)
