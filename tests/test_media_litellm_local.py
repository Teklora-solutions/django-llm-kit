"""The LiteLLM media backend end to end, against a local OpenAI- and Mistral-shaped server.

The other media tests stub LiteLLM's functions. Here LiteLLM itself builds each request and reads
each response, so a mismatch between what the kit passes and what LiteLLM accepts shows up. The
server is on 127.0.0.1: no live provider is reached."""

import base64
import io
import json
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from llm_kit.core.touchpoints import touchpoint
from tests.test_media import _setup as _media_setup  # noqa: F401, F811
from tests.test_media import env

PNG = b"\x89PNG\r\n\x1a\n" + b"1" * 40
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 40


def wav(seconds=1.0, rate=8000):
    out = io.BytesIO()
    with wave.open(out, "wb") as clip:
        clip.setnchannels(1)
        clip.setsampwidth(2)
        clip.setframerate(rate)
        clip.writeframes(b"\x00\x00" * int(seconds * rate))
    return out.getvalue()


SEEN: list[dict] = []
VIDEO_CHECKS: list[str] = []


class Provider(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, status, body, content_type="application/json"):
        data = json.dumps(body).encode() if not isinstance(body, bytes) else body
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _record(self):
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            body = b""
            while (size := int(self.rfile.readline().strip() or b"0", 16)) > 0:
                body += self.rfile.read(size)
                self.rfile.readline()
            self.rfile.readline()
        else:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
        entry = {
            "method": self.command,
            "path": self.path,
            "auth": self.headers.get("Authorization"),
            "type": self.headers.get("Content-Type", ""),
            "body": body,
        }
        SEEN.append(entry)
        return entry

    def do_POST(self):
        entry = self._record()
        path = self.path.split("?")[0]
        if path.endswith("/images/generations"):
            sent = json.loads(entry["body"])
            n = int(sent.get("n") or 1)
            data = [{"b64_json": base64.b64encode(PNG).decode(), "revised_prompt": "a calm sea"} for _ in range(n)]
            usage = {
                "input_tokens": 12,
                "output_tokens": 1000,
                "total_tokens": 1012,
                "input_tokens_details": {"text_tokens": 12, "image_tokens": 0},
            }
            return self._send(200, {"created": 1, "data": data, "usage": usage})
        if path.endswith("/images/edits"):
            return self._send(200, {"created": 1, "data": [{"b64_json": base64.b64encode(PNG).decode()}]})
        if path.endswith("/audio/speech"):
            return self._send(200, wav(0.5), "audio/wav")
        if path.endswith("/audio/transcriptions"):
            body = {
                "text": "hello there",
                "language": "english",
                "duration": 2.0,
                "segments": [{"id": 0, "start": 0.0, "end": 2.0, "text": "hello there"}],
            }
            return self._send(200, body)
        if path.endswith("/videos"):
            return self._send(200, {"id": "video_1", "object": "video", "status": "queued", "model": "sora-2"})
        if path.endswith("/ocr"):
            body = {
                "pages": [{"index": 0, "markdown": "# Invoice", "images": [], "dimensions": None}],
                "model": "mistral-ocr-2505",
                "usage_info": {"pages_processed": 1, "doc_size_bytes": 10},
            }
            return self._send(200, body)
        return self._send(404, {"error": {"message": f"no route {path}"}})

    def do_GET(self):
        self._record()
        path = self.path.split("?")[0]
        if path.endswith("/videos/video_1/content"):
            return self._send(200, MP4, "video/mp4")
        if path.endswith("/videos/video_1"):
            VIDEO_CHECKS.append(path)
            status = "completed" if len(VIDEO_CHECKS) >= 2 else "in_progress"
            body = {
                "id": "video_1",
                "object": "video",
                "status": status,
                "progress": 100 if status == "completed" else 40,
                "model": "sora-2",
                "seconds": "4",
                "size": "1280x720",
            }
            return self._send(200, body)
        return self._send(404, {"error": {"message": f"no route {path}"}})


@pytest.fixture(scope="module")
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    httpd.shutdown()


@pytest.fixture(autouse=True)
def _litellm(settings, _media_setup, monkeypatch):  # noqa: F811
    pytest.importorskip("litellm")
    import socket

    real = socket.getaddrinfo

    def local_only(host, *args, **kwargs):  # a request that escapes to a real provider fails here
        if host not in ("127.0.0.1", "localhost"):
            raise OSError(f"test tried to reach {host}")
        return real(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", local_only)
    SEEN.clear()
    VIDEO_CHECKS.clear()
    settings.LLM_KIT = {**settings.LLM_KIT, "MEDIA_BACKEND": "litellm"}


def point(name, default, server):
    return touchpoint(name, default=default, params={"api_base": server})


def test_image_generation(server):
    result = point("e2e.image", "openai/gpt-image-1", server).image("a calm sea", n=2, size="1024x1024", env=env())
    (request,) = SEEN
    sent = json.loads(request["body"])
    assert request["path"] == "/v1/images/generations" and request["auth"] == "Bearer o"
    assert sent["model"] == "gpt-image-1" and sent["prompt"] == "a calm sea" and sent["n"] == 2
    assert sent["size"] == "1024x1024"
    assert [m.read() for m in result.media] == [PNG, PNG] and result.revised_prompt == "a calm sea"
    assert result.units["images"] == 2 and result.units["output_image_tokens"] == 1000


def test_image_edit(server):
    result = point("e2e.edit", "openai/gpt-image-1", server).edit_image("night", PNG, env=env())
    (request,) = SEEN
    assert request["path"] == "/v1/images/edits" and request["type"].startswith("multipart/form-data")
    assert PNG in request["body"] and b"night" in request["body"]
    assert result.first.read() == PNG


def test_speech(server):
    result = point("e2e.speech", "openai/tts-1", server).speak("Hi there", voice="alloy", format="wav", env=env())
    (request,) = SEEN
    sent = json.loads(request["body"])
    assert request["path"] == "/v1/audio/speech"
    assert sent == {"model": "tts-1", "input": "Hi there", "voice": "alloy", "response_format": "wav"}
    assert result.first.mime_type == "audio/wav" and result.duration == 0.5
    assert result.units == {"characters": 8, "output_seconds": 0.5}


def test_transcription(server):
    result = point("e2e.ears", "openai/whisper-1", server).transcribe(wav(2), language="en", timestamps=True, env=env())
    (request,) = SEEN
    body = request["body"]
    assert request["path"] == "/v1/audio/transcriptions" and request["type"].startswith("multipart/form-data")
    assert b"verbose_json" in body and b"whisper-1" in body and b'name="language"' in body
    assert b'name="timestamp_granularities[]"' in body and wav(2)[:44] in body
    assert result.text == "hello there" and result.language == "english" and result.duration == 2.0
    assert result.segments[0]["text"] == "hello there" and result.units == {"input_seconds": 2.0}


def test_video(server, settings):
    settings.LLM_KIT = {**settings.LLM_KIT, "MEDIA_POLL_EVERY": 0}
    result = point("e2e.video", "openai/sora-2", server).video("waves", seconds=4, size="1280x720", env=env())
    paths = [(r["method"], r["path"]) for r in SEEN]
    assert paths[0] == ("POST", "/v1/videos")
    assert ("GET", "/v1/videos/video_1") in paths and paths[-1] == ("GET", "/v1/videos/video_1/content")
    assert b"waves" in SEEN[0]["body"] and b"sora-2" in SEEN[0]["body"]
    assert result.first.read() == MP4 and result.first.mime_type == "video/mp4"
    assert result.units == {"videos": 1, "output_seconds": 4.0, "resolution": "720p", "size": "1280x720"}


def test_ocr(server):
    reader = point("e2e.ocr", "mistral/mistral-ocr-latest", server)
    result = reader.ocr(b"%PDF-1.7 tiny", env=env(keys={"mistral": ["m"]}))
    (request,) = SEEN
    sent = json.loads(request["body"])
    assert request["path"] == "/v1/ocr" and request["auth"] == "Bearer m"
    assert sent["model"] == "mistral-ocr-latest" and sent["document"]["type"] == "document_url"
    assert sent["document"]["document_url"].startswith("data:application/pdf;base64,")
    assert result.text == "# Invoice" and result.units == {"pages": 1}


def test_the_async_paths(server, settings):
    import asyncio

    settings.LLM_KIT = {**settings.LLM_KIT, "MEDIA_POLL_EVERY": 0}

    async def every():
        image = await point("e2e.aimage", "openai/gpt-image-1", server).aimage("x", env=env())
        edit = await point("e2e.aedit", "openai/gpt-image-1", server).aedit_image("x", PNG, env=env())
        speech = await point("e2e.aspeech", "openai/tts-1", server).aspeak("Hi", voice="alloy", env=env())
        heard = await point("e2e.aears", "openai/whisper-1", server).atranscribe(wav(1), env=env())
        video = await point("e2e.avideo", "openai/sora-2", server).avideo("waves", env=env())
        text = await point("e2e.aocr", "mistral/mistral-ocr-latest", server).aocr(
            b"%PDF-1.7", env=env(keys={"mistral": ["m"]})
        )
        return image, edit, speech, heard, video, text

    image, edit, speech, heard, video, text = asyncio.run(every())
    assert image.first.read() == PNG and edit.first.read() == PNG
    assert speech.first.mime_type == "audio/wav" and heard.text == "hello there"
    assert video.first.read() == MP4 and text.text == "# Invoice"
