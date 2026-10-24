import base64
import json
import mimetypes
import time
import uuid
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from images import prepare_jpeg


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    confidence: float | None = None


@dataclass(frozen=True)
class FakeModel:
    """Deterministic model: canned answers, token counts and simulated latency."""

    answers: dict[str, str]
    latency_s: float
    input_tokens: int
    output_tokens: int
    price: Price

    def complete(self, prompt: str, image: str = "") -> Completion:
        return Completion(
            text=self.answers.get(image or prompt, ""),
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            latency_s=self.latency_s,
        )


def post_json(url: str, body: dict, headers: dict) -> dict:
    """POST JSON and return the decoded object. The error text stays off the message."""
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"model request failed with HTTP {error.code}") from error


def image_paths(image: str) -> list[str]:
    """Case images arrive as one string: absolute paths joined by newlines."""
    return image.split("\n") if image else []


def encode(data: bytes) -> str:
    return base64.b64encode(data).decode()


@dataclass(frozen=True)
class JevModel:
    """TypeSafe System One. JEV answers a Choice; the choice key is the label."""

    model: str
    price: Price
    api_key: str
    criteria: dict[str, str | None]
    instructions: str = "Elige la única etiqueta que corresponde al caso."
    state: str = ""

    def complete(self, prompt: str, image: str = "") -> Completion:
        started = time.perf_counter()
        # Only the bytes travel, never the file name: it can carry the answer.
        assets = {
            f"image-{uuid.uuid4().hex}": {
                "type": "image",
                "format": "jpg",
                "data_base64": encode(prepare_jpeg(Path(path))),
            }
            for path in image_paths(image)
        }
        body = {
            "model": self.model,
            "state": "\n\n".join(part for part in (self.state, prompt, *assets) if part),
            "questions": {
                "etiqueta": {
                    "type": "choice",
                    "instructions": self.instructions,
                    "criteria": self.criteria,
                }
            },
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if assets:
            body["assets"] = assets
            headers["TypeSafe-Danger-Zone-Images"] = "true"
        payload = post_json("https://api.typesafe.ai/v1/systemone", body, headers)
        usage = payload["usage"]
        answer = payload["answers"]["etiqueta"]
        return Completion(
            text=answer["choice"],
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            latency_s=time.perf_counter() - started,
            confidence=answer.get("confidence"),
        )


@dataclass(frozen=True)
class GeminiModel:
    """Gemini through the Gemini API (AI Studio). The reply text is the label."""

    model: str
    price: Price
    api_key: str
    system: str = ""
    instructions: str = ""
    temperature: float = 0

    def complete(self, prompt: str, image: str = "") -> Completion:
        started = time.perf_counter()
        text = "\n\n".join(part for part in (self.instructions, prompt) if part)
        # Only the bytes travel, never the file name: it can carry the answer.
        parts = [{"text": text}] + [
            {
                "inline_data": {
                    "mime_type": mimetypes.guess_type(path)[0] or "image/jpeg",
                    "data": encode(Path(path).read_bytes()),
                }
            }
            for path in image_paths(image)
        ]
        url = (
            "https://generativelanguage.googleapis.com/v1beta"
            f"/models/{self.model}:generateContent"
        )
        body = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": self.temperature},
        }
        if self.system:
            body["systemInstruction"] = {"parts": [{"text": self.system}]}
        payload = post_json(url, body, {"x-goog-api-key": self.api_key})
        parts = payload["candidates"][0]["content"]["parts"]
        usage = payload.get("usageMetadata", {})
        return Completion(
            text="".join(part.get("text", "") for part in parts).strip(),
            input_tokens=int(usage.get("promptTokenCount", 0)),
            # thinking tokens are billed as output
            output_tokens=int(usage.get("candidatesTokenCount", 0))
            + int(usage.get("thoughtsTokenCount", 0)),
            latency_s=time.perf_counter() - started,
        )
