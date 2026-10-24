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


# A 429 or a 5xx from the vendor is usually gone a second later. Three tries, then surface it.
RETRY_STATUSES = {429, 500, 502, 503, 504}
ATTEMPTS = 3
RETRY_PAUSE_S = 1


class _Retry(Exception):
    def __init__(self, final: RuntimeError):
        self.final = final


def _post_once(request) -> dict:
    """One POST. A transient failure raises _Retry; anything else propagates."""
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        final = RuntimeError(f"model request failed with HTTP {error.code}")
        if error.code in RETRY_STATUSES:
            raise _Retry(final) from error
        raise final from error
    except urllib.error.URLError as error:
        # urllib wraps a timeout as URLError(TimeoutError(...)).
        if isinstance(error.reason, TimeoutError):
            raise _Retry(RuntimeError("model request failed: timed out")) from error
        raise


def post_json(url: str, body: dict, headers: dict) -> dict:
    """POST JSON and return the decoded object. The error text stays off the message."""
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(ATTEMPTS):
        try:
            return _post_once(request)
        except _Retry as retry:
            if attempt == ATTEMPTS - 1:
                raise retry.final from retry
            time.sleep(RETRY_PAUSE_S)


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


OPENAI_URL = "https://api.openai.com/v1/chat/completions"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
# A breed name fits in far less. The cap leaves room if the model writes a short thought first.
LABEL_MAX_TOKENS = 1024


@dataclass(frozen=True)
class OpenAIModel:
    """Chat Completions. DeepSeek is the same request on its own URL."""

    model: str
    price: Price
    api_key: str
    url: str = OPENAI_URL
    system: str = ""
    instructions: str = ""
    temperature: float = 0
    thinking: str | None = None

    def complete(self, prompt: str, image: str = "") -> Completion:
        started = time.perf_counter()
        text = "\n\n".join(part for part in (self.instructions, prompt) if part)
        # Only the bytes travel, never the file name: it can carry the answer.
        content = [{"type": "text", "text": text}]
        for path in image_paths(image):
            mime = mimetypes.guess_type(path)[0] or "image/jpeg"
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{encode(Path(path).read_bytes())}"},
            })
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": content})
        body = {"model": self.model, "temperature": self.temperature, "messages": messages}
        if self.thinking:
            body["thinking"] = {"type": self.thinking}
        payload = post_json(self.url, body, {"Authorization": f"Bearer {self.api_key}"})
        try:
            raw = payload["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError):
            raw = ""
        if isinstance(raw, list):
            raw = "".join(part.get("text", "") for part in raw if isinstance(part, dict))
        reply = str(raw).strip()
        if not reply:
            raise RuntimeError("chat response had no text")
        usage = payload.get("usage") or {}
        return Completion(
            text=reply,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            latency_s=time.perf_counter() - started,
        )


@dataclass(frozen=True)
class AnthropicModel:
    """Anthropic Messages. The text blocks are the label; a thinking block is not."""

    model: str
    price: Price
    api_key: str
    system: str = ""
    instructions: str = ""
    temperature: float = 0
    max_tokens: int = LABEL_MAX_TOKENS

    def complete(self, prompt: str, image: str = "") -> Completion:
        started = time.perf_counter()
        text = "\n\n".join(part for part in (self.instructions, prompt) if part)
        # Images first: Claude reads them better that way. The file name stays home.
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mimetypes.guess_type(path)[0] or "image/jpeg",
                    "data": encode(Path(path).read_bytes()),
                },
            }
            for path in image_paths(image)
        ]
        content.append({"type": "text", "text": text})
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": content}],
        }
        if self.system:
            body["system"] = self.system
        payload = post_json(
            ANTHROPIC_URL,
            body,
            {"x-api-key": self.api_key, "anthropic-version": ANTHROPIC_VERSION},
        )
        try:
            reply = "".join(
                block.get("text", "")
                for block in payload["content"]
                if isinstance(block, dict) and block.get("type") == "text"
            ).strip()
        except (KeyError, TypeError):
            reply = ""
        if not reply:
            raise RuntimeError("anthropic response had no text")
        usage = payload.get("usage") or {}
        return Completion(
            text=reply,
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            latency_s=time.perf_counter() - started,
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
        try:
            parts = payload["candidates"][0]["content"]["parts"]
            text = "".join(part.get("text", "") for part in parts).strip()
        except (KeyError, IndexError, TypeError):
            text = ""
        if not text:
            raise RuntimeError("gemini response had no text")
        usage = payload.get("usageMetadata", {})
        return Completion(
            text=text,
            input_tokens=int(usage.get("promptTokenCount", 0)),
            # thinking tokens are billed as output
            output_tokens=int(usage.get("candidatesTokenCount", 0))
            + int(usage.get("thoughtsTokenCount", 0)),
            latency_s=time.perf_counter() - started,
        )
