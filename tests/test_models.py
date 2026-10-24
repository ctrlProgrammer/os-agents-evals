import io
import json
import urllib.error

import pytest

from models import AnthropicModel, GeminiModel, JevModel, OpenAIModel, Price, post_json


class _Response:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


def _http_error(status):
    return urllib.error.HTTPError("https://example.test", status, "no", {}, io.BytesIO(b""))


def test_post_json_retries_a_transient_status_then_returns_the_body(monkeypatch):
    for status in (429, 500, 502, 503, 504):
        calls = {"n": 0}
        pauses = []

        def urlopen(request, timeout):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _http_error(status)
            return _Response({"ok": True})

        monkeypatch.setattr("models.urllib.request.urlopen", urlopen)
        monkeypatch.setattr("models.time.sleep", lambda seconds: pauses.append(seconds))

        assert post_json("https://example.test", {"q": 1}, {}) == {"ok": True}
        assert calls["n"] == 2
        assert pauses == [1]


def test_post_json_does_not_retry_a_rejected_request(monkeypatch):
    calls = {"n": 0}
    pauses = []

    def urlopen(request, timeout):
        calls["n"] += 1
        raise _http_error(401)

    monkeypatch.setattr("models.urllib.request.urlopen", urlopen)
    monkeypatch.setattr("models.time.sleep", lambda seconds: pauses.append(seconds))

    with pytest.raises(RuntimeError, match="HTTP 401"):
        post_json("https://example.test", {}, {})
    assert calls["n"] == 1
    assert pauses == []


def test_post_json_retries_a_timeout_then_returns_the_body(monkeypatch):
    calls = {"n": 0}
    pauses = []

    def urlopen(request, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.URLError(TimeoutError("timed out"))
        return _Response({"ok": True})

    monkeypatch.setattr("models.urllib.request.urlopen", urlopen)
    monkeypatch.setattr("models.time.sleep", lambda seconds: pauses.append(seconds))

    assert post_json("https://example.test", {}, {}) == {"ok": True}
    assert calls["n"] == 2
    assert pauses == [1]


def test_post_json_stops_after_three_timeouts(monkeypatch):
    calls = {"n": 0}
    pauses = []

    def urlopen(request, timeout):
        calls["n"] += 1
        raise urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr("models.urllib.request.urlopen", urlopen)
    monkeypatch.setattr("models.time.sleep", lambda seconds: pauses.append(seconds))

    with pytest.raises(RuntimeError, match="timed out"):
        post_json("https://example.test", {}, {})
    assert calls["n"] == 3
    assert pauses == [1, 1]


def test_jev_returns_the_choice_and_its_token_counts(monkeypatch):
    seen = {}

    def poster(url, body, headers):
        seen["url"] = url
        seen["body"] = body
        seen["authorization"] = headers["Authorization"]
        return {
            "answers": {"etiqueta": {"choice": "fraude"}},
            "usage": {"input_tokens": 120, "output_tokens": 12},
        }

    monkeypatch.setattr("models.post_json", poster)
    model = JevModel(
        model="jev-latest",
        criteria={"fraude": "la zona declarada contradice el GPS"},
        price=Price(input_per_mtok=1.0, output_per_mtok=2.0),
        api_key="test-key",
    )

    completion = model.complete("El anuncio dice El Poblado y el GPS dice Centro-Oriental.")

    assert completion.text == "fraude"
    assert (completion.input_tokens, completion.output_tokens) == (120, 12)
    assert seen["url"] == "https://api.typesafe.ai/v1/systemone"
    assert seen["authorization"] == "Bearer test-key"
    assert seen["body"]["state"].startswith("El anuncio")
    question = seen["body"]["questions"]["etiqueta"]
    assert question["type"] == "choice"
    assert question["criteria"]["fraude"] == "la zona declarada contradice el GPS"


def test_jev_builds_its_question_from_the_json_prompt(monkeypatch):
    seen = {}

    def poster(url, body, headers):
        seen["body"] = body
        return {
            "answers": {"etiqueta": {"choice": "ok"}},
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    monkeypatch.setattr("models.post_json", poster)
    model = JevModel(
        model="jev-latest",
        price=Price(input_per_mtok=1.0, output_per_mtok=2.0),
        api_key="test-key",
        criteria={"ok": "consistente", "fraude": "la zona contradice el GPS"},
        instructions="Cuál etiqueta describe la publicación?",
        state="Publicación de una inmobiliaria.",
    )

    model.complete("GPS distinto")

    body = seen["body"]
    assert body["state"] == "Publicación de una inmobiliaria.\n\nGPS distinto"
    question = body["questions"]["etiqueta"]
    assert question["instructions"] == "Cuál etiqueta describe la publicación?"
    assert question["criteria"] == {"ok": "consistente", "fraude": "la zona contradice el GPS"}


def test_jev_reports_the_confidence_of_its_choice(monkeypatch):
    def poster(url, body, headers):
        return {
            "answers": {
                "etiqueta": {
                    "type": "choice",
                    "choice": "nevera/cocacola",
                    "confidence": 0.81,
                    "probabilities": {"nevera/cocacola": 0.88, "ninguno": 0.12},
                }
            },
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }

    monkeypatch.setattr("models.post_json", poster)
    model = JevModel(
        model="jev-latest",
        criteria={"nevera/cocacola": None, "ninguno": None},
        price=Price(input_per_mtok=1.0, output_per_mtok=2.0),
        api_key="test-key",
    )

    assert model.complete("foto").confidence == 0.81


def test_gemini_without_text_is_an_error(monkeypatch):
    def poster(url, body, headers):
        return {"candidates": []}

    monkeypatch.setattr("models.post_json", poster)
    model = GeminiModel(
        model="gemini-2.5-flash",
        price=Price(input_per_mtok=0.30, output_per_mtok=2.50),
        api_key="gemini-key",
    )

    with pytest.raises(RuntimeError, match="no text"):
        model.complete("Clasifica este anuncio.")


def test_gemini_blank_text_is_an_error(monkeypatch):
    def poster(url, body, headers):
        return {"candidates": [{"content": {"parts": [{"text": "  "}]}}]}

    monkeypatch.setattr("models.post_json", poster)
    model = GeminiModel(
        model="gemini-2.5-flash",
        price=Price(input_per_mtok=0.30, output_per_mtok=2.50),
        api_key="gemini-key",
    )

    with pytest.raises(RuntimeError, match="no text"):
        model.complete("Clasifica este anuncio.")


def test_openai_returns_the_choice_and_its_token_counts(monkeypatch):
    seen = {}

    def poster(url, body, headers):
        seen["url"] = url
        seen["body"] = body
        seen["authorization"] = headers["Authorization"]
        return {
            "choices": [{"message": {"content": " Abyssinian "}}],
            "usage": {"prompt_tokens": 80, "completion_tokens": 4},
        }

    monkeypatch.setattr("models.post_json", poster)
    model = OpenAIModel(
        model="gpt-4o-mini",
        price=Price(input_per_mtok=0.15, output_per_mtok=0.60),
        api_key="openai-key",
    )

    completion = model.complete("Name the breed.")

    assert completion.text == "Abyssinian"
    assert (completion.input_tokens, completion.output_tokens) == (80, 4)
    assert seen["url"] == "https://api.openai.com/v1/chat/completions"
    assert seen["authorization"] == "Bearer openai-key"
    assert seen["body"]["model"] == "gpt-4o-mini"
    assert seen["body"]["temperature"] == 0
    assert seen["body"]["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Name the breed."}]}
    ]


def test_openai_applies_the_json_prompt_around_the_case_text(monkeypatch):
    seen = {}

    def poster(url, body, headers):
        seen["body"] = body
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    monkeypatch.setattr("models.post_json", poster)
    model = OpenAIModel(
        model="gpt-4o-mini",
        price=Price(input_per_mtok=0.15, output_per_mtok=0.60),
        api_key="openai-key",
        system="You name breeds.",
        instructions="Reply with one label.",
        temperature=0.2,
    )

    model.complete("a photo")

    assert seen["body"]["messages"] == [
        {"role": "system", "content": "You name breeds."},
        {"role": "user", "content": [{"type": "text", "text": "Reply with one label.\n\na photo"}]},
    ]
    assert seen["body"]["temperature"] == 0.2
    assert "thinking" not in seen["body"]


def test_openai_sends_the_image_bytes_and_never_the_file_name(monkeypatch, tmp_path):
    photo = tmp_path / "Abyssinian_201.jpg"
    photo.write_bytes(b"first-bytes")
    seen = {}

    def poster(url, body, headers):
        seen["body"] = body
        return {"choices": [{"message": {"content": "Abyssinian"}}], "usage": {}}

    monkeypatch.setattr("models.post_json", poster)
    model = OpenAIModel(
        model="gpt-4o-mini",
        price=Price(input_per_mtok=0.15, output_per_mtok=0.60),
        api_key="openai-key",
    )

    model.complete("Name the breed.", str(photo))

    content = seen["body"]["messages"][0]["content"]
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,Zmlyc3QtYnl0ZXM="},
    }
    assert "Abyssinian_201" not in json.dumps(seen["body"])


def test_openai_without_text_is_an_error(monkeypatch):
    monkeypatch.setattr(
        "models.post_json",
        lambda url, body, headers: {"choices": [{"message": {"content": "  "}}]},
    )
    model = OpenAIModel(
        model="gpt-4o-mini",
        price=Price(input_per_mtok=0.15, output_per_mtok=0.60),
        api_key="openai-key",
    )

    with pytest.raises(RuntimeError, match="no text"):
        model.complete("Name the breed.")


def test_anthropic_returns_the_text_block_and_its_token_counts(monkeypatch):
    seen = {}

    def poster(url, body, headers):
        seen["url"] = url
        seen["body"] = body
        seen["headers"] = headers
        return {
            "content": [{"type": "text", "text": " Egyptian_Mau "}],
            "usage": {"input_tokens": 90, "output_tokens": 3},
        }

    monkeypatch.setattr("models.post_json", poster)
    model = AnthropicModel(
        model="claude-haiku-5-5",
        price=Price(input_per_mtok=0.10, output_per_mtok=0.50),
        api_key="anthropic-key",
    )

    completion = model.complete("Name the breed.")

    assert completion.text == "Egyptian_Mau"
    assert (completion.input_tokens, completion.output_tokens) == (90, 3)
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["headers"]["x-api-key"] == "anthropic-key"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    assert seen["body"]["model"] == "claude-haiku-5-5"
    assert seen["body"]["max_tokens"] == 1024
    assert seen["body"]["temperature"] == 0
    assert "system" not in seen["body"]
    assert seen["body"]["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Name the breed."}]}
    ]


def test_anthropic_sends_the_image_before_the_prompt_and_never_the_file_name(monkeypatch, tmp_path):
    photo = tmp_path / "Abyssinian_201.jpg"
    photo.write_bytes(b"first-bytes")
    seen = {}

    def poster(url, body, headers):
        seen["body"] = body
        return {"content": [{"type": "text", "text": "Abyssinian"}], "usage": {}}

    monkeypatch.setattr("models.post_json", poster)
    model = AnthropicModel(
        model="claude-haiku-5-5",
        price=Price(input_per_mtok=0.10, output_per_mtok=0.50),
        api_key="anthropic-key",
        system="You name breeds.",
        instructions="Reply with one label.",
        max_tokens=32,
    )

    model.complete("a photo", str(photo))

    assert seen["body"]["system"] == "You name breeds."
    assert seen["body"]["max_tokens"] == 32
    assert seen["body"]["messages"][0]["content"] == [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": "Zmlyc3QtYnl0ZXM=",
            },
        },
        {"type": "text", "text": "Reply with one label.\n\na photo"},
    ]
    assert "Abyssinian_201" not in json.dumps(seen["body"])


def test_anthropic_scores_the_text_block_and_not_the_thinking_block(monkeypatch):
    monkeypatch.setattr(
        "models.post_json",
        lambda url, body, headers: {
            "content": [
                {"type": "thinking", "thinking": "maybe a cat"},
                {"type": "text", "text": "Siamese"},
            ],
            "usage": {"input_tokens": 1, "output_tokens": 2},
        },
    )
    model = AnthropicModel(
        model="claude-haiku-5-5",
        price=Price(input_per_mtok=0.10, output_per_mtok=0.50),
        api_key="anthropic-key",
    )

    assert model.complete("Name the breed.").text == "Siamese"


def test_anthropic_without_text_is_an_error(monkeypatch):
    monkeypatch.setattr(
        "models.post_json",
        lambda url, body, headers: {"content": [{"type": "thinking", "thinking": "hmm"}]},
    )
    model = AnthropicModel(
        model="claude-haiku-5-5",
        price=Price(input_per_mtok=0.10, output_per_mtok=0.50),
        api_key="anthropic-key",
    )

    with pytest.raises(RuntimeError, match="no text"):
        model.complete("Name the breed.")


def test_deepseek_posts_a_chat_completion_with_thinking_off(monkeypatch):
    seen = {}

    def poster(url, body, headers):
        seen["url"] = url
        seen["body"] = body
        seen["authorization"] = headers["Authorization"]
        return {
            "choices": [{"message": {"content": "Siamese", "reasoning_content": "a long thought"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8},
        }

    monkeypatch.setattr("models.post_json", poster)
    model = OpenAIModel(
        model="deepseek-flash",
        price=Price(input_per_mtok=0.15, output_per_mtok=0.60),
        api_key="deepseek-key",
        url="https://api.deepseek.com/chat/completions",
        thinking="disabled",
    )

    completion = model.complete("Name the breed.")

    assert completion.text == "Siamese"
    assert completion.output_tokens == 8
    assert seen["url"] == "https://api.deepseek.com/chat/completions"
    assert seen["authorization"] == "Bearer deepseek-key"
    assert seen["body"]["thinking"] == {"type": "disabled"}
    assert seen["body"]["model"] == "deepseek-flash"


def test_gemini_returns_the_ai_studio_text_and_token_counts(monkeypatch):
    seen = {}

    def poster(url, body, headers):
        seen["url"] = url
        seen["body"] = body
        seen["api_key"] = headers["x-goog-api-key"]
        return {
            "candidates": [{"content": {"parts": [{"text": " anomalia_datos "}]}}],
            "usageMetadata": {"promptTokenCount": 80, "candidatesTokenCount": 4},
        }

    monkeypatch.setattr("models.post_json", poster)
    model = GeminiModel(
        model="gemini-2.5-flash",
        price=Price(input_per_mtok=0.30, output_per_mtok=2.50),
        api_key="gemini-key",
    )

    completion = model.complete("Clasifica este anuncio.")

    assert completion.text == "anomalia_datos"
    assert (completion.input_tokens, completion.output_tokens) == (80, 4)
    assert seen["url"] == (
        "https://generativelanguage.googleapis.com/v1beta"
        "/models/gemini-2.5-flash:generateContent"
    )
    assert seen["api_key"] == "gemini-key"
    assert seen["body"]["contents"][0]["parts"][0]["text"] == "Clasifica este anuncio."


def test_gemini_applies_the_json_prompt_around_the_case_text(monkeypatch):
    seen = {}

    def poster(url, body, headers):
        seen["body"] = body
        return {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}

    monkeypatch.setattr("models.post_json", poster)
    model = GeminiModel(
        model="gemini-2.5-flash",
        price=Price(input_per_mtok=0.30, output_per_mtok=2.50),
        api_key="gemini-key",
        system="Eres un auditor.",
        instructions="Responde solo con la etiqueta.",
        temperature=0.2,
    )

    model.complete("GPS distinto")

    body = seen["body"]
    assert body["systemInstruction"] == {"parts": [{"text": "Eres un auditor."}]}
    assert body["contents"][0]["parts"][0]["text"] == "Responde solo con la etiqueta.\n\nGPS distinto"
    assert body["generationConfig"]["temperature"] == 0.2


def test_gemini_without_a_system_prompt_sends_none(monkeypatch):
    seen = {}

    def poster(url, body, headers):
        seen["body"] = body
        return {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}

    monkeypatch.setattr("models.post_json", poster)
    model = GeminiModel(
        model="gemini-2.5-flash",
        price=Price(input_per_mtok=0.30, output_per_mtok=2.50),
        api_key="gemini-key",
    )

    model.complete("hola")

    assert "systemInstruction" not in seen["body"]
    assert seen["body"]["contents"][0]["parts"][0]["text"] == "hola"
    assert seen["body"]["generationConfig"]["temperature"] == 0


def test_gemini_counts_thinking_tokens_as_output(monkeypatch):
    def poster(url, body, headers):
        return {
            "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
            "usageMetadata": {
                "promptTokenCount": 80,
                "candidatesTokenCount": 4,
                "thoughtsTokenCount": 30,
            },
        }

    monkeypatch.setattr("models.post_json", poster)
    model = GeminiModel(
        model="gemini-2.5-flash",
        price=Price(input_per_mtok=0.30, output_per_mtok=2.50),
        api_key="gemini-key",
    )

    completion = model.complete("Clasifica.")

    assert completion.output_tokens == 34


def test_gemini_sends_the_image_bytes_and_never_the_file_name(monkeypatch, tmp_path):
    first = tmp_path / "Abyssinian_201.jpg"
    second = tmp_path / "Bengal_7.png"
    first.write_bytes(b"first-bytes")
    second.write_bytes(b"second-bytes")
    seen = {}

    def poster(url, body, headers):
        seen["body"] = body
        return {
            "candidates": [{"content": {"parts": [{"text": "Abyssinian"}]}}],
            "usageMetadata": {"promptTokenCount": 300, "candidatesTokenCount": 2},
        }

    monkeypatch.setattr("models.post_json", poster)
    model = GeminiModel(
        model="gemini-2.5-flash",
        price=Price(input_per_mtok=0.30, output_per_mtok=2.50),
        api_key="gemini-key",
    )

    model.complete("Name the breed.", f"{first}\n{second}")

    parts = seen["body"]["contents"][0]["parts"]
    assert parts[0] == {"text": "Name the breed."}
    assert parts[1] == {"inline_data": {"mime_type": "image/jpeg", "data": "Zmlyc3QtYnl0ZXM="}}
    assert parts[2] == {"inline_data": {"mime_type": "image/png", "data": "c2Vjb25kLWJ5dGVz"}}
    assert "Abyssinian_201" not in json.dumps(seen["body"])


def test_jev_sends_the_prepared_image_as_an_asset_and_never_the_file_name(monkeypatch, tmp_path):
    photo = tmp_path / "Abyssinian_201.jpg"
    photo.write_bytes(b"original")
    seen = {}

    def poster(url, body, headers):
        seen["body"] = body
        seen["headers"] = headers
        return {
            "answers": {"etiqueta": {"choice": "Abyssinian"}},
            "usage": {"input_tokens": 500, "output_tokens": 5},
        }

    monkeypatch.setattr("models.post_json", poster)
    monkeypatch.setattr("models.prepare_jpeg", lambda path: b"resized")
    model = JevModel(
        model="jev-latest",
        criteria={"Abyssinian": None, "Bengal": None},
        price=Price(input_per_mtok=0.042, output_per_mtok=0),
        api_key="test-key",
    )

    completion = model.complete("Name the breed.", str(photo))

    (placeholder, asset), = seen["body"]["assets"].items()
    assert asset == {"type": "image", "format": "jpg", "data_base64": "cmVzaXplZA=="}
    assert placeholder in seen["body"]["state"]
    assert seen["headers"]["TypeSafe-Danger-Zone-Images"] == "true"
    assert seen["body"]["questions"]["etiqueta"]["criteria"] == {"Abyssinian": None, "Bengal": None}
    assert "Abyssinian_201" not in json.dumps(seen["body"])
    assert completion.text == "Abyssinian"


def test_a_text_only_jev_request_carries_no_assets_and_no_images_header(monkeypatch):
    seen = {}

    def poster(url, body, headers):
        seen["body"] = body
        seen["headers"] = headers
        return {"answers": {"etiqueta": {"choice": "ok"}}, "usage": {"input_tokens": 1, "output_tokens": 1}}

    monkeypatch.setattr("models.post_json", poster)
    model = JevModel(
        model="jev-latest",
        criteria={"ok": None},
        price=Price(input_per_mtok=0.042, output_per_mtok=0),
        api_key="test-key",
    )

    model.complete("Clasifica.")

    assert "assets" not in seen["body"]
    assert "TypeSafe-Danger-Zone-Images" not in seen["headers"]
