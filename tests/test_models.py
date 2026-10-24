import json

from models import GeminiModel, JevModel, Price


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
