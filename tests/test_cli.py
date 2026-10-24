import json
import time

import pytest

from cli import format_row, main
from runner import EvalResult, wilson


def write_prompt(dataset, version, kind, content):
    folder = dataset / "prompts" / version
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{kind}.json").write_text(json.dumps(content))


def test_the_row_shows_the_confidence_interval_of_the_accuracy():
    result = EvalResult(
        model="jev",
        accuracy=0.95,
        cost_per_1k=0.17,
        p95_latency_s=0.62,
        cases=40,
        per_label={},
        accuracy_ci95=wilson(38, 40),
        f1_macro=0.733,
    )

    row = format_row(result)
    assert row.startswith("jev: 95% (IC 83-99%) exactitud")
    assert "F1 0.73" in row


def test_run_prints_one_comparison_row_per_model(tmp_path, capsys):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        '{"prompt": "Capital of France?", "expected": "Paris"}\n'
        '{"prompt": "2 + 2?", "expected": "4"}\n'
    )
    config = tmp_path / "models.json"
    config.write_text(json.dumps({
        "fake": {
            "type": "fake",
            "answers": {"Capital of France?": "Paris", "2 + 2?": "5"},
            "latency_s": 0.1,
            "input_tokens": 100,
            "output_tokens": 10,
            "price": {"input_per_mtok": 3.0, "output_per_mtok": 15.0},
        }
    }))

    main(["run", str(dataset), "--models", "fake", "--config", str(config)])

    assert capsys.readouterr().out == "fake: 50% (IC 9-91%) exactitud, $0.45/1k consultas, p95 0.10s, F1 0.50\n"


def test_run_scores_text_and_image_cases_from_a_dataset_folder(tmp_path, capsys):
    image = tmp_path / "images" / "pet.png"
    image.parent.mkdir()
    image.write_bytes(b"not-a-real-image")
    (tmp_path / "dataset.json").write_text(json.dumps([
        {"type": "image", "path": "images/pet.png", "expected": "abyssinian"},
        {"type": "text", "text": "2 + 2?", "expected": "4"},
    ]))
    config = tmp_path / "models.json"
    config.write_text(json.dumps({
        "fake": {
            "type": "fake",
            "answers": {
                str(image.resolve()): "Abyssinian",
                "2 + 2?": "4",
            },
            "latency_s": 0.1,
            "input_tokens": 100,
            "output_tokens": 10,
            "price": {"input_per_mtok": 3.0, "output_per_mtok": 15.0},
        }
    }))

    main(["run", str(tmp_path), "--models", "fake", "--config", str(config)])

    assert capsys.readouterr().out == "fake: 100% (IC 34-100%) exactitud, $0.45/1k consultas, p95 0.10s, F1 1.00\n"


def test_run_passes_every_image_when_a_case_has_several(tmp_path, capsys):
    first = tmp_path / "images" / "a.png"
    second = tmp_path / "images" / "b.png"
    first.parent.mkdir()
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    (tmp_path / "dataset.json").write_text(json.dumps([
        {
            "type": "image",
            "path": ["images/a.png", "images/b.png"],
            "text": "Which figure matches?",
            "expected": "A",
        },
    ]))
    both = f"{first.resolve()}\n{second.resolve()}"
    config = tmp_path / "models.json"
    config.write_text(json.dumps({
        "fake": {
            "type": "fake",
            "answers": {both: "A"},
            "latency_s": 0.1,
            "input_tokens": 100,
            "output_tokens": 10,
            "price": {"input_per_mtok": 3.0, "output_per_mtok": 15.0},
        }
    }))

    main(["run", str(tmp_path), "--models", "fake", "--config", str(config)])

    assert capsys.readouterr().out == "fake: 100% (IC 21-100%) exactitud, $0.45/1k consultas, p95 0.10s, F1 1.00\n"


def test_jev_uses_the_criteria_of_the_chosen_prompt_version(tmp_path, capsys, monkeypatch):
    (tmp_path / "dataset.json").write_text(json.dumps([
        {"type": "text", "text": "GPS distinto", "expected": "fraude"},
        {"type": "text", "text": "todo bien", "expected": "ok"},
    ]))
    write_prompt(tmp_path, "v1", "jev", {"criteria": {"viejo": None}})
    write_prompt(tmp_path, "v2", "jev", {
        "instructions": "Cuál etiqueta?",
        "state": "Publicación.",
        "criteria": {"fraude": "la zona contradice el GPS", "ok": None, "ninguno": None},
    })
    config = tmp_path / "models.json"
    config.write_text(json.dumps({
        "jev": {
            "type": "jev",
            "model": "jev-latest",
            "api_key_env": "JEV_API_KEY",
            "price": {"input_per_mtok": 0.042, "output_per_mtok": 0},
        }
    }))
    seen = []

    def poster(url, body, headers):
        assert headers["Authorization"] == "Bearer jev-secret"
        seen.append(body)
        return {
            "answers": {"etiqueta": {"choice": "fraude"}},
            "usage": {"input_tokens": 1_000_000, "output_tokens": 0},
        }

    monkeypatch.setenv("JEV_API_KEY", "jev-secret")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "not-this-one")
    monkeypatch.setattr("models.post_json", poster)

    main(["run", str(tmp_path), "--models", "jev", "--config", str(config),
          "--prompt", "v2", "--max-evals", "1"])

    row = capsys.readouterr().out
    assert row.startswith("jev: 100% (IC 21-100%) exactitud, $42.00/1k consultas, p95 ")
    (body,) = seen
    assert body["state"] == "Publicación.\n\nGPS distinto"
    question = body["questions"]["etiqueta"]
    assert question["instructions"] == "Cuál etiqueta?"
    assert question["criteria"] == {"fraude": "la zona contradice el GPS", "ok": None, "ninguno": None}


def test_the_default_prompt_version_is_v1(tmp_path, monkeypatch):
    (tmp_path / "dataset.json").write_text(json.dumps([
        {"type": "text", "text": "uno", "expected": "a"},
    ]))
    write_prompt(tmp_path, "v1", "jev", {"criteria": {"a": None}})
    config = tmp_path / "models.json"
    config.write_text(json.dumps({
        "jev": {"type": "jev", "price": {"input_per_mtok": 0, "output_per_mtok": 0}}
    }))
    seen = []

    def poster(url, body, headers):
        seen.append(body["questions"]["etiqueta"]["criteria"])
        return {"answers": {"etiqueta": {"choice": "a"}}, "usage": {"input_tokens": 1, "output_tokens": 1}}

    monkeypatch.setenv("TYPESAFE_API_KEY", "jev-secret")
    monkeypatch.setattr("models.post_json", poster)

    main(["run", str(tmp_path), "--models", "jev", "--config", str(config)])

    assert seen == [{"a": None}]


def test_a_missing_prompt_version_stops_the_run_naming_the_file(tmp_path, monkeypatch):
    (tmp_path / "dataset.json").write_text(json.dumps([
        {"type": "text", "text": "uno", "expected": "a"},
    ]))
    config = tmp_path / "models.json"
    config.write_text(json.dumps({
        "gemini": {"type": "gemini", "model": "gemini-2.5-flash",
                   "price": {"input_per_mtok": 0, "output_per_mtok": 0}}
    }))
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")

    with pytest.raises(SystemExit) as error:
        main(["run", str(tmp_path), "--models", "gemini", "--config", str(config), "--prompt", "v7"])

    assert str(tmp_path / "prompts" / "v7" / "gemini.json") in str(error.value)


def test_a_jev_prompt_without_criteria_is_rejected(tmp_path, monkeypatch):
    (tmp_path / "dataset.json").write_text(json.dumps([
        {"type": "text", "text": "uno", "expected": "a"},
    ]))
    write_prompt(tmp_path, "v1", "jev", {"instructions": "Cuál?"})
    config = tmp_path / "models.json"
    config.write_text(json.dumps({
        "jev": {"type": "jev", "price": {"input_per_mtok": 0, "output_per_mtok": 0}}
    }))
    monkeypatch.setenv("TYPESAFE_API_KEY", "jev-secret")

    with pytest.raises(SystemExit) as error:
        main(["run", str(tmp_path), "--models", "jev", "--config", str(config)])

    assert "criteria" in str(error.value)


def test_a_jev_run_prints_and_saves_its_confidence(tmp_path, capsys, monkeypatch):
    write_prompt(tmp_path, "v1", "jev", {"criteria": {"nevera/cocacola": None, "ninguno": None}})
    (tmp_path / "dataset.json").write_text(json.dumps([
        {"type": "text", "text": "uno", "expected": "nevera/cocacola"},
        {"type": "text", "text": "dos", "expected": "ninguno"},
    ]))
    config = tmp_path / "models.json"
    config.write_text(json.dumps({
        "jev": {
            "type": "jev",
            "price": {"input_per_mtok": 0, "output_per_mtok": 0},
        }
    }))

    def poster(url, body, headers):
        right = body["state"].endswith("uno")
        return {
            "answers": {"etiqueta": {
                "choice": "nevera/cocacola",
                "confidence": 0.9 if right else 0.4,
            }},
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    monkeypatch.setenv("TYPESAFE_API_KEY", "jev-secret")
    monkeypatch.setattr("models.post_json", poster)

    main(["run", str(tmp_path), "--models", "jev", "--config", str(config)])

    assert "confianza 0.90 si acierta, 0.40 si falla" in capsys.readouterr().out
    result = json.loads((tmp_path / "result" / "v1" / "jev.json").read_text())
    assert (result["confidence_correct"], result["confidence_wrong"]) == (0.9, 0.4)


def test_run_scores_a_gemini_ai_studio_reply(tmp_path, capsys, monkeypatch):
    write_prompt(tmp_path, "v1", "gemini", {"instructions": "Clasifica."})
    (tmp_path / "dataset.json").write_text(json.dumps([
        {"type": "text", "text": "precio imposible", "expected": "anomalia_datos"},
    ]))
    config = tmp_path / "models.json"
    config.write_text(json.dumps({
        "gemini": {
            "type": "gemini",
            "model": "gemini-2.5-flash",
            "api_key_env": "GEMINI_API_KEY",
            "price": {"input_per_mtok": 0.30, "output_per_mtok": 2.50},
        }
    }))

    def poster(url, body, headers):
        assert headers["x-goog-api-key"] == "gemini-secret"
        return {
            "candidates": [{"content": {"parts": [{"text": "anomalia_datos"}]}}],
            "usageMetadata": {"promptTokenCount": 1_000_000, "candidatesTokenCount": 0},
        }

    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    monkeypatch.setenv("TYPESAFE_API_KEY", "not-this-one")
    monkeypatch.setattr("models.post_json", poster)

    main(["run", str(tmp_path), "--models", "gemini", "--config", str(config)])

    row = capsys.readouterr().out
    assert row.startswith("gemini: 100% (IC 21-100%) exactitud, $300.00/1k consultas, p95 ")


def fake_config(tmp_path, answers):
    config = tmp_path / "models.json"
    config.write_text(json.dumps({
        "fake": {
            "type": "fake",
            "answers": answers,
            "latency_s": 0.1,
            "input_tokens": 100,
            "output_tokens": 10,
            "price": {"input_per_mtok": 3.0, "output_per_mtok": 15.0},
        }
    }))
    return config


def test_run_saves_the_metrics_under_result_in_the_dataset_folder(tmp_path):
    dataset = tmp_path / "pets"
    dataset.mkdir()
    (dataset / "dataset.json").write_text(json.dumps([
        {"type": "text", "text": "2 + 2?", "expected": "4"},
        {"type": "text", "text": "3 + 3?", "expected": "6"},
    ]))
    config = fake_config(tmp_path, {"2 + 2?": "4", "3 + 3?": "7"})

    main(["run", str(dataset), "--models", "fake", "--config", str(config)])

    result = json.loads((dataset / "result" / "v1" / "fake.json").read_text())
    assert result["version"] == "v1"
    assert result["prompt"] is None
    assert len(result["hash"]) == 8
    assert {key: result[key] for key in ("dataset", "model", "cases", "accuracy", "p95_latency_s")} == {
        "dataset": "pets",
        "model": "fake",
        "cases": 2,
        "accuracy": 0.5,
        "p95_latency_s": 0.1,
    }
    assert round(result["cost_per_1k"], 6) == 0.45
    assert result["per_label"]["4"]["f1"] == 1.0
    assert result["per_label"]["6"]["f1"] == 0.0
    assert result["confusion"] == {"4": {"4": 1}, "6": {"(otra)": 1}}
    assert result["f1_macro"] == 0.5
    assert result["run_at"].endswith("Z")


def test_a_new_prompt_keeps_the_old_result_and_a_rerun_replaces_its_own(tmp_path, monkeypatch):
    dataset = tmp_path / "pets"
    dataset.mkdir()
    (dataset / "dataset.json").write_text(json.dumps([
        {"type": "text", "text": "2 + 2?", "expected": "4"},
    ]))
    write_prompt(dataset, "v1", "gemini", {"instructions": "Name it."})
    write_prompt(dataset, "v2", "gemini", {"instructions": "Name it twice."})
    config = tmp_path / "models.json"
    config.write_text(json.dumps({
        "gemini": {"type": "gemini", "model": "gemini-2.5-flash",
                   "price": {"input_per_mtok": 0, "output_per_mtok": 0}}
    }))

    def poster(url, body, headers):
        text = body["contents"][0]["parts"][0]["text"]
        answer = "4" if text.startswith("Name it.\n") else "5"
        return {"candidates": [{"content": {"parts": [{"text": answer}]}}]}

    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    monkeypatch.setattr("models.post_json", poster)
    run = ["run", str(dataset), "--models", "gemini", "--config", str(config)]

    main(run + ["--prompt", "v1"])
    main(run + ["--prompt", "v1"])
    assert [path.name for path in (dataset / "result").glob("*/gemini.json")] == ["gemini.json"]

    main(run + ["--prompt", "v2"])

    saved = {
        version: json.loads((dataset / "result" / version / "gemini.json").read_text())
        for version in ("v1", "v2")
    }
    assert {version: (item["version"], item["prompt"], item["accuracy"]) for version, item in saved.items()} == {
        "v1": ("v1", "prompts/v1/gemini.json", 1.0),
        "v2": ("v2", "prompts/v2/gemini.json", 0.0),
    }


def four_case_dataset(tmp_path):
    dataset = tmp_path / "digits"
    dataset.mkdir()
    (dataset / "dataset.json").write_text(json.dumps([
        {"type": "text", "text": text, "expected": text} for text in ("a", "b", "c", "d")
    ]))
    return dataset


def balanced_dataset(tmp_path):
    """10 of a, 10 of b and 1 of c. An even spread of 6 never reaches c."""
    dataset = tmp_path / "pets"
    dataset.mkdir()
    items = [
        {"type": "text", "text": f"{label}-{index}", "expected": label}
        for label, count in (("a", 10), ("b", 10), ("c", 1))
        for index in range(count)
    ]
    (dataset / "dataset.json").write_text(json.dumps(items))
    return dataset


def test_max_evals_takes_every_class_and_records_the_sample(tmp_path):
    dataset = balanced_dataset(tmp_path)
    config = fake_config(tmp_path, {})

    main(["run", str(dataset), "--models", "fake", "--config", str(config), "--max-evals", "6", "--seed", "7"])

    result = json.loads((dataset / "result" / "v1" / "fake-n6.json").read_text())
    assert result["per_label"]["c"]["total"] == 1
    assert result["per_label"]["a"]["total"] + result["per_label"]["b"]["total"] == 5
    assert result["sample"] == {"strategy": "stratified", "seed": 7}


def test_max_evals_runs_cases_spread_over_the_dataset(tmp_path):
    dataset = four_case_dataset(tmp_path)
    # only the 1st and 3rd case are answered right: a head cut would score 50%
    config = fake_config(tmp_path, {"a": "a", "c": "c"})

    main(["run", str(dataset), "--models", "fake", "--config", str(config), "--max-evals", "2", "--no-stratified"])

    result = json.loads((dataset / "result" / "v1" / "fake-n2.json").read_text())
    assert (result["cases"], result["accuracy"]) == (2, 1.0)
    assert result["sample"] == {"strategy": "spread", "seed": None}


def test_max_evals_above_the_dataset_size_runs_every_case(tmp_path):
    dataset = four_case_dataset(tmp_path)
    config = fake_config(tmp_path, {})

    main(["run", str(dataset), "--models", "fake", "--config", str(config), "--max-evals", "50"])

    assert json.loads((dataset / "result" / "v1" / "fake.json").read_text())["cases"] == 4


def test_a_limited_run_does_not_replace_the_full_result(tmp_path):
    dataset = four_case_dataset(tmp_path)
    config = fake_config(tmp_path, {})
    run = ["run", str(dataset), "--models", "fake", "--config", str(config)]

    main(run)
    main(run + ["--max-evals", "2"])

    results = dataset / "result" / "v1"
    assert json.loads((results / "fake.json").read_text())["cases"] == 4
    assert json.loads((results / "fake-n2.json").read_text())["cases"] == 2


def test_max_evals_below_one_is_rejected(tmp_path):
    dataset = four_case_dataset(tmp_path)
    config = fake_config(tmp_path, {})

    with pytest.raises(SystemExit):
        main(["run", str(dataset), "--models", "fake", "--config", str(config), "--max-evals", "0"])


def test_workers_run_the_calls_at_once_and_the_result_keeps_the_wall_time(tmp_path, monkeypatch):
    (tmp_path / "dataset.json").write_text(json.dumps([
        {"type": "text", "text": str(number), "expected": "a"} for number in range(8)
    ]))
    write_prompt(tmp_path, "v1", "gemini", {})
    config = tmp_path / "models.json"
    config.write_text(json.dumps({
        "gemini": {"type": "gemini", "model": "gemini-2.5-flash",
                   "price": {"input_per_mtok": 0, "output_per_mtok": 0}}
    }))

    def poster(url, body, headers):
        time.sleep(0.2)
        return {"candidates": [{"content": {"parts": [{"text": "a"}]}}]}

    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    monkeypatch.setattr("models.post_json", poster)

    main(["run", str(tmp_path), "--models", "gemini", "--config", str(config), "--workers", "4"])

    result = json.loads((tmp_path / "result" / "v1" / "gemini.json").read_text())
    assert result["accuracy"] == 1.0
    assert 0.4 <= result["wall_s"] < 1.0


def test_workers_below_one_are_rejected(tmp_path):
    dataset = four_case_dataset(tmp_path)
    config = fake_config(tmp_path, {})

    with pytest.raises(SystemExit):
        main(["run", str(dataset), "--models", "fake", "--config", str(config), "--workers", "0"])
