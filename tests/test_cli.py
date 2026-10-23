import json

from os_agents_evals.cli import main


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

    assert capsys.readouterr().out == "fake: 50% exactitud, $0.45/1k consultas, p95 0.10s\n"
