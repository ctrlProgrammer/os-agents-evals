import argparse
import json
from pathlib import Path

from models import FakeModel, Price
from runner import Case, run_eval


def load_cases(path: Path) -> list[Case]:
    lines = path.read_text().splitlines()
    return [Case(**json.loads(line)) for line in lines if line.strip()]


def build_model(spec: dict) -> FakeModel:
    return FakeModel(
        answers=spec["answers"],
        latency_s=spec["latency_s"],
        input_tokens=spec["input_tokens"],
        output_tokens=spec["output_tokens"],
        price=Price(**spec["price"]),
    )


def format_row(result) -> str:
    return (
        f"{result.model}: {result.accuracy:.0%} exactitud, "
        f"${result.cost_per_1k:.2f}/1k consultas, p95 {result.p95_latency_s:.2f}s"
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="os-evals")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run a dataset against models")
    run.add_argument("dataset", type=Path)
    run.add_argument("--models", required=True, help="comma-separated model names")
    run.add_argument("--config", type=Path, default=Path("models.json"))
    args = parser.parse_args(argv)

    cases = load_cases(args.dataset)
    specs = json.loads(args.config.read_text())
    for name in args.models.split(","):
        print(format_row(run_eval(cases, name, build_model(specs[name]))))
