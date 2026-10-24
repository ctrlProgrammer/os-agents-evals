import argparse
import json
import os
import random
from pathlib import Path

from models import FakeModel, GeminiModel, JevModel, Price
from prompts import load_prompt
from results import save_result
from runner import Case, run_eval


def load_cases(path: Path) -> list[Case]:
    if (path / "dataset.json").is_file():
        items = json.loads((path / "dataset.json").read_text())
        cases = []
        for item in items:
            prompt = item.get("text", "")
            raw_path = item.get("path", "")
            paths = raw_path if isinstance(raw_path, list) else [raw_path]
            # shortcut: several images travel as one newline-joined path, split when a vision client opens them
            image = "\n".join(str((path / item_path).resolve()) for item_path in paths) if item["type"] == "image" else ""
            cases.append(Case(prompt=prompt, expected=item["expected"], image=image))
        return cases
    lines = path.read_text().splitlines()
    return [Case(**json.loads(line)) for line in lines if line.strip()]


def _env_key(spec: dict, default_name: str) -> str:
    """Read the service key from the env var named by the spec. The value never lives in the file."""
    name = spec.get("api_key_env", default_name)
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"{name} is required")
    return value


def _only(prompt: dict, keys: tuple[str, ...]) -> dict:
    return {key: prompt[key] for key in keys if key in prompt}


def build_model(spec: dict, prompt: dict) -> FakeModel | JevModel | GeminiModel:
    """Build the model named by spec['type']: fake, jev, or gemini.

    `prompt` is the model's own JSON prompt. JEV needs `criteria`; Gemini takes
    `system`, `instructions` and `temperature`.
    """
    price = Price(**spec["price"])
    kind = spec.get("type", "fake")
    if kind == "fake":
        return FakeModel(
            answers=spec["answers"],
            latency_s=spec["latency_s"],
            input_tokens=spec["input_tokens"],
            output_tokens=spec["output_tokens"],
            price=price,
        )
    if kind == "jev":
        if not prompt.get("criteria"):
            raise SystemExit("the JEV prompt needs a non-empty criteria")
        return JevModel(
            model=spec.get("model", "jev-latest"),
            price=price,
            api_key=_env_key(spec, "TYPESAFE_API_KEY"),
            criteria=prompt["criteria"],
            **_only(prompt, ("instructions", "state")),
        )
    if kind == "gemini":
        return GeminiModel(
            model=spec["model"],
            price=price,
            api_key=_env_key(spec, "GEMINI_API_KEY"),
            **_only(prompt, ("system", "instructions", "temperature")),
        )
    raise SystemExit(f"unknown model type: {kind}")


def spread(cases: list[Case], limit: int | None) -> list[Case]:
    """At most `limit` cases, evenly spaced, so a dataset sorted by class keeps its variety."""
    if limit is None or limit >= len(cases):
        return cases
    return [cases[i * len(cases) // limit] for i in range(limit)]


def _quotas(counts: dict[str, int], limit: int) -> dict[str, int]:
    """How many cases each class gives. A class smaller than its share gives all of it and the rest is shared again."""
    quota = {label: 0 for label in counts}
    left = limit
    active = [label for label in counts if counts[label] > 0]
    while left and active:
        share, extra = divmod(left, len(active))
        if share == 0:
            for label in active[:left]:
                quota[label] += 1
            break
        still_active = []
        for index, label in enumerate(active):
            room = counts[label] - quota[label]
            give = min(share + (index < extra), room)
            quota[label] += give
            left -= give
            if quota[label] < counts[label]:
                still_active.append(label)
        active = still_active
    return quota


def stratified(cases: list[Case], limit: int | None, seed: int) -> list[Case]:
    """At most `limit` cases, about the same number from each class. `seed` repeats the pick."""
    if limit is None or limit >= len(cases):
        return cases
    groups: dict[str, list[tuple[int, Case]]] = {}
    for index, case in enumerate(cases):
        groups.setdefault(case.expected, []).append((index, case))
    quota = _quotas({label: len(items) for label, items in groups.items()}, limit)
    picked_by = random.Random(seed)
    chosen = []
    for label, items in groups.items():
        take = quota[label]
        chosen.extend(items if take == len(items) else picked_by.sample(items, take))
    return [case for _, case in sorted(chosen)]


def format_row(result) -> str:
    low = round(result.accuracy_ci95[0] * 100)
    high = round(result.accuracy_ci95[1] * 100)
    row = (
        f"{result.model}: {result.accuracy:.0%} (IC {low}-{high}%) exactitud, "
        f"${result.cost_per_1k:.2f}/1k consultas, p95 {result.p95_latency_s:.2f}s, "
        f"F1 {result.f1_macro:.2f}"
    )
    if result.confidence_correct is None and result.confidence_wrong is None:
        return row
    shown = [
        f"{value:.2f} si {when}"
        for value, when in (
            (result.confidence_correct, "acierta"),
            (result.confidence_wrong, "falla"),
        )
        if value is not None
    ]
    return f"{row}, confianza {', '.join(shown)}"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="os-evals")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run a dataset against models")
    run.add_argument("dataset", type=Path)
    run.add_argument("--models", required=True, help="comma-separated model names")
    run.add_argument("--config", type=Path, default=Path("models.json"))
    run.add_argument("--prompt", default="v1", help="prompt version: datasets/<name>/prompts/<version>/<model type>.json")
    run.add_argument("--workers", type=int, default=1, help="calls to the model made at once")
    run.add_argument("--max-evals", type=int, help="run only this many cases, the same number from each class")
    run.add_argument("--no-stratified", action="store_true", help="pick the cases evenly spaced instead")
    run.add_argument("--seed", type=int, default=7, help="repeats the stratified pick")
    args = parser.parse_args(argv)
    if args.max_evals is not None and args.max_evals < 1:
        parser.error("--max-evals must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    dataset = args.dataset
    if not (dataset / "dataset.json").is_file() and not dataset.is_file():
        dataset = Path("datasets") / dataset
    all_cases = load_cases(dataset)
    if args.no_stratified:
        cases = spread(all_cases, args.max_evals)
        sample = {"strategy": "spread", "seed": None}
    else:
        cases = stratified(all_cases, args.max_evals, args.seed)
        sample = {"strategy": "stratified", "seed": args.seed}
    specs = json.loads(args.config.read_text())
    is_folder = (dataset / "dataset.json").is_file()
    for name in args.models.split(","):
        kind = specs[name].get("type", "fake")
        prompt, prompt_file = load_prompt(dataset, args.prompt, kind) if is_folder and kind != "fake" else ({}, None)
        result = run_eval(cases, name, build_model(specs[name], prompt), args.workers)
        print(format_row(result))
        if is_folder:
            save_result(dataset, result, args.prompt, prompt_file, sample, limited=len(cases) < len(all_cases))
