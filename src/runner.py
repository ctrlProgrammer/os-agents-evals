import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    prompt: str
    expected: str
    image: str = ""


@dataclass(frozen=True)
class EvalResult:
    model: str
    accuracy: float
    cost_per_1k: float
    p95_latency_s: float
    cases: int
    per_label: dict
    confidence_correct: float | None = None
    confidence_wrong: float | None = None
    wall_s: float = 0.0


def normalize(text: str) -> str:
    return text.strip().lower()


def cost(completion, price) -> float:
    return (
        completion.input_tokens * price.input_per_mtok
        + completion.output_tokens * price.output_per_mtok
    ) / 1_000_000


def p95(values) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def mean(values) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def run_eval(cases, name, model, workers: int = 1) -> EvalResult:
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # map keeps the order of the cases, so every answer stays with its case
        completions = list(pool.map(lambda case: model.complete(case.prompt, case.image), cases))
    wall_s = time.perf_counter() - started
    per_label: dict = {}
    confidence = {True: [], False: []}
    for case, completion in zip(cases, completions):
        counts = per_label.setdefault(case.expected, {"correct": 0, "total": 0})
        hit = normalize(completion.text) == normalize(case.expected)
        counts["total"] += 1
        counts["correct"] += hit
        if completion.confidence is not None:
            confidence[hit].append(completion.confidence)
    hits = sum(counts["correct"] for counts in per_label.values())
    total_cost = sum(cost(completion, model.price) for completion in completions)
    return EvalResult(
        model=name,
        accuracy=hits / len(cases),
        cost_per_1k=total_cost / len(cases) * 1000,
        p95_latency_s=p95(completion.latency_s for completion in completions),
        cases=len(cases),
        per_label=per_label,
        confidence_correct=mean(confidence[True]),
        confidence_wrong=mean(confidence[False]),
        wall_s=wall_s,
    )
