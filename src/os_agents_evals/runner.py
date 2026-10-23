import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    prompt: str
    expected: str


@dataclass(frozen=True)
class EvalResult:
    model: str
    accuracy: float
    cost_per_1k: float
    p95_latency_s: float


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


def run_eval(cases, name, model) -> EvalResult:
    completions = [model.complete(case.prompt) for case in cases]
    hits = sum(
        normalize(completion.text) == normalize(case.expected)
        for case, completion in zip(cases, completions)
    )
    total_cost = sum(cost(completion, model.price) for completion in completions)
    return EvalResult(
        model=name,
        accuracy=hits / len(cases),
        cost_per_1k=total_cost / len(cases) * 1000,
        p95_latency_s=p95(completion.latency_s for completion in completions),
    )
