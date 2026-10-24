import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field


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
    accuracy_ci95: tuple[float, float] = (0.0, 0.0)
    confusion: dict = field(default_factory=dict)
    f1_macro: float = 0.0
    invalid_count: int | None = None
    invalid_rate: float | None = None
    invalid_samples: list | None = None


def normalize(text: str) -> str:
    return text.strip().lower()


def cost(completion, price) -> float:
    return (
        completion.input_tokens * price.input_per_mtok
        + completion.output_tokens * price.output_per_mtok
    ) / 1_000_000


def wilson(hits: int, total: int) -> tuple[float, float]:
    """95% Wilson interval of hits/total. It stays inside 0 and 1 when the rate is 0 or 1."""
    if total == 0:
        return (0.0, 0.0)
    z = 1.96
    proportion = hits / total
    z2 = z * z
    denominator = 1 + z2 / total
    center = (proportion + z2 / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z2 / (4 * total * total)) / denominator
    return (center - margin, center + margin)


def p95(values) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def mean(values) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def invalid_report(completions, labels: list[str] | None) -> tuple:
    """How many answers are outside `labels`. None when the run has no label list."""
    if labels is None:
        return None, None, None
    valid = {normalize(label) for label in labels}
    samples = []
    count = 0
    for completion in completions:
        if normalize(completion.text) in valid:
            continue
        count += 1
        if len(samples) < 3:
            samples.append(completion.text.strip()[:80])
    return count, count / len(completions), samples


def run_eval(cases, name, model, workers: int = 1, labels: list[str] | None = None) -> EvalResult:
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # map keeps the order of the cases, so every answer stays with its case
        completions = list(pool.map(lambda case: model.complete(case.prompt, case.image), cases))
    wall_s = time.perf_counter() - started
    known = {}
    for case in cases:
        known.setdefault(normalize(case.expected), case.expected)
    per_label: dict = {}
    confusion: dict = {}
    confidence = {True: [], False: []}
    for case, completion in zip(cases, completions):
        predicted = known.get(normalize(completion.text), "(otra)")
        row = confusion.setdefault(case.expected, {})
        row[predicted] = row.get(predicted, 0) + 1
        counts = per_label.setdefault(case.expected, {"correct": 0, "total": 0})
        hit = predicted == case.expected
        counts["total"] += 1
        counts["correct"] += hit
        if completion.confidence is not None:
            confidence[hit].append(completion.confidence)
    predicted_totals: dict = {}
    for row in confusion.values():
        for label, count in row.items():
            predicted_totals[label] = predicted_totals.get(label, 0) + count
    f1s = []
    for label, counts in per_label.items():
        recall = counts["correct"] / counts["total"]
        precision = counts["correct"] / predicted_totals[label] if predicted_totals.get(label) else 0.0
        counts["precision"] = precision
        counts["recall"] = recall
        counts["f1"] = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        f1s.append(counts["f1"])
    hits = sum(counts["correct"] for counts in per_label.values())
    total_cost = sum(cost(completion, model.price) for completion in completions)
    invalid_count, invalid_rate, invalid_samples = invalid_report(completions, labels)
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
        accuracy_ci95=wilson(hits, len(cases)),
        confusion=confusion,
        f1_macro=sum(f1s) / len(f1s) if f1s else 0.0,
        invalid_count=invalid_count,
        invalid_rate=invalid_rate,
        invalid_samples=invalid_samples,
    )
