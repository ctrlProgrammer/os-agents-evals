import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Case:
    prompt: str
    expected: str
    image: str = ""
    meta: dict = field(default_factory=dict)


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
    cost_per_1k_correct: float | None = None
    by_confidence: list | None = None
    per_group: dict = field(default_factory=dict)
    complete: bool = True
    errors: int = 0
    error_samples: list = field(default_factory=list)


def answered_cases(cases, outcomes):
    """Split a run into scored cases and up to three error texts."""
    scored = []
    samples = []
    errors = 0
    for case, (completion, error) in zip(cases, outcomes):
        if error is None:
            scored.append((case, completion))
            continue
        errors += 1
        if len(samples) < 3:
            samples.append(str(error).strip())
    return scored, errors, samples


def call_case(model, case):
    """One case. A failure stays with that case so the rest of the run continues."""
    try:
        return model.complete(case.prompt, case.image), None
    except Exception as error:
        return None, error


CONFIDENCE_THRESHOLDS = (0.5, 0.7, 0.9, 0.95, 0.99)


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
    if not ordered:
        return 0.0
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def mean(values) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def confidence_table(scored: list[tuple[float | None, bool]]) -> list | None:
    """Coverage and accuracy at each threshold, for models that report a confidence."""
    if not scored or all(confidence is None for confidence, _hit in scored):
        return None
    rows = []
    for threshold in CONFIDENCE_THRESHOLDS:
        kept = [hit for confidence, hit in scored if confidence is not None and confidence >= threshold]
        rows.append({
            "threshold": threshold,
            "coverage": len(kept) / len(scored),
            "accuracy": None if not kept else sum(kept) / len(kept),
        })
    return rows


def invalid_report(completions, labels: list[str] | None) -> tuple:
    """How many answers are outside `labels`. None when the run has no label list."""
    if labels is None:
        return None, None, None
    if not completions:
        return 0, 0.0, []
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
        outcomes = list(pool.map(lambda case: call_case(model, case), cases))
    wall_s = time.perf_counter() - started
    scored_cases, errors, error_samples = answered_cases(cases, outcomes)
    known = {}
    for case in cases:
        known.setdefault(normalize(case.expected), case.expected)
    per_label: dict = {}
    confusion: dict = {}
    per_group: dict = {}
    confidence = {True: [], False: []}
    scored = []
    for case, completion in scored_cases:
        predicted = known.get(normalize(completion.text), "(otra)")
        row = confusion.setdefault(case.expected, {})
        row[predicted] = row.get(predicted, 0) + 1
        counts = per_label.setdefault(case.expected, {"correct": 0, "total": 0})
        hit = predicted == case.expected
        counts["total"] += 1
        counts["correct"] += hit
        scored.append((completion.confidence, bool(hit)))
        for key, value in case.meta.items():
            group = per_group.setdefault(key, {}).setdefault(str(value), {"correct": 0, "total": 0})
            group["total"] += 1
            group["correct"] += hit
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
    completions = [completion for _case, completion in scored_cases]
    answered = len(completions)
    total_cost = sum(cost(completion, model.price) for completion in completions)
    invalid_count, invalid_rate, invalid_samples = invalid_report(completions, labels)
    return EvalResult(
        model=name,
        accuracy=hits / answered if answered else 0.0,
        cost_per_1k=(total_cost / answered * 1000) if answered else 0.0,
        p95_latency_s=p95(completion.latency_s for completion in completions),
        cases=len(cases),
        per_label=per_label,
        confidence_correct=mean(confidence[True]),
        confidence_wrong=mean(confidence[False]),
        wall_s=wall_s,
        accuracy_ci95=wilson(hits, answered),
        confusion=confusion,
        f1_macro=sum(f1s) / len(f1s) if f1s else 0.0,
        invalid_count=invalid_count,
        invalid_rate=invalid_rate,
        invalid_samples=invalid_samples,
        cost_per_1k_correct=None if hits == 0 else total_cost / hits * 1000,
        by_confidence=confidence_table(scored),
        per_group=per_group,
        complete=errors == 0,
        errors=errors,
        error_samples=error_samples,
    )
