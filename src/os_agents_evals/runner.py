from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    prompt: str
    expected: str


@dataclass(frozen=True)
class EvalResult:
    model: str
    accuracy: float


def normalize(text: str) -> str:
    return text.strip().lower()


def run_eval(cases, name, model) -> EvalResult:
    hits = sum(
        normalize(model.complete(case.prompt).text) == normalize(case.expected)
        for case in cases
    )
    return EvalResult(model=name, accuracy=hits / len(cases))
