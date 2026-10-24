from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float


@dataclass(frozen=True)
class FakeModel:
    """Deterministic model: canned answers, token counts and simulated latency."""

    answers: dict[str, str]
    latency_s: float
    input_tokens: int
    output_tokens: int
    price: Price

    def complete(self, prompt: str) -> Completion:
        return Completion(
            text=self.answers.get(prompt, ""),
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            latency_s=self.latency_s,
        )
