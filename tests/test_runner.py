from models import Completion, FakeModel, Price
from runner import Case, p95, run_eval


def two_cases():
    return [
        Case(prompt="Capital of France?", expected="Paris"),
        Case(prompt="2 + 2?", expected="4"),
    ]


def fake(answers, latency_s=0.1):
    return FakeModel(
        answers=answers,
        latency_s=latency_s,
        input_tokens=100,
        output_tokens=10,
        price=Price(input_per_mtok=3.0, output_per_mtok=15.0),
    )


def test_accuracy_is_share_of_exact_matches_ignoring_case_and_spaces():
    model = fake({"Capital of France?": "  paris ", "2 + 2?": "5"})

    result = run_eval(two_cases(), "fake", model)

    assert result.accuracy == 0.5


def test_cost_per_1k_queries_comes_from_tokens_and_price_per_mtok():
    model = fake({})

    result = run_eval(two_cases(), "fake", model)

    assert round(result.cost_per_1k, 6) == 0.45


def test_p95_latency_uses_nearest_rank():
    latencies = [round(0.1 * i, 1) for i in range(1, 21)]

    assert p95(latencies) == 1.9


def test_result_reports_p95_latency_of_the_model():
    result = run_eval(two_cases(), "fake", fake({}, latency_s=0.1))

    assert result.p95_latency_s == 0.1


def test_image_case_is_scored_from_the_answer_for_its_path():
    case = Case(prompt="Name the breed.", expected="abyssinian", image="images/abyssinian.png")
    model = fake({"images/abyssinian.png": "Abyssinian"})

    result = run_eval([case], "oxford", model)

    assert result.accuracy == 1.0


class ConfidentModel:
    """Answers with a fixed confidence, right for 'si' and wrong for anything else."""

    price = Price(input_per_mtok=0.0, output_per_mtok=0.0)

    def __init__(self, confidences):
        self.confidences = dict(confidences)

    def complete(self, prompt, image=""):
        return Completion(
            text="si" if prompt in ("a", "b") else "no",
            input_tokens=1,
            output_tokens=1,
            latency_s=0.1,
            confidence=self.confidences[prompt],
        )


def test_result_averages_the_confidence_of_right_and_wrong_answers():
    cases = [
        Case(prompt="a", expected="si"),
        Case(prompt="b", expected="si"),
        Case(prompt="c", expected="si"),
    ]
    model = ConfidentModel({"a": 1.0, "b": 0.8, "c": 0.4})

    result = run_eval(cases, "jev", model)

    assert round(result.confidence_correct, 6) == 0.9
    assert result.confidence_wrong == 0.4


def test_models_without_confidence_leave_it_empty():
    result = run_eval(two_cases(), "fake", fake({}))

    assert result.confidence_correct is None
    assert result.confidence_wrong is None


def test_result_counts_cases_and_hits_per_expected_label():
    model = fake({"Capital of France?": "Paris", "2 + 2?": "5"})

    result = run_eval(two_cases(), "fake", model)

    assert result.cases == 2
    assert result.per_label == {
        "Paris": {"correct": 1, "total": 1},
        "4": {"correct": 0, "total": 1},
    }
