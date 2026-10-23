from os_agents_evals.models import FakeModel, Price
from os_agents_evals.runner import Case, run_eval


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
