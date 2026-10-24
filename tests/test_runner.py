import time

from models import Completion, FakeModel, Price
from runner import Case, p95, run_eval, wilson


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


def test_wilson_interval_for_38_hits_out_of_40():
    assert [round(bound, 3) for bound in wilson(38, 40)] == [0.835, 0.986]


def test_result_carries_the_wilson_interval_of_its_accuracy():
    result = run_eval(two_cases(), "fake", fake({"Capital of France?": "Paris", "2 + 2?": "5"}))

    assert result.accuracy == 0.5
    assert [round(bound, 3) for bound in result.accuracy_ci95] == [round(bound, 3) for bound in wilson(1, 2)]


def test_accuracy_is_share_of_exact_matches_ignoring_case_and_spaces():
    model = fake({"Capital of France?": "  paris ", "2 + 2?": "5"})

    result = run_eval(two_cases(), "fake", model)

    assert result.accuracy == 0.5


def test_cost_per_hit_spreads_the_total_cost_over_the_hits_only():
    result = run_eval(two_cases(), "fake", fake({"Capital of France?": "Paris", "2 + 2?": "5"}))

    assert round(result.cost_per_1k, 6) == 0.45
    assert round(result.cost_per_1k_correct, 6) == 0.9


def test_cost_per_hit_is_empty_when_nothing_is_right():
    result = run_eval(two_cases(), "fake", fake({}))

    assert result.cost_per_1k_correct is None


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


def test_accuracy_by_confidence_threshold():
    cases = [Case(prompt=prompt, expected="si") for prompt in "abcd"]
    model = ConfidentModel({"a": 1.0, "b": 0.9, "c": 0.6, "d": 0.4})

    result = run_eval(cases, "jev", model)
    by = {row["threshold"]: row for row in result.by_confidence}

    assert [row["threshold"] for row in result.by_confidence] == [0.5, 0.7, 0.9, 0.95, 0.99]
    assert round(by[0.5]["coverage"], 3) == 0.75
    assert round(by[0.5]["accuracy"], 3) == 0.667
    assert (by[0.9]["coverage"], by[0.9]["accuracy"]) == (0.5, 1.0)


def test_a_threshold_nobody_reaches_has_no_accuracy():
    result = run_eval([Case(prompt="a", expected="si")], "jev", ConfidentModel({"a": 0.1}))
    top = result.by_confidence[-1]

    assert top["threshold"] == 0.99
    assert top["coverage"] == 0.0
    assert top["accuracy"] is None


def test_models_without_confidence_have_no_threshold_table():
    result = run_eval(two_cases(), "fake", fake({}))

    assert result.by_confidence is None


def test_models_without_confidence_leave_it_empty():
    result = run_eval(two_cases(), "fake", fake({}))

    assert result.confidence_correct is None
    assert result.confidence_wrong is None


def test_confusion_matrix_and_f1_per_class():
    cases = [Case(prompt=str(index), expected=label) for index, label in enumerate(["a", "a", "b", "b"])]
    model = fake({"0": "a", "1": "b", "2": "b", "3": "b"})

    result = run_eval(cases, "fake", model)

    assert result.confusion == {"a": {"a": 1, "b": 1}, "b": {"b": 2}}
    assert (result.per_label["a"]["precision"], result.per_label["a"]["recall"]) == (1.0, 0.5)
    assert round(result.per_label["b"]["precision"], 3) == 0.667
    assert result.per_label["b"]["recall"] == 1.0
    assert round(result.f1_macro, 3) == 0.733


def test_invalid_answers_are_counted_and_still_score_as_misses():
    cases = [Case(prompt=prompt, expected=expected) for prompt, expected in (("1", "a"), ("2", "a"), ("3", "b"))]
    model = fake({"1": "a", "2": "A. porque es la primera", "3": "b"})

    result = run_eval(cases, "fake", model, labels=["a", "b"])

    assert result.invalid_count == 1
    assert round(result.invalid_rate, 3) == 0.333
    assert result.invalid_samples == ["A. porque es la primera"]
    assert result.accuracy == 2 / 3


def test_invalid_samples_keep_three_and_clip_at_80_characters():
    text = "x" * 100
    cases = [Case(prompt=str(index), expected="a") for index in range(4)]
    model = fake({str(index): text for index in range(4)})

    result = run_eval(cases, "fake", model, labels=["a"])

    assert result.invalid_count == 4
    assert result.invalid_samples == ["x" * 80] * 3


def test_without_labels_the_invalid_fields_stay_empty():
    result = run_eval(two_cases(), "fake", fake({}))

    assert result.invalid_count is None
    assert result.invalid_rate is None
    assert result.invalid_samples is None


def test_an_answer_outside_the_labels_counts_as_otra():
    result = run_eval([Case(prompt="1", expected="a")], "fake", fake({"1": "no se"}))

    assert result.confusion == {"a": {"(otra)": 1}}
    assert result.per_label["a"]["precision"] == 0.0
    assert result.f1_macro == 0.0


def test_hits_are_counted_per_meta_group():
    cases = [
        Case(prompt="1", expected="a", meta={"tipo": "nevera"}),
        Case(prompt="2", expected="a", meta={"tipo": "nevera"}),
        Case(prompt="3", expected="b", meta={"tipo": "congelador"}),
    ]
    model = fake({"1": "a", "2": "no", "3": "b"})

    result = run_eval(cases, "fake", model)

    assert result.per_group == {
        "tipo": {
            "nevera": {"correct": 1, "total": 2},
            "congelador": {"correct": 1, "total": 1},
        }
    }


def test_a_case_counts_toward_every_group_it_names():
    result = run_eval(
        [Case(prompt="1", expected="a", meta={"tipo": "nevera", "marca": "hit"})],
        "fake",
        fake({"1": "a"}),
    )

    assert result.per_group["tipo"]["nevera"] == {"correct": 1, "total": 1}
    assert result.per_group["marca"]["hit"] == {"correct": 1, "total": 1}


def test_cases_without_meta_leave_the_groups_empty():
    result = run_eval(two_cases(), "fake", fake({}))

    assert result.per_group == {}


def test_result_counts_cases_and_hits_per_expected_label():
    model = fake({"Capital of France?": "Paris", "2 + 2?": "5"})

    result = run_eval(two_cases(), "fake", model)

    assert result.cases == 2
    assert result.per_label["Paris"]["correct"] == 1
    assert result.per_label["4"] == {"correct": 0, "total": 1, "precision": 0.0, "recall": 0.0, "f1": 0.0}


class SlowModel:
    """Echoes the prompt after a pause. Earlier cases pause longer, so a mixed-up order shows."""

    price = Price(input_per_mtok=0.0, output_per_mtok=0.0)

    def __init__(self, pause_s):
        self.pause_s = pause_s

    def complete(self, prompt, image=""):
        time.sleep(self.pause_s * (8 - int(prompt)) / 8)
        return Completion(text=prompt, input_tokens=1, output_tokens=1, latency_s=self.pause_s)


def numbered_cases(count=8):
    return [Case(prompt=str(number), expected=str(number)) for number in range(count)]


def test_workers_overlap_the_calls_and_keep_every_answer_with_its_case():
    sequential = run_eval(numbered_cases(), "slow", SlowModel(0.2), workers=1)

    parallel = run_eval(numbered_cases(), "slow", SlowModel(0.2), workers=4)

    assert parallel.accuracy == 1.0
    assert parallel.per_label == sequential.per_label
    assert list(parallel.per_label) == [str(number) for number in range(8)]
    assert parallel.wall_s < sequential.wall_s / 2


def test_result_reports_the_wall_time_of_the_run():
    result = run_eval(numbered_cases(2), "slow", SlowModel(0.1))

    assert result.wall_s >= 0.1
