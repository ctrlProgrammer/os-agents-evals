from collections import Counter

from cli import stratified
from runner import Case


def cases_of(*counts):
    return [
        Case(prompt=f"{label}-{index}", expected=label)
        for label, count in counts
        for index in range(count)
    ]


def test_stratified_takes_the_same_number_from_each_class():
    picked = stratified(cases_of(("a", 10), ("b", 10), ("c", 10)), 6, seed=7)

    assert Counter(case.expected for case in picked) == {"a": 2, "b": 2, "c": 2}


def test_the_same_seed_picks_the_same_cases_and_another_seed_differs():
    cases = cases_of(("a", 10), ("b", 10), ("c", 10))

    first = [case.prompt for case in stratified(cases, 6, seed=7)]

    assert first == [case.prompt for case in stratified(cases, 6, seed=7)]
    assert first != [case.prompt for case in stratified(cases, 6, seed=8)]


def test_a_small_class_gives_all_it_has_and_the_rest_is_shared():
    picked = stratified(cases_of(("a", 50), ("b", 50), ("c", 1)), 6, seed=7)
    counts = Counter(case.expected for case in picked)

    assert counts["c"] == 1
    assert sorted((counts["a"], counts["b"])) == [2, 3]
