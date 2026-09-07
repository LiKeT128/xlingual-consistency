import math

from xlingual.metrics import (
    accuracy_by_language,
    agreement_with_english,
    cohens_kappa,
    consistency,
    divergent_items,
    kappa_label,
    wilson_interval,
)

LANGS = ("en", "ru", "uk", "sk")


def rows(*specs):
    """specs: (item_id, language, status, category)"""
    return [
        {"item_id": i, "language": l, "status": s, "category": c, "detail": "", "judged": False}
        for i, l, s, c in specs
    ]


def test_wilson_interval_brackets_the_point_estimate():
    low, high = wilson_interval(8, 10)
    assert low < 0.8 < high
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_wilson_interval_is_wide_at_small_n():
    narrow = wilson_interval(80, 100)
    wide = wilson_interval(8, 10)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_accuracy_ignores_errors():
    data = rows(
        ("a", "en", "pass", "factual"),
        ("b", "en", "fail", "factual"),
        ("c", "en", "error", "factual"),
    )
    assert accuracy_by_language(data)["en"].total == 2
    assert accuracy_by_language(data)["en"].passed == 1


def test_consistency_counts_only_complete_items():
    data = rows(
        ("a", "en", "pass", "factual"),
        ("a", "ru", "pass", "factual"),
        ("a", "uk", "pass", "factual"),
        ("a", "sk", "pass", "factual"),
        ("b", "en", "pass", "factual"),
        ("b", "ru", "fail", "factual"),
        ("b", "uk", "pass", "factual"),
        ("b", "sk", "pass", "factual"),
        ("c", "en", "pass", "factual"),
    )
    result = consistency(data, LANGS)
    assert result.total == 2
    assert result.passed == 1


def test_agreement_with_english():
    data = rows(
        ("a", "en", "pass", "factual"),
        ("a", "ru", "pass", "factual"),
        ("a", "uk", "fail", "factual"),
        ("a", "sk", "pass", "factual"),
    )
    agreement = agreement_with_english(data, LANGS)
    assert agreement["ru"].value == 1.0
    assert agreement["uk"].value == 0.0


def test_divergent_items_lists_the_split():
    data = rows(
        ("a", "en", "pass", "geopolitical"),
        ("a", "ru", "fail", "geopolitical"),
        ("a", "uk", "pass", "geopolitical"),
        ("a", "sk", "pass", "geopolitical"),
    )
    diverged = divergent_items(data, LANGS)
    assert len(diverged) == 1
    assert diverged[0]["failed"] == ["ru"]
    assert diverged[0]["passed"] == ["en", "sk", "uk"]


def test_kappa_is_one_for_perfect_agreement():
    pairs = [("pass", "pass"), ("fail", "fail"), ("pass", "pass"), ("fail", "fail")]
    assert cohens_kappa(pairs) == 1.0


def test_kappa_is_zero_when_agreement_is_only_chance():
    # Rater B always says pass; on a set that is 50/50 that is chance-level.
    pairs = [("pass", "pass"), ("fail", "pass"), ("pass", "pass"), ("fail", "pass")]
    assert abs(cohens_kappa(pairs)) < 1e-9


def test_kappa_is_nan_without_data():
    assert math.isnan(cohens_kappa([]))
    assert kappa_label(float("nan")) == "not measured"


def test_kappa_labels():
    assert kappa_label(0.85) == "almost perfect"
    assert kappa_label(0.5) == "moderate"
    assert kappa_label(-0.1) == "worse than chance"
