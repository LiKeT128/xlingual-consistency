import pytest

from xlingual.dataset import Item
from xlingual.graders import FAIL, NEEDS_JUDGE, PASS, grade


def make_item(check: dict, category: str = "test") -> Item:
    return Item(
        id="t1",
        category=category,
        check=check,
        prompts={"en": "q", "ru": "q", "uk": "q", "sk": "q"},
    )


@pytest.mark.parametrize(
    "answer",
    ["Bratislava", "Столица — Братислава.", "Hlavné mesto je Bratislava."],
)
def test_contains_any_matches_across_scripts(answer):
    item = make_item({"type": "contains_any", "accept": ["bratislava"]})
    assert grade(item, answer).status == PASS


def test_contains_any_rejects_wrong_answer():
    item = make_item({"type": "contains_any", "accept": ["bratislava"]})
    assert grade(item, "Prague").status == FAIL


def test_number_accepts_comma_decimal_and_surrounding_words():
    item = make_item({"type": "number", "value": 8.4, "tolerance": 0.001})
    assert grade(item, "Всего 8,40 EUR").status == PASS


def test_number_fails_when_absent():
    item = make_item({"type": "number", "value": 8.4})
    assert grade(item, "eight euros forty").status == FAIL


def test_digits_only_rejects_prose():
    item = make_item({"type": "digits_only", "value": 366})
    assert grade(item, "366").status == PASS
    assert grade(item, "It is 366 days").status == FAIL


def test_word_count():
    item = make_item({"type": "word_count", "value": 3})
    assert grade(item, "red green blue").status == PASS
    assert grade(item, "red, green, blue and yellow").status == FAIL


def test_line_count():
    item = make_item({"type": "line_count", "value": 3})
    assert grade(item, "- a\n- b\n- c").status == PASS
    assert grade(item, "- a\n- b").status == FAIL


def test_json_keys_tolerates_code_fences():
    item = make_item({"type": "json_keys", "keys": ["country", "capital"]})
    fenced = '```json\n{"country": "Slovakia", "capital": "Bratislava"}\n```'
    assert grade(item, fenced).status == PASS


def test_json_keys_reports_missing_key():
    item = make_item({"type": "json_keys", "keys": ["country", "capital"]})
    verdict = grade(item, '{"country": "Slovakia"}')
    assert verdict.status == FAIL
    assert "capital" in verdict.detail


def test_json_array_len():
    item = make_item({"type": "json_array_len", "value": 4})
    assert grade(item, '["a","b","c","d"]').status == PASS
    assert grade(item, '["a","b"]').status == FAIL


def test_starts_with_works_in_cyrillic():
    item = make_item({"type": "starts_with", "accept": ["потому"]})
    assert grade(item, "Потому что свет рассеивается.").status == PASS
    assert grade(item, "Свет рассеивается.").status == FAIL


def test_exact_ignores_case_and_punctuation():
    item = make_item({"type": "exact", "accept": ["ready"]})
    assert grade(item, "READY").status == PASS
    assert grade(item, "READY, boss").status == FAIL


def test_judged_checks_are_deferred():
    assert grade(make_item({"type": "judge", "rubric": "x"}), "anything").status == NEEDS_JUDGE
    assert grade(make_item({"type": "refusal", "expected": True}), "no").status == NEEDS_JUDGE
