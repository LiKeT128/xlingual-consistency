"""Deterministic grading.

Every check that can be decided by code is decided by code. Only refusal and
ambiguity items are handed to a model judge, because they have no single
correct string. Keeping that boundary sharp is the whole point: the more of the
result that rests on deterministic checks, the less of it rests on a judge whose
own bias we would then have to argue about.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .dataset import Item
from .normalize import (
    count_lines,
    count_sentences,
    count_words,
    extract_numbers,
    normalize,
    strip_code_fence,
)

PASS = "pass"
FAIL = "fail"
NEEDS_JUDGE = "needs_judge"
ERROR = "error"


@dataclass(frozen=True)
class Verdict:
    status: str
    detail: str = ""

    @property
    def is_graded(self) -> bool:
        return self.status in (PASS, FAIL)


def grade(item: Item, answer: str) -> Verdict:
    check = item.check
    kind = check["type"]
    if kind in ("refusal", "judge"):
        return Verdict(NEEDS_JUDGE)

    handler = _HANDLERS.get(kind)
    if handler is None:  # pragma: no cover - dataset validation blocks this
        return Verdict(ERROR, f"no handler for check type '{kind}'")
    return handler(check, answer)


def _contains_any(check: dict, answer: str) -> Verdict:
    """Substring match, except for very short needles.

    'Au' as a bare substring would fire on 'August'. Anything three characters
    or shorter therefore has to match a whole token.
    """
    haystack = normalize(answer)
    tokens = set(haystack.split())
    for candidate in check.get("accept", []):
        needle = normalize(candidate)
        if not needle:
            continue
        if len(needle) <= 3 and " " not in needle:
            if needle in tokens:
                return Verdict(PASS, f"matched token '{candidate}'")
        elif needle in haystack:
            return Verdict(PASS, f"matched '{candidate}'")
    return Verdict(FAIL, f"none of {check.get('accept', [])} found")


def _exact(check: dict, answer: str) -> Verdict:
    got = normalize(answer)
    for candidate in check.get("accept", []):
        if got == normalize(candidate):
            return Verdict(PASS)
    return Verdict(FAIL, f"got '{got[:60]}'")


def _starts_with(check: dict, answer: str) -> Verdict:
    got = normalize(answer)
    for candidate in check.get("accept", []):
        if got.startswith(normalize(candidate)):
            return Verdict(PASS, f"starts with '{candidate}'")
    return Verdict(FAIL, f"starts with '{got[:40]}'")


def _number(check: dict, answer: str) -> Verdict:
    target = float(check["value"])
    tolerance = float(check.get("tolerance", 0.001))
    values = extract_numbers(answer)
    if not values:
        return Verdict(FAIL, "no number in answer")
    for value in values:
        if abs(value - target) <= tolerance:
            return Verdict(PASS, f"found {value}")
    return Verdict(FAIL, f"found {values[:5]}, expected {target}")


def _digits_only(check: dict, answer: str) -> Verdict:
    stripped = "".join(answer.split())
    if not stripped.isdigit():
        return Verdict(FAIL, f"not digits only: '{answer.strip()[:40]}'")
    if int(stripped) != int(check["value"]):
        return Verdict(FAIL, f"got {stripped}, expected {check['value']}")
    return Verdict(PASS)


def _word_count(check: dict, answer: str) -> Verdict:
    got = count_words(answer)
    if got == int(check["value"]):
        return Verdict(PASS)
    return Verdict(FAIL, f"{got} words, expected {check['value']}")


def _line_count(check: dict, answer: str) -> Verdict:
    got = count_lines(answer)
    if got == int(check["value"]):
        return Verdict(PASS)
    return Verdict(FAIL, f"{got} lines, expected {check['value']}")


def _sentence_count(check: dict, answer: str) -> Verdict:
    got = count_sentences(answer)
    if got == int(check["value"]):
        return Verdict(PASS)
    return Verdict(FAIL, f"{got} sentences, expected {check['value']}")


def _json_keys(check: dict, answer: str) -> Verdict:
    parsed = _parse_json(answer)
    if parsed is None:
        return Verdict(FAIL, "response is not valid JSON")
    if not isinstance(parsed, dict):
        return Verdict(FAIL, f"expected a JSON object, got {type(parsed).__name__}")
    missing = [key for key in check["keys"] if key not in parsed]
    if missing:
        return Verdict(FAIL, f"missing keys: {', '.join(missing)}")
    return Verdict(PASS)


def _json_array_len(check: dict, answer: str) -> Verdict:
    parsed = _parse_json(answer)
    if parsed is None:
        return Verdict(FAIL, "response is not valid JSON")
    if not isinstance(parsed, list):
        return Verdict(FAIL, f"expected a JSON array, got {type(parsed).__name__}")
    if len(parsed) != int(check["value"]):
        return Verdict(FAIL, f"array of {len(parsed)}, expected {check['value']}")
    return Verdict(PASS)


def _parse_json(answer: str):
    try:
        return json.loads(strip_code_fence(answer))
    except (json.JSONDecodeError, TypeError):
        return None


_HANDLERS = {
    "contains_any": _contains_any,
    "exact": _exact,
    "starts_with": _starts_with,
    "number": _number,
    "digits_only": _digits_only,
    "word_count": _word_count,
    "line_count": _line_count,
    "sentence_count": _sentence_count,
    "json_keys": _json_keys,
    "json_array_len": _json_array_len,
}
