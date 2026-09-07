"""Metrics.

Three numbers matter here and they answer different questions:

* accuracy      - is the model right in this language?
* consistency   - does it give the same verdict in all four languages?
* Cohen's kappa - can the judge that produced part of the accuracy be trusted?

Accuracy without a confidence interval is misleading on a set this size, so
every rate is reported with a Wilson interval. With 12 factual items per
language, the interval is wide - saying so is the honest part.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from .graders import FAIL, PASS


@dataclass(frozen=True)
class Rate:
    passed: int
    total: int

    @property
    def value(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.passed, self.total)

    def __str__(self) -> str:
        if not self.total:
            return "n/a"
        low, high = self.interval
        return f"{self.value:.0%} [{low:.0%}-{high:.0%}] n={self.total}"


def wilson_interval(passed: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval - behaves sensibly at small n, unlike the normal one."""
    if total == 0:
        return (0.0, 0.0)
    phat = passed / total
    denominator = 1 + z**2 / total
    centre = phat + z**2 / (2 * total)
    margin = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * total)) / total)
    return (
        max(0.0, (centre - margin) / denominator),
        min(1.0, (centre + margin) / denominator),
    )


def accuracy_by_language(grades: list[dict]) -> dict[str, Rate]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in grades:
        if row["status"] not in (PASS, FAIL):
            continue
        counts[row["language"]][1] += 1
        if row["status"] == PASS:
            counts[row["language"]][0] += 1
    return {lang: Rate(p, t) for lang, (p, t) in counts.items()}


def accuracy_by_language_and_category(grades: list[dict]) -> dict[tuple[str, str], Rate]:
    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for row in grades:
        if row["status"] not in (PASS, FAIL):
            continue
        key = (row["category"], row["language"])
        counts[key][1] += 1
        if row["status"] == PASS:
            counts[key][0] += 1
    return {key: Rate(p, t) for key, (p, t) in counts.items()}


def consistency(grades: list[dict], languages: tuple[str, ...]) -> Rate:
    """Share of items where every language produced the same verdict."""
    by_item: dict[str, dict[str, str]] = defaultdict(dict)
    for row in grades:
        if row["status"] in (PASS, FAIL):
            by_item[row["item_id"]][row["language"]] = row["status"]

    complete = [v for v in by_item.values() if len(v) == len(languages)]
    agreed = sum(1 for v in complete if len(set(v.values())) == 1)
    return Rate(agreed, len(complete))


def consistency_by_category(grades: list[dict], languages: tuple[str, ...]) -> dict[str, Rate]:
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in grades:
        by_category[row["category"]].append(row)
    return {cat: consistency(rows, languages) for cat, rows in by_category.items()}


def agreement_with_english(grades: list[dict], languages: tuple[str, ...]) -> dict[str, Rate]:
    """For each language, how often its verdict matches the English verdict."""
    by_item: dict[str, dict[str, str]] = defaultdict(dict)
    for row in grades:
        if row["status"] in (PASS, FAIL):
            by_item[row["item_id"]][row["language"]] = row["status"]

    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for verdicts in by_item.values():
        english = verdicts.get("en")
        if english is None:
            continue
        for lang in languages:
            if lang == "en" or lang not in verdicts:
                continue
            counts[lang][1] += 1
            if verdicts[lang] == english:
                counts[lang][0] += 1
    return {lang: Rate(p, t) for lang, (p, t) in counts.items()}


def divergent_items(grades: list[dict], languages: tuple[str, ...]) -> list[dict]:
    """Items where the model passed in some languages and failed in others."""
    by_item: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in grades:
        if row["status"] in (PASS, FAIL):
            by_item[row["item_id"]][row["language"]] = row

    out = []
    for item_id, verdicts in sorted(by_item.items()):
        statuses = {row["status"] for row in verdicts.values()}
        if len(verdicts) == len(languages) and len(statuses) > 1:
            out.append(
                {
                    "item_id": item_id,
                    "category": next(iter(verdicts.values()))["category"],
                    "passed": sorted(l for l, r in verdicts.items() if r["status"] == PASS),
                    "failed": sorted(l for l, r in verdicts.items() if r["status"] == FAIL),
                    "details": {l: r["detail"] for l, r in sorted(verdicts.items())},
                }
            )
    return out


def cohens_kappa(pairs: list[tuple[str, str]]) -> float:
    """Chance-corrected agreement between two raters over the same items.

    Raw agreement flatters a judge on an imbalanced set: if 90% of answers pass,
    a rater that always says PASS scores 90%. Kappa removes that floor.
    """
    if not pairs:
        return float("nan")

    n = len(pairs)
    labels = sorted({label for pair in pairs for label in pair})
    observed = sum(1 for a, b in pairs if a == b) / n

    expected = 0.0
    for label in labels:
        p_a = sum(1 for a, _ in pairs if a == label) / n
        p_b = sum(1 for _, b in pairs if b == label) / n
        expected += p_a * p_b

    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def kappa_label(kappa: float) -> str:
    """Landis & Koch bands, so the number is readable without the paper."""
    if kappa != kappa:  # NaN
        return "not measured"
    if kappa < 0.0:
        return "worse than chance"
    if kappa < 0.20:
        return "slight"
    if kappa < 0.40:
        return "fair"
    if kappa < 0.60:
        return "moderate"
    if kappa < 0.80:
        return "substantial"
    return "almost perfect"
