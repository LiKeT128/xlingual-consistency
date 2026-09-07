"""Turn graded answers into a Markdown report."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from . import LANGUAGE_NAMES, LANGUAGES
from .dataset import Item, categories
from .metrics import (
    accuracy_by_language,
    accuracy_by_language_and_category,
    agreement_with_english,
    cohens_kappa,
    consistency,
    consistency_by_category,
    divergent_items,
    kappa_label,
)


def build_report(
    model: str,
    provider: str,
    items: list[Item],
    grades: list[dict],
    answers: list[dict],
    human_labels: list[dict] | None = None,
) -> str:
    languages = LANGUAGES
    lines: list[str] = []
    add = lines.append

    add(f"# Cross-lingual consistency: `{model}`")
    add("")
    add(f"Provider: `{provider}` · Items: {len(items)} · "
        f"Languages: {', '.join(LANGUAGE_NAMES[l] for l in languages)} · "
        f"Run date: {date.today().isoformat()}")
    add("")

    overall = consistency(grades, languages)
    add("## Headline")
    add("")
    add(f"- **Cross-lingual consistency: {overall}** — share of items where all four "
        f"languages produced the same verdict.")
    for lang, rate in sorted(agreement_with_english(grades, languages).items()):
        add(f"- Agreement with English, {LANGUAGE_NAMES[lang]}: {rate}")
    add("")

    add("## Accuracy by language")
    add("")
    add("| Language | Accuracy (95% CI) |")
    add("| --- | --- |")
    by_lang = accuracy_by_language(grades)
    for lang in languages:
        rate = by_lang.get(lang)
        add(f"| {LANGUAGE_NAMES[lang]} | {rate if rate else 'n/a'} |")
    add("")

    add("## Accuracy by category")
    add("")
    header = "| Category | " + " | ".join(LANGUAGE_NAMES[l] for l in languages) + " | Consistency |"
    add(header)
    add("| --- | " + " | ".join("---" for _ in languages) + " | --- |")
    cell = accuracy_by_language_and_category(grades)
    cons = consistency_by_category(grades, languages)
    for category in categories(items):
        row = [category]
        for lang in languages:
            rate = cell.get((category, lang))
            row.append(f"{rate.value:.0%}" if rate and rate.total else "n/a")
        consistency_rate = cons.get(category)
        row.append(f"{consistency_rate.value:.0%}" if consistency_rate and consistency_rate.total else "n/a")
        add("| " + " | ".join(row) + " |")
    add("")

    if human_labels:
        pairs = _pairs(grades, human_labels)
        kappa = cohens_kappa(pairs)
        add("## Judge reliability")
        add("")
        add(f"A human annotator labelled {len(pairs)} judged answers independently. "
            f"Cohen's kappa between the model judge and the human is "
            f"**{kappa:.2f}** ({kappa_label(kappa)}).")
        add("")
        add("Judged categories (refusal, ambiguity) should be read with that number in "
            "mind. The deterministic categories do not depend on the judge at all.")
        add("")
    else:
        add("## Judge reliability")
        add("")
        add("_Not measured yet._ Run `xlc annotate` to label a sample by hand, then "
            "regenerate this report. Judged categories (refusal, ambiguity) carry an "
            "unquantified error until you do.")
        add("")

    divergences = divergent_items(grades, languages)
    add("## Where the languages disagree")
    add("")
    if not divergences:
        add("No item produced different verdicts across languages.")
    else:
        add(f"{len(divergences)} of {len(items)} items split across languages.")
        add("")
        answer_index = {(a["item_id"], a["language"]): a for a in answers}
        item_index = {i.id: i for i in items}
        for entry in divergences[:12]:
            item = item_index.get(entry["item_id"])
            add(f"### `{entry['item_id']}` ({entry['category']})")
            add("")
            add(f"Passed in: {', '.join(entry['passed']) or 'none'} · "
                f"Failed in: {', '.join(entry['failed']) or 'none'}")
            add("")
            for lang in entry["failed"]:
                record = answer_index.get((entry["item_id"], lang))
                if not record:
                    continue
                if item:
                    add(f"- **{LANGUAGE_NAMES[lang]} prompt:** {item.prompts[lang]}")
                answer_text = " ".join(record["answer"].split())[:280]
                add(f"- **Answer:** {answer_text}")
                add(f"- **Grader said:** {entry['details'].get(lang, '')}")
                add("")
    add("")

    add("## How to read this")
    add("")
    add("- Accuracy is measured against a fixed reference answer, not against the "
        "English answer, so a model that is wrong in all four languages scores low "
        "rather than 'consistent'.")
    add("- Consistency is measured separately, so a model can be consistently wrong "
        "and the two numbers will show it.")
    add("- Intervals are Wilson 95%. With this many items per cell they are wide; "
        "differences smaller than the interval are not findings.")
    return "\n".join(lines) + "\n"


def _pairs(grades: list[dict], human_labels: list[dict]) -> list[tuple[str, str]]:
    judge_index = {
        (row["item_id"], row["language"]): row["status"]
        for row in grades
        if row.get("judged")
    }
    pairs = []
    for label in human_labels:
        key = (label["item_id"], label["language"])
        if key in judge_index:
            pairs.append((judge_index[key], label["status"]))
    return pairs


def write_report(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def load_human_labels(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
