import json

import pytest

from xlingual import LANGUAGES
from xlingual.dataset import DatasetError, categories, load_items

VALID = {
    "id": "x1",
    "category": "factual",
    "check": {"type": "contains_any", "accept": ["a"]},
    "prompts": {"en": "q", "ru": "q", "uk": "q", "sk": "q"},
}


def write(tmp_path, *records):
    path = tmp_path / "set.jsonl"
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records),
        encoding="utf-8",
    )
    return path


def test_the_shipped_dataset_is_valid():
    items = load_items()
    assert len(items) >= 40
    for item in items:
        assert set(item.prompts) == set(LANGUAGES)
        assert all(item.prompts[lang].strip() for lang in LANGUAGES)


def test_the_shipped_dataset_covers_every_category():
    items = load_items()
    assert set(categories(items)) == {
        "factual",
        "arithmetic",
        "instruction",
        "refusal",
        "ambiguity",
        "geopolitical",
    }


def test_judged_items_carry_a_rubric_or_are_refusals():
    for item in load_items():
        if item.check["type"] == "judge":
            assert item.check.get("rubric"), f"{item.id} has no rubric"


def test_missing_language_is_rejected(tmp_path):
    broken = dict(VALID, prompts={"en": "q", "ru": "q", "uk": "q"})
    with pytest.raises(DatasetError, match="missing prompts"):
        load_items(write(tmp_path, broken))


def test_unknown_check_type_is_rejected(tmp_path):
    broken = dict(VALID, check={"type": "vibes"})
    with pytest.raises(DatasetError, match="unknown check type"):
        load_items(write(tmp_path, broken))


def test_duplicate_ids_are_rejected(tmp_path):
    with pytest.raises(DatasetError, match="duplicate id"):
        load_items(write(tmp_path, VALID, VALID))


def test_empty_file_is_rejected(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(DatasetError, match="no items"):
        load_items(path)
