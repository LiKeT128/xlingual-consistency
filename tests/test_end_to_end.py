"""The whole pipeline, offline, in CI.

Every bug that reached a user so far was invisible to unit tests and obvious
the moment the commands were actually run: a wrong model id that produced 200
error records while reporting success, a rate budget that throttled itself into
failure, a status line that never redrew, a preflight call that ran before any
output. This test runs collect -> grade -> report against the mock provider so
that class of defect fails here instead of in someone's terminal.
"""

from __future__ import annotations

import pytest

from xlingual import LANGUAGES
from xlingual.config import Config
from xlingual.dataset import load_items
from xlingual.report import build_report
from xlingual.runner import (
    collect_answers,
    grade_answers,
    load_answers,
    load_grades,
    results_dir,
)


@pytest.fixture()
def mock_config(monkeypatch, tmp_path):
    monkeypatch.setenv("XLC_PROVIDER", "mock")
    monkeypatch.setenv("XLC_MODEL", "mock")
    monkeypatch.setenv("XLC_RPM", "6000")
    monkeypatch.setenv("XLC_CONCURRENCY", "8")
    monkeypatch.setattr("xlingual.runner.RESULTS_DIR", tmp_path)
    return Config.from_env()


def test_the_full_pipeline_produces_a_report(mock_config, capsys):
    items = load_items()

    collect_answers(mock_config, items)
    out = capsys.readouterr().out

    # The header must be printed before the first network call, so the command
    # never sits silent while something slow happens.
    assert out.index("Collecting 200 answers") < out.index("Checking model")
    assert "no failures" in out

    answers = load_answers(mock_config)
    assert len(answers) == len(items) * len(LANGUAGES) == 200
    assert all(row["answer"] and not row["error"] for row in answers)

    grade_answers(mock_config, items)
    out = capsys.readouterr().out
    assert "Graded 152 answers deterministically" in out
    assert "Judging 48 answers" in out

    grades = load_grades(mock_config)
    assert len(grades) == 200
    assert {g["status"] for g in grades} <= {"pass", "fail", "error"}
    assert sum(1 for g in grades if g["judged"]) == 48

    report = build_report(
        mock_config.model, mock_config.provider, items, grades, answers, []
    )
    assert "# Cross-lingual consistency: `mock`" in report
    assert "Cross-lingual consistency:" in report
    assert "## Accuracy by category" in report
    for category in ("factual", "arithmetic", "instruction", "refusal",
                     "ambiguity", "geopolitical"):
        assert category in report


def test_rerunning_collect_is_a_no_op(mock_config, capsys):
    items = load_items()[:3]
    collect_answers(mock_config, items)
    capsys.readouterr()

    collect_answers(mock_config, items)
    out = capsys.readouterr().out
    assert "already collected" in out
    # No duplicate rows: a second pass must not re-ask what it already has.
    assert len(load_answers(mock_config)) == 3 * len(LANGUAGES)


def test_grading_is_resumable_and_does_not_double_count(mock_config, capsys):
    items = load_items()[:5]
    collect_answers(mock_config, items)
    grade_answers(mock_config, items)
    first = len(load_grades(mock_config))
    capsys.readouterr()

    grade_answers(mock_config, items)
    assert "already graded" in capsys.readouterr().out
    assert len(load_grades(mock_config)) == first


def test_no_judge_leaves_the_judged_items_ungraded(mock_config, capsys):
    all_items = load_items()
    items = [i for i in all_items if i.category == "refusal"][:2]
    items += [i for i in all_items if i.category == "factual"][:2]
    assert any(i.needs_judge for i in items), "the slice must contain judged items"

    collect_answers(mock_config, items)
    grade_answers(mock_config, items, use_judge=False)

    out = capsys.readouterr().out
    assert "Skipping 8 judged answers" in out
    grades = load_grades(mock_config)
    assert all(not g["judged"] for g in grades)
    # Only the deterministic half was graded; the judged items wait for a run
    # with the judge enabled.
    assert len(grades) == 2 * len(LANGUAGES)


def test_results_are_kept_per_model(mock_config, tmp_path):
    assert results_dir(mock_config) == tmp_path / "mock"
