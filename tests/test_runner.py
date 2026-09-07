import pytest

from xlingual.runner import report_failures


def test_clean_run_reports_no_failures(capsys):
    report_failures(200, [])
    assert "no failures" in capsys.readouterr().out


def test_partial_failure_warns_but_continues(capsys):
    report_failures(200, ["fact_001/ru: HTTP 429"] * 5)
    out = capsys.readouterr().out
    assert "WARNING: 5 of 200" in out
    assert "and 2 more" in out
    assert "Rerun the same command" in out


def test_total_failure_stops_the_pipeline(capsys):
    # A run where nothing succeeded must not look like a successful run:
    # this is what let a wrong model id produce 200 error records silently.
    with pytest.raises(SystemExit, match="nothing to grade"):
        report_failures(3, ["a: 404", "b: 404", "c: 404"])
    assert "WARNING: 3 of 3" in capsys.readouterr().out
