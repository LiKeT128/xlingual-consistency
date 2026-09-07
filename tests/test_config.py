import pytest

from xlingual.config import env_float, env_int, env_str


def test_missing_variable_falls_back_to_the_default(monkeypatch):
    monkeypatch.delenv("XLC_RPM", raising=False)
    assert env_int("XLC_RPM", 15) == 15


def test_an_empty_value_is_treated_as_absent(monkeypatch):
    # "XLC_RPM=" in a .env file sets the variable to "", which os.environ.get
    # returns instead of the default. Deleting a value means "use the default".
    monkeypatch.setenv("XLC_RPM", "")
    assert env_int("XLC_RPM", 15) == 15
    assert env_str("XLC_RPM", "fallback") == "fallback"


def test_whitespace_only_is_treated_as_absent(monkeypatch):
    monkeypatch.setenv("XLC_RPM", "   ")
    assert env_int("XLC_RPM", 15) == 15


def test_a_real_value_is_used(monkeypatch):
    monkeypatch.setenv("XLC_RPM", " 10 ")
    assert env_int("XLC_RPM", 15) == 10


def test_garbage_gives_a_readable_error_not_a_traceback(monkeypatch):
    monkeypatch.setenv("XLC_RPM", "ten")
    with pytest.raises(SystemExit) as excinfo:
        env_int("XLC_RPM", 15)
    message = str(excinfo.value)
    assert "must be a whole number" in message
    assert "'ten'" in message
    assert "default (15)" in message


def test_float_settings_behave_the_same(monkeypatch):
    monkeypatch.setenv("XLC_TEMPERATURE", "")
    assert env_float("XLC_TEMPERATURE", 0.0) == 0.0
    monkeypatch.setenv("XLC_TEMPERATURE", "0.7")
    assert env_float("XLC_TEMPERATURE", 0.0) == 0.7
    monkeypatch.setenv("XLC_TEMPERATURE", "hot")
    with pytest.raises(SystemExit, match="must be a number"):
        env_float("XLC_TEMPERATURE", 0.0)
