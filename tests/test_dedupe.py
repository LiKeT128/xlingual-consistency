from xlingual.runner import dedupe_answers


def row(item_id, language, answer="", error=""):
    return {"item_id": item_id, "language": language, "answer": answer, "error": error}


def test_a_successful_retry_replaces_the_stored_error():
    rows = [row("fact_001", "ru", error="HTTP 429"), row("fact_001", "ru", answer="Bratislava")]
    result = dedupe_answers(rows)
    assert len(result) == 1
    assert result[0]["answer"] == "Bratislava"


def test_an_error_never_overwrites_a_good_answer():
    rows = [row("fact_001", "ru", answer="Bratislava"), row("fact_001", "ru", error="HTTP 500")]
    result = dedupe_answers(rows)
    assert len(result) == 1
    assert result[0]["answer"] == "Bratislava"


def test_distinct_prompts_are_all_kept():
    rows = [row("a", "en", "x"), row("a", "ru", "y"), row("b", "en", "z")]
    assert len(dedupe_answers(rows)) == 3


def test_the_later_of_two_successes_wins():
    rows = [row("a", "en", "old"), row("a", "en", "new")]
    assert dedupe_answers(rows)[0]["answer"] == "new"
