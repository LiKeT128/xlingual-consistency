from xlingual.normalize import (
    count_lines,
    count_sentences,
    count_words,
    extract_numbers,
    normalize,
    strip_code_fence,
)


def test_cyrillic_and_latin_collapse_to_the_same_form():
    assert normalize("Братислава") == normalize("Bratislava")


def test_russian_and_ukrainian_spellings_of_ukraine_match():
    assert normalize("Украина") == normalize("Україна") == "ukraina"


def test_slovak_diacritics_are_stripped():
    assert normalize("Tichý oceán") == "tichy ocean"


def test_punctuation_and_case_are_ignored():
    assert normalize("  Bratislava!  ") == "bratislava"


def test_time_separators_collapse_to_one_form():
    # Punctuation becomes a space rather than vanishing, so "17:25" and "17.25"
    # land on the same string and either written form of a time matches.
    assert normalize("17:25") == normalize("17.25") == "17 25"


def test_word_count_ignores_punctuation():
    assert count_words("red, green and blue.") == 4
    assert count_words("") == 0


def test_line_count_skips_blank_lines():
    assert count_lines("a\n\nb\nc\n") == 3


def test_sentence_count():
    assert count_sentences("One sentence only.") == 1
    assert count_sentences("First. Second.") == 2
    assert count_sentences("No terminator here") == 1
    assert count_sentences("") == 0


def test_extract_numbers_handles_comma_decimals():
    assert extract_numbers("Итого 8,40 EUR") == [8.4]
    assert extract_numbers("no digits") == []


def test_strip_code_fence_removes_wrapper():
    fenced = '```json\n{"a": 1}\n```'
    assert strip_code_fence(fenced) == '{"a": 1}'
