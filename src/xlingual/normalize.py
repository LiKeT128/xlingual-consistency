"""Text normalisation shared by every grader.

Answers come back in four languages and two scripts, so a naive string
comparison fails constantly: the same fact is written "Bratislava",
"Братислава" and "Bratislave". Everything - both the model's answer and the
accepted forms in the dataset - is pushed through :func:`normalize` before
comparison, so the two sides are always compared in the same shape.
"""

from __future__ import annotations

import re
import unicodedata

# Cyrillic -> Latin. Deliberately lossy: "и" and "і" both become "i", so a
# Russian and a Ukrainian spelling of the same name collapse onto one form.
# That is the point - we are matching facts, not transliterating properly.
_CYRILLIC = {
    "а": "a", "б": "b", "в": "v", "г": "g", "ґ": "g", "д": "d", "е": "e",
    "ё": "e", "є": "e", "ж": "zh", "з": "z", "и": "i", "і": "i", "ї": "i",
    "й": "i", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p",
    "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "c",
    "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "", "э": "e",
    "ю": "iu", "я": "ia",
}

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")
_SENTENCE_END = re.compile(r"[.!?…]+(?:\s|$)")
_CODE_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")


def strip_accents(text: str) -> str:
    """Drop combining marks so 'Tichý' and 'Tichy' compare equal."""
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def transliterate(text: str) -> str:
    """Map Cyrillic letters onto Latin ones, leaving everything else alone."""
    return "".join(_CYRILLIC.get(ch, ch) for ch in text)


def normalize(text: str) -> str:
    """Lowercase, de-accent, transliterate, drop punctuation, squeeze spaces."""
    if not text:
        return ""
    lowered = text.lower()
    lowered = strip_accents(lowered)
    lowered = transliterate(lowered)
    lowered = _PUNCT.sub(" ", lowered)
    return _WHITESPACE.sub(" ", lowered).strip()


def strip_code_fence(text: str) -> str:
    """Remove a wrapping ``` fence, which models add even when told not to."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines)
    return cleaned.strip()


def count_words(text: str) -> int:
    normalized = normalize(text)
    return len(normalized.split()) if normalized else 0


def count_lines(text: str) -> int:
    return len([line for line in text.strip().splitlines() if line.strip()])


def count_sentences(text: str) -> int:
    """Count sentence terminators in the raw text, before punctuation is dropped."""
    stripped = text.strip()
    if not stripped:
        return 0
    hits = _SENTENCE_END.findall(stripped)
    if not hits:
        return 1  # a single sentence with no final period still counts as one
    # A terminator that is not at the very end still closes a sentence; if the
    # text does not end with one, the trailing fragment is a sentence too.
    trailing = not _SENTENCE_END.search(stripped[-2:] + " ")
    return len(hits) + (1 if trailing else 0)


_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")


def extract_numbers(text: str) -> list[float]:
    """Pull every number out of a response, tolerating comma decimals."""
    values: list[float] = []
    for raw in _NUMBER.findall(text.replace(" ", " ")):
        try:
            values.append(float(raw.replace(",", ".")))
        except ValueError:
            continue
    return values
