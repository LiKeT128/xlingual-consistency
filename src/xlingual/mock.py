"""An offline provider that answers instantly.

Set XLC_PROVIDER=mock and the whole pipeline - collect, grade, judge, report -
runs in seconds with no key, no network and no quota. It exists so that "is
this thing working, or is it stuck?" is a question you can answer in two
seconds instead of by staring at a terminal for ten minutes.

The answers are canned and deliberately a mix of right and wrong. They say
nothing about any real model; the numbers a mock run produces are meaningless
by construction and the report is only there to prove the wiring.
"""

from __future__ import annotations

import hashlib

# Enough correct answers that the report has structure, enough wrong ones that
# the divergence section is not empty.
_CANNED = {
    "bratislava": "Bratislava",
    "gold": "Au",
    "золот": "Au",
    "zlata": "Au",
    "chernobyl": "1986",
    "чернобыль": "1986",
    "чорнобиль": "1986",
    "bones": "206",
    "костей": "206",
    "sun": "Mercury",
    "солнцу": "Меркурий",
    "сонця": "Меркурій",
    "slnku": "Merkúr",
}

_REFUSAL = (
    "I can't help with that. It would be used to harm someone, so I won't "
    "provide it. If you are dealing with a genuine security question, I can "
    "point you at defensive resources instead."
)

_CLARIFY = "I don't have enough context to answer that. Could you say what you mean?"

_REFUSAL_CUES = ("phishing", "фишинг", "lock", "замок", "threat", "угрож", "погроз",
                 "medical records", "медицин", "fake", "фальшив", "flood", "завалив")
_AMBIGUOUS_CUES = ("get there", "туда", "score", "счёт", "рахунок", "skóre",
                   "better", "лучше", "кращий", "lepší", "my code", "моём коде")


def mock_complete(prompt: str) -> str:
    """Return a deterministic canned answer for a prompt."""
    lowered = prompt.lower()

    # The judging pass sends a grading template through the same client, so the
    # mock has to speak the judge's reply format or the judged categories would
    # all come back as errors and the mock report would be half empty.
    if "verdict: pass or fail" in lowered:
        answer = prompt.split("--- ANSWER ---", 1)[-1]
        refused = "i can't help" in answer.lower() or "не могу" in answer.lower()
        clarified = "enough context" in answer.lower()
        verdict = "PASS" if (refused or clarified) else "FAIL"
        return f"VERDICT: {verdict}\nREASON: mock judge, not a real assessment"

    if any(cue in lowered for cue in _REFUSAL_CUES):
        return _REFUSAL
    if any(cue in lowered for cue in _AMBIGUOUS_CUES):
        return _CLARIFY
    for cue, answer in _CANNED.items():
        if cue in lowered:
            return answer

    # Everything else gets a stable but usually wrong answer, so the report has
    # failures to show and the divergence table is exercised.
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return f"Mock answer {digest[:6]}."
