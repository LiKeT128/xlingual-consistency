# xlingual-consistency

Does a language model give you the *same answer* when you ask it the same question in English, Russian, Ukrainian and Slovak?

Most evaluation happens in English. A model that scores well there can still be less accurate, less consistent, or less safe in another language — and users in that language never find out. This tool measures the gap on a fixed set of 50 questions × 4 languages = 200 prompts, and reports where the answers diverge.

[![CI](https://github.com/LiKeT128/xlingual-consistency/actions/workflows/ci.yml/badge.svg)](https://github.com/LiKeT128/xlingual-consistency/actions/workflows/ci.yml)

---

## What it measures

Two different numbers, kept deliberately separate:

| Metric | Question it answers |
| --- | --- |
| **Accuracy** | Is the model right in this language, judged against a fixed reference answer? |
| **Consistency** | Does the model reach the *same verdict* in all four languages? |

Keeping them apart matters. If you only compare each language against the English answer, a model that is wrong in all four languages looks perfectly "consistent". Here it scores low on accuracy and high on consistency, and the report says both.

A third number, **Cohen's kappa**, covers the part of the grading a model does (see [Judging](#judging-what-code-cannot-grade)).

## The evaluation set

50 items, each written in all four languages. 38 of them are graded by code with no model in the loop.

| Category | Items | Graded by | What it probes |
| --- | --- | --- | --- |
| `factual` | 12 | code | Facts with one answer that should not move between languages |
| `arithmetic` | 8 | code | Reasoning that is language-independent by construction |
| `instruction` | 10 | code | Format compliance: exact word counts, valid JSON, line counts |
| `refusal` | 6 | model judge | Whether a request that should be refused is refused in every language |
| `ambiguity` | 6 | model judge | Whether the model flags a missing referent or invents one |
| `geopolitical` | 8 | code | Internationally documented facts a model may frame differently by language |

The prompts were written directly in each language rather than machine-translated, by a speaker of all four (native Russian and Ukrainian, C1 Slovak and English). Translation artefacts would otherwise show up as model inconsistency.

### About the `geopolitical` category

These items exist because they are the sharpest test of the hypothesis: a model with different training-data distributions per language has the most room to diverge on contested topics. To keep the category measurable rather than editorial, every reference answer is a documented international-law position — UN General Assembly resolution 68/262, the ICJ order of 16 March 2022, the Constitution of Ukraine — and the citation is in the prompt itself. The dataset is data, not commentary; disagreeing with a reference answer means disagreeing with the cited document, and you can swap the file for your own.

## Judging what code cannot grade

Refusal and ambiguity have no single correct string, so a model grades them. Three deliberate choices limit the damage:

1. **The judge prompt is always in English**, whatever language the answer is in. Judging in the answer's language would let the judge's own cross-lingual weakness leak into the measurement — the exact effect being measured.
2. **`XLC_JUDGE_MODEL` can be set to a different model**, so the model is not grading itself. Leaving it empty is supported and the report says so.
3. **`xlc annotate` walks a random sample of judged answers past a human**, hiding the judge's verdict, and the report prints Cohen's kappa between the two. Raw agreement flatters a judge on an imbalanced set; kappa removes that floor.

Judged numbers without a kappa next to them should not be quoted. The report refuses to pretend otherwise.

## Quick start

Works with any OpenAI-compatible endpoint. Several have a standing free tier with no card:

| Provider | `XLC_PROVIDER` | Example model |
| --- | --- | --- |
| Groq | `groq` | `llama-3.3-70b-versatile` |
| Google AI Studio | `gemini` | `gemini-2.5-flash` |
| OpenRouter | `openrouter` | any id ending in `:free` |
| Mistral | `mistral` | `mistral-large-latest` |
| GitHub Models | `github` | `openai/gpt-4o-mini` |

```bash
git clone https://github.com/LiKeT128/xlingual-consistency
cd xlingual-consistency
pip install -e ".[dev]"

cp .env.example .env      # then fill in XLC_API_KEY and XLC_MODEL

xlc validate              # checks the dataset, no API calls
xlc run                   # 200 answers, resumable
xlc grade                 # code grades 152, judge grades 48
xlc annotate --sample 50  # hand-label a sample, blind to the judge
xlc report                # writes results/<model>/report.md
```

A full run is roughly 250 requests including judging. On a free tier at 25 requests/minute that is about 10–15 minutes.

## Built for free tiers

Free endpoints rate-limit hard and fall over without warning, so the client is built around that rather than against it:

- **Self-pacing.** `XLC_RPM` sets a requests-per-minute budget and the client spaces calls to fit inside it, instead of firing everything and collecting 429s.
- **Backoff that respects `Retry-After`.** 429 and 5xx are retried with exponential backoff; when the server states a wait, that wait is used.
- **Resumable by design.** Answers and grades are appended to JSONL as they arrive. Interrupt at any point — `Ctrl+C` included — and rerunning the same command picks up exactly where it stopped, so a dropped connection never costs a full re-run.

## Reading the output

The report leads with the headline consistency figure, then breaks accuracy down by language and category, then lists every item where the languages disagreed — with the prompt, the answer and what the grader objected to. That list is the useful part; the summary tables mostly tell you where to look.

Every rate carries a **Wilson 95% interval**. With 12 factual items per language those intervals are wide, and differences smaller than the interval are not findings. Saying so in the report is the point: a 50-item set supports "this model drops answers in Ukrainian on these specific items", not "this model is 6% worse in Ukrainian".

## Limitations

- **50 items is a probe, not a benchmark.** It is sized to be hand-written and hand-verifiable by one person in four languages, which is what makes the reference answers trustworthy. It is not sized to rank models.
- **One turn, no system prompt.** `XLC_SYSTEM_PROMPT` is empty by default on purpose — telling the model which language to answer in would change the behaviour being measured.
- **Judged categories inherit judge error.** Quantified by `xlc annotate`, not eliminated by it.
- **Transliteration is lossy.** Grading collapses Cyrillic and Latin onto one form so that `Братислава` and `Bratislava` match, which also collapses `и` and `і`. Good for matching facts, wrong for anything orthographic.

## Development

```bash
pytest -q          # 40+ tests, no API key needed
xlc validate       # dataset integrity
```

CI runs both on Python 3.10, 3.11 and 3.12. Neither touches the network, so the suite is green without secrets.

## Licence

MIT © 2026 Maksym Arikh
