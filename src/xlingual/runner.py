"""Collect model answers and grade them.

Both stages append to JSONL files under results/ and skip work that is already
recorded, so an interrupted run - which on a free tier is the normal case, not
the exception - is resumed by running the same command again.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from . import LANGUAGES
from .client import ChatClient, ProviderError
from .config import Config, RESULTS_DIR
from .dataset import Item
from .graders import ERROR, NEEDS_JUDGE, grade
from .judge import judge_answer
from .progress import Progress, format_elapsed
from .ratelimit import estimate_seconds, format_duration

ANSWERS_FILE = "answers.jsonl"
GRADES_FILE = "grades.jsonl"


@dataclass
class Answer:
    item_id: str
    language: str
    model: str
    prompt: str
    answer: str
    error: str = ""


@dataclass
class Grade:
    item_id: str
    language: str
    category: str
    model: str
    status: str
    detail: str
    judged: bool


def _results_dir(config: Config) -> Path:
    slug = "".join(c if c.isalnum() or c in "-._" else "_" for c in config.model)
    path = RESULTS_DIR / slug
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _append(path: Path, record) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def dedupe_answers(rows: list[dict]) -> list[dict]:
    """One row per (item, language), preferring a real answer over an error.

    Retrying a failed run appends a second row for the same prompt, so without
    this the same prompt would be graded twice and counted twice - once as the
    error it was and once as the answer it became.
    """
    best: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (row["item_id"], row["language"])
        current = best.get(key)
        if current is None or (current.get("error") and not row.get("error")):
            best[key] = row
        elif not current.get("error") and not row.get("error"):
            best[key] = row  # later successful attempt wins
    return list(best.values())


def preflight(config: Config, client: ChatClient) -> None:
    """Spend one request proving the model id works before spending two hundred.

    This is a network call, and on a throttled key it can retry for a while, so
    it announces itself first and ticks while it waits. A silent check is
    indistinguishable from a hung program - which is precisely how it read
    before this printed anything.
    """
    print(f"Checking model '{config.model}' on {config.provider}...", end="", flush=True)
    started = time.monotonic()
    stop = threading.Event()

    def tick() -> None:
        while not stop.wait(1.0):
            sys.stdout.write(f"\r  Checking model '{config.model}' on {config.provider}... "
                             f"{format_elapsed(time.monotonic() - started)}   ")
            sys.stdout.flush()

    ticker = threading.Thread(target=tick, daemon=True)
    ticker.start()
    try:
        client.complete("Reply with the single word: ok")
    except ProviderError as exc:
        stop.set()
        ticker.join(timeout=2)
        raise SystemExit(
            f"\n\nPreflight failed for model '{config.model}' on provider "
            f"'{config.provider}':\n\n  {exc}\n\n"
            "Run 'xlc models' to see the ids this key can actually call, then fix "
            "XLC_MODEL in .env."
        ) from exc
    finally:
        stop.set()
        ticker.join(timeout=2)
    print(f"\r  Model '{config.model}' responded in "
          f"{format_elapsed(time.monotonic() - started)}." + " " * 20, flush=True)


def collect_answers(config: Config, items: list[Item], *, languages=LANGUAGES) -> Path:
    """Ask the model every prompt in every language, skipping what we already have."""
    out_dir = _results_dir(config)
    path = out_dir / ANSWERS_FILE
    done = {(row["item_id"], row["language"]) for row in _read_jsonl(path) if not row.get("error")}

    client = ChatClient(config)
    pending = [(item, lang) for item in items for lang in languages if (item.id, lang) not in done]

    # Everything the user needs to see is printed before the first network call,
    # so the command never sits silent while something slow happens.
    workers = min(config.concurrency, max(len(pending), 1))
    estimate = estimate_seconds(len(pending), config.requests_per_minute, workers=workers)
    print(f"Collecting {len(pending)} answers with {config.model} ({config.provider})", flush=True)
    if done:
        print(f"  {len(done)} already collected, skipping those", flush=True)
    print(f"  {workers} workers, {config.requests_per_minute} req/min budget "
          f"-> about {format_duration(estimate)}", flush=True)

    if not pending:
        print("Nothing to do: every answer is already collected.", flush=True)
        return path

    preflight(config, client)

    failures: list[str] = []
    write_lock = threading.Lock()

    with Progress(len(pending), limiter=client.limiter) as progress:
        client.on_wait = progress.task_waiting

        def work(job: tuple[Item, str]) -> tuple[Answer, str | None]:
            item, language = job
            prompt = item.prompts[language]
            progress.task_started()
            try:
                text = client.complete(prompt)
                return Answer(item.id, language, config.model, prompt, text), None
            except ProviderError as exc:
                return (
                    Answer(item.id, language, config.model, prompt, "", str(exc)),
                    f"{item.id}/{language}: {exc}",
                )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(work, job) for job in pending]
            for future in as_completed(futures):
                record, failure = future.result()
                with write_lock:
                    _append(path, record)
                    if failure:
                        failures.append(failure)
                progress.task_finished(ok=failure is None)
        client.on_wait = None

    report_discovered_rate(config, client)
    report_failures(len(pending), failures)
    return path


def report_discovered_rate(config: Config, client: ChatClient) -> None:
    """If the provider throttled us, say what rate actually worked."""
    limiter = client.limiter
    if not limiter.throttled:
        return
    print(
        f"NOTE: {config.provider} rate-limited this run; the budget settled at "
        f"{limiter.capacity} req/min (configured {config.requests_per_minute}).\n"
        f"      Set XLC_RPM={limiter.capacity} in .env to start there next time "
        f"and skip the discovery."
    )


def report_failures(attempted: int, failures: list[str]) -> None:
    """Say plainly how many calls failed. Silence here reads as success."""
    if not failures:
        print(f"Collected {attempted} answers, no failures.")
        return

    print(f"WARNING: {len(failures)} of {attempted} calls failed and were stored as errors.")
    for line in failures[:3]:
        print(f"  {line[:160]}")
    if len(failures) > 3:
        print(f"  ... and {len(failures) - 3} more")
    print("Rerun the same command to retry only the failed ones.")

    if len(failures) == attempted:
        raise SystemExit(
            "\nEvery call failed, so there is nothing to grade. Fix the error above "
            "and rerun 'xlc run'."
        )


def grade_answers(config: Config, items: list[Item], *, use_judge: bool = True) -> Path:
    """Grade collected answers, calling the judge only where code cannot decide."""
    out_dir = _results_dir(config)
    answers = dedupe_answers(_read_jsonl(out_dir / ANSWERS_FILE))
    if not answers:
        raise SystemExit("No answers found. Run 'xlc run' first.")

    grades_path = out_dir / GRADES_FILE
    done = {(row["item_id"], row["language"]) for row in _read_jsonl(grades_path)}
    by_id = {item.id: item for item in items}

    pending = [row for row in answers if (row["item_id"], row["language"]) not in done]
    if not pending:
        print(f"All {len(answers)} answers already graded.")
        return grades_path

    # Deterministic checks are pure functions over text, so they all run first
    # and instantly. Only what is left over costs a network round trip.
    needs_judge: list[dict] = []
    for row in pending:
        item = by_id.get(row["item_id"])
        if item is None:
            continue
        if row.get("error"):
            _append(grades_path, _grade_record(item, row, ERROR, row["error"], False))
            continue
        verdict = grade(item, row["answer"])
        if verdict.status == NEEDS_JUDGE:
            needs_judge.append(row)
            continue
        _append(grades_path, _grade_record(item, row, verdict.status, verdict.detail, False))

    print(f"Graded {len(pending) - len(needs_judge)} answers deterministically.")

    if not needs_judge:
        return grades_path
    if not use_judge:
        print(f"Skipping {len(needs_judge)} judged answers (--no-judge).")
        return grades_path

    client = ChatClient(config)
    workers = min(config.concurrency, len(needs_judge))
    estimate = estimate_seconds(len(needs_judge), config.requests_per_minute, workers=workers)
    print(f"Judging {len(needs_judge)} answers with {config.judge_model} "
          f"-> about {format_duration(estimate)}")

    write_lock = threading.Lock()

    with Progress(len(needs_judge), limiter=client.limiter) as progress:
        client.on_wait = progress.task_waiting

        def work(row: dict):
            progress.task_started()
            item = by_id[row["item_id"]]
            return row, judge_answer(client, item, row["language"], row["answer"])

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(work, row) for row in needs_judge]
            for future in as_completed(futures):
                row, verdict = future.result()
                item = by_id[row["item_id"]]
                with write_lock:
                    _append(
                        grades_path,
                        _grade_record(item, row, verdict.status, verdict.detail, True),
                    )
                progress.task_finished(ok=verdict.status != ERROR)
        client.on_wait = None

    return grades_path


def _grade_record(item: Item, row: dict, status: str, detail: str, judged: bool) -> Grade:
    return Grade(
        item_id=item.id,
        language=row["language"],
        category=item.category,
        model=row["model"],
        status=status,
        detail=detail,
        judged=judged,
    )


def load_grades(config: Config) -> list[dict]:
    return _read_jsonl(_results_dir(config) / GRADES_FILE)


def load_answers(config: Config) -> list[dict]:
    return dedupe_answers(_read_jsonl(_results_dir(config) / ANSWERS_FILE))


def results_dir(config: Config) -> Path:
    return _results_dir(config)
