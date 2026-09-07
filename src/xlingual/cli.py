"""Command line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import LANGUAGES
from .annotate import HUMAN_FILE, annotate
from .config import Config, DATA_FILE, RESULTS_DIR
from .dataset import DatasetError, categories, load_items
from .report import build_report, load_human_labels, write_report
from .runner import (
    collect_answers,
    grade_answers,
    load_answers,
    load_grades,
    results_dir,
)


def _utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):  # pragma: no cover
        pass


def cmd_validate(args: argparse.Namespace) -> int:
    """Check the dataset without touching any API - this is what CI runs."""
    try:
        items = load_items(Path(args.data) if args.data else None)
    except DatasetError as exc:
        print(f"FAILED: {exc}")
        return 1

    print(f"OK: {len(items)} items, {len(LANGUAGES)} languages "
          f"= {len(items) * len(LANGUAGES)} prompts")
    for category in categories(items):
        count = sum(1 for item in items if item.category == category)
        judged = sum(1 for item in items if item.category == category and item.needs_judge)
        suffix = f" ({judged} judged by model)" if judged else " (all deterministic)"
        print(f"  {category:<14} {count:>3} items{suffix}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = Config.from_env()
    items = load_items(Path(args.data) if args.data else None)
    if args.limit:
        items = items[: args.limit]
    collect_answers(config, items, languages=tuple(args.languages))
    print(f"Answers written to {results_dir(config)}")
    return 0


def cmd_grade(args: argparse.Namespace) -> int:
    config = Config.from_env()
    items = load_items(Path(args.data) if args.data else None)
    grade_answers(config, items, use_judge=not args.no_judge)
    print(f"Grades written to {results_dir(config)}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    config = Config.from_env()
    items = load_items(Path(args.data) if args.data else None)
    grades = load_grades(config)
    if not grades:
        raise SystemExit("No grades found. Run 'xlc run' then 'xlc grade' first.")
    answers = load_answers(config)
    humans = load_human_labels(results_dir(config) / HUMAN_FILE)

    content = build_report(config.model, config.provider, items, grades, answers, humans)
    target = Path(args.out) if args.out else results_dir(config) / "report.md"
    write_report(target, content)
    print(content)
    print(f"\nReport written to {target}")
    return 0


def cmd_annotate(args: argparse.Namespace) -> int:
    config = Config.from_env()
    items = load_items(Path(args.data) if args.data else None)
    grades = load_grades(config)
    answers = load_answers(config)
    if not grades:
        raise SystemExit("No grades found. Run 'xlc run' then 'xlc grade' first.")
    annotate(results_dir(config), items, grades, answers, args.sample)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xlc",
        description="Measure whether an LLM gives the same answer in English, "
                    "Russian, Ukrainian and Slovak.",
    )
    parser.add_argument("--data", help=f"dataset path (default: {DATA_FILE})")
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="check the dataset, no API calls")
    p_validate.set_defaults(func=cmd_validate)

    p_run = sub.add_parser("run", help="collect model answers")
    p_run.add_argument("--limit", type=int, help="only the first N items")
    p_run.add_argument("--languages", nargs="+", default=list(LANGUAGES))
    p_run.set_defaults(func=cmd_run)

    p_grade = sub.add_parser("grade", help="grade collected answers")
    p_grade.add_argument("--no-judge", action="store_true",
                         help="skip judged categories entirely")
    p_grade.set_defaults(func=cmd_grade)

    p_report = sub.add_parser("report", help="write the Markdown report")
    p_report.add_argument("--out", help="output path")
    p_report.set_defaults(func=cmd_report)

    p_annotate = sub.add_parser("annotate", help="hand-label a sample of judged answers")
    p_annotate.add_argument("--sample", type=int, default=50)
    p_annotate.set_defaults(func=cmd_annotate)

    return parser


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except DatasetError as exc:
        print(f"Dataset error: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is saved - run the same command again to resume.")
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
