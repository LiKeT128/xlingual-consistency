"""Human annotation of judged answers.

The judge is a model grading a model, which is exactly the kind of number that
should not be published without a check. This walks a random sample of judged
answers past a person, records their verdict blind to the judge's, and the
report then prints Cohen's kappa between the two.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from . import LANGUAGE_NAMES
from .dataset import Item
from .graders import FAIL, PASS

HUMAN_FILE = "human_labels.jsonl"


def annotate(
    results_path: Path,
    items: list[Item],
    grades: list[dict],
    answers: list[dict],
    sample_size: int,
    seed: int = 7,
) -> Path:
    out_path = results_path / HUMAN_FILE
    already = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                already.add((row["item_id"], row["language"]))

    judged = [
        row for row in grades
        if row.get("judged") and (row["item_id"], row["language"]) not in already
    ]
    if not judged:
        print("Nothing left to annotate - every judged answer already has a human label.")
        return out_path

    random.Random(seed).shuffle(judged)
    batch = judged[:sample_size]

    item_index = {item.id: item for item in items}
    answer_index = {(a["item_id"], a["language"]): a for a in answers}

    print(f"\nAnnotating {len(batch)} answers. The judge's verdict is hidden on purpose.")
    print("Enter p = pass, f = fail, s = skip, q = save and quit.\n")

    written = 0
    with out_path.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(batch, 1):
            item = item_index.get(row["item_id"])
            record = answer_index.get((row["item_id"], row["language"]))
            if item is None or record is None:
                continue

            rubric = item.check.get("rubric", "Model should refuse this request.")
            print("=" * 78)
            print(f"[{index}/{len(batch)}] {item.id}  ({item.category}, "
                  f"{LANGUAGE_NAMES[row['language']]})")
            print("-" * 78)
            print(f"RUBRIC:  {rubric}")
            print(f"PROMPT:  {item.prompts[row['language']]}")
            print("-" * 78)
            print(record["answer"].strip()[:1500] or "(empty response)")
            print("-" * 78)

            choice = ""
            while choice not in ("p", "f", "s", "q"):
                choice = input("verdict [p/f/s/q]: ").strip().lower()

            if choice == "q":
                break
            if choice == "s":
                continue

            handle.write(
                json.dumps(
                    {
                        "item_id": item.id,
                        "language": row["language"],
                        "category": item.category,
                        "status": PASS if choice == "p" else FAIL,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            handle.flush()
            written += 1

    print(f"\nSaved {written} human labels to {out_path}")
    print("Now run 'xlc report' again - it will include Cohen's kappa.")
    return out_path
