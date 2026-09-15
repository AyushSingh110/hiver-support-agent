from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

GOLDEN_DIR = config.PROJECT_ROOT / "golden"
GOLDEN_SET = GOLDEN_DIR / "golden_set_v1.csv"

RETEST_SIZE = 45
BATCH_QUOTAS = {"b01": 27, "b02": 18}
RETEST_SEED = 44

# Only these reach the annotator. Everything else would reveal the mapping.
VISIBLE_COLUMNS = ["retest_id", "text"]
LABEL_COLUMNS = ["primary_intent", "secondary_intent", "confidence",
                 "is_ambiguous", "needs_discussion", "notes"]


def load_golden() -> pd.DataFrame:
    if not GOLDEN_SET.exists():
        raise FileNotFoundError(f"Frozen golden set not found at {GOLDEN_SET}")
    golden = pd.read_csv(GOLDEN_SET, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if len(golden) != 248:
        raise ValueError(f"Golden set must be 248 rows, found {len(golden)}")
    return golden


def select_rows(golden: pd.DataFrame, seed: int) -> pd.DataFrame:
    picked = []
    for batch, quota in sorted(BATCH_QUOTAS.items()):
        pool = golden[golden["batch"] == batch]
        if len(pool) < quota:
            raise ValueError(f"Batch {batch} has {len(pool)} rows, need {quota}")
        picked.append(pool.sample(n=quota, random_state=seed))
    selected = pd.concat(picked)

    # Shuffle before assigning ids so position reveals nothing about origin.
    shuffled = selected.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    shuffled["retest_id"] = [f"rt_{i:03d}" for i in range(1, len(shuffled) + 1)]
    return shuffled


def write_blank(rows: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(
            f"{path} already exists and is frozen. Delete it deliberately only if you "
            "intend to void the retest; regenerating changes which examples are shown."
        )
    blank = rows[VISIBLE_COLUMNS].copy()
    for column in LABEL_COLUMNS:
        blank[column] = ""
    blank.to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)


def write_key(rows: pd.DataFrame, path: Path, seed: int) -> None:
    key = {
        "seed": seed,
        "size": len(rows),
        "batch_quotas": BATCH_QUOTAS,
        "warning": "Contains the retest mapping and original labels. Do not open before the retest.",
        "mapping": {
            row["retest_id"]: {
                "annotation_id": row["annotation_id"],
                "conversation_id": row["conversation_id"],
                "batch": row["batch"],
                "original_primary_intent": row["primary_intent"],
                "original_confidence": row["confidence"],
            }
            for _, row in rows.iterrows()
        },
    }
    path.write_text(json.dumps(key, indent=2), encoding="utf-8")


def verify(rows: pd.DataFrame, golden: pd.DataFrame, blank_path: Path) -> dict:
    blank = pd.read_csv(blank_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    golden_by_id = golden.set_index("annotation_id")
    mapped_text = rows["annotation_id"].map(golden_by_id["text"])

    leaked = [c for c in blank.columns if c not in VISIBLE_COLUMNS + LABEL_COLUMNS]
    return {
        "row_count": len(blank),
        "row_count_is_45": len(blank) == RETEST_SIZE,
        "retest_id_unique": not blank["retest_id"].duplicated().any(),
        "annotation_id_unique": not rows["annotation_id"].duplicated().any(),
        "all_map_to_golden": bool(rows["annotation_id"].isin(golden["annotation_id"]).all()),
        "text_identity_matches": int((rows["text"] == mapped_text).sum()),
        "label_columns_all_blank": bool((blank[LABEL_COLUMNS] == "").all().all()),
        "no_identifying_columns_leaked": leaked == [],
        "batch_strata": rows["batch"].value_counts().to_dict(),
        "batch_strata_correct": rows["batch"].value_counts().to_dict() == BATCH_QUOTAS,
        "visible_columns": list(blank.columns),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an opaque test-retest batch.")
    parser.add_argument("--run", default="r01")
    parser.add_argument("--seed", type=int, default=RETEST_SEED)
    args = parser.parse_args()

    golden = load_golden()
    rows = select_rows(golden, args.seed)

    blank_path = GOLDEN_DIR / f"retest_{args.run}_blank.csv"
    write_blank(rows, blank_path)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    key_path = config.REPORTS_DIR / f"phase5_retest_{args.run}_key.json"
    write_key(rows, key_path, args.seed)

    checks = verify(rows, golden, blank_path)
    failed = [
        name for name, value in checks.items()
        if isinstance(value, bool) and not value
    ]
    if checks["text_identity_matches"] != RETEST_SIZE:
        failed.append("text_identity_matches")
    if failed:
        raise RuntimeError(f"Retest batch verification failed: {', '.join(failed)}")

    print(f"Blank batch : {blank_path} ({checks['row_count']} rows)")
    print(f"Key         : {key_path} (hidden - do not open before the retest)")
    print(f"Columns     : {checks['visible_columns']}")
    print("\nLabel columns are empty. No original label or identifier is exposed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
