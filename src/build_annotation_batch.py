from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src import config

BRAND = "AmericanAir"
GOLDEN_DIR = config.PROJECT_ROOT / "golden"

RANDOM_QUOTA = 120
TARGETED_QUOTA = 30

# Probes come from Phase 4 term evidence, not from cluster assignments. They are a
# lexical bias by construction, which is why these rows are flagged `targeted`.
RARE_INTENT_PROBES = {
    "loyalty_and_lounge": r"\b(?:aadvantage|admirals club|elite status|executive platinum|"
                          r"lounge|miles posted|my miles)\b",
    "booking_fees_and_fare_rules": r"\b(?:basic economy|fare rules|change fee|refund|voucher|"
                                   r"rebook fee|award ticket)\b",
    "staff_and_service_complaint": r"\b(?:gate agent|flight attendant|steward|crew was|"
                                   r"agent was)\b",
    "seating_and_upgrade": r"\b(?:upgrade|first class|business class|seat assignment|"
                           r"extra legroom)\b",
}

ANNOTATION_COLUMNS = [
    "annotation_id",
    "conversation_id",
    "tweet_id",
    "created_at",
    "sampling_stratum",
    "text_display",
    "primary_intent",
    "secondary_intent",
    "confidence",
    "is_ambiguous",
    "needs_discussion",
    "notes",
]
LABEL_COLUMNS = ["primary_intent", "secondary_intent", "confidence",
                 "is_ambiguous", "needs_discussion", "notes"]

NEWLINE_PATTERN = re.compile(r"[\r\n]+")


def load_openings() -> pd.DataFrame:
    conversations = pd.read_parquet(config.PROCESSED_DIR / "conversations.parquet")
    clean_ids = set(
        conversations.loc[
            (conversations["status"] == "clean") & (conversations["brand"] == BRAND),
            "conversation_id",
        ]
    )
    if not clean_ids:
        raise ValueError(f"No clean {BRAND} conversations found; run src.reconstruct first")

    parts = []
    for batch in pq.ParquetFile(config.PROCESSED_DIR / "conversation_turns.parquet").iter_batches(
        batch_size=250_000,
        columns=["conversation_id", "tweet_id", "brand", "speaker", "author_id", "text",
                 "turn_index", "created_at"],
    ):
        block = batch.to_pandas()
        block = block[
            (block["brand"] == BRAND)
            & (block["turn_index"] == 0)
            & (block["speaker"] == "customer")
            & (block["conversation_id"].isin(clean_ids))
        ]
        if not block.empty:
            parts.append(block[["conversation_id", "tweet_id", "author_id", "text", "created_at"]])

    openings = pd.concat(parts, ignore_index=True)
    return openings.sort_values("conversation_id").reset_index(drop=True)


def already_labelled_conversations() -> set[str]:
    seen = set()
    for path in sorted(GOLDEN_DIR.glob("*_labelled.csv")):
        frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
        seen.update(frame["conversation_id"].dropna())
    return seen


def sample_random_stratified(openings: pd.DataFrame, quota: int, seed: int) -> pd.DataFrame:
    # Month-proportional so a single incident or busy week cannot dominate the batch.
    month = openings["created_at"].dt.strftime("%Y-%m")
    shares = month.value_counts(normalize=True).sort_index()
    allocation = (shares * quota).round().astype(int)

    shortfall = quota - int(allocation.sum())
    if shortfall:
        allocation.iloc[allocation.to_numpy().argmax()] += shortfall

    picked = []
    for period, count in allocation.items():
        pool = openings[month == period]
        if count <= 0 or pool.empty:
            continue
        picked.append(pool.sample(n=min(count, len(pool)), random_state=seed))
    return pd.concat(picked).sort_values("conversation_id")


def sample_targeted(openings: pd.DataFrame, quota: int, seed: int) -> pd.DataFrame:
    # head(0) rather than an empty DataFrame: a dtype-less frame would degrade
    # created_at to object when concatenated with the random pool.
    if quota <= 0:
        return openings.head(0).assign(probe=pd.Series(dtype="object"))

    per_probe = max(quota // len(RARE_INTENT_PROBES), 1)
    picked, used = [], set()
    for intent, pattern in sorted(RARE_INTENT_PROBES.items()):
        matcher = re.compile(pattern, re.I)
        pool = openings[
            openings["text"].map(lambda value: bool(matcher.search(str(value))))
            & ~openings["conversation_id"].isin(used)
        ]
        if pool.empty:
            continue
        chosen = pool.sample(n=min(per_probe, len(pool)), random_state=seed)
        chosen = chosen.assign(probe=intent)
        used.update(chosen["conversation_id"])
        picked.append(chosen)
    if not picked:
        return pd.DataFrame(columns=list(openings.columns) + ["probe"])
    return pd.concat(picked).sort_values("conversation_id")


def build_rows(random_pool: pd.DataFrame, targeted_pool: pd.DataFrame, batch: str) -> pd.DataFrame:
    combined = pd.concat(
        [random_pool.assign(sampling_stratum="random"),
         targeted_pool.assign(sampling_stratum="targeted")],
        ignore_index=True,
    ).sort_values(["sampling_stratum", "conversation_id"]).reset_index(drop=True)

    rows = pd.DataFrame(
        {
            "annotation_id": [f"{batch}_{i:04d}" for i in range(1, len(combined) + 1)],
            "conversation_id": combined["conversation_id"],
            "tweet_id": combined["tweet_id"],
            "created_at": combined["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "sampling_stratum": combined["sampling_stratum"],
            "text_display": combined["text"].fillna("").map(
                lambda value: NEWLINE_PATTERN.sub(" / ", str(value)).strip()
            ),
        }
    )
    for column in LABEL_COLUMNS:
        rows[column] = ""
    return rows[ANNOTATION_COLUMNS]


def verify_blank_batch(path: Path, expected: pd.DataFrame) -> dict:
    reloaded = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    checks = {
        "row_count_matches": len(reloaded) == len(expected),
        "columns_match": list(reloaded.columns) == ANNOTATION_COLUMNS,
        "all_label_columns_empty": bool(
            (reloaded[LABEL_COLUMNS] == "").all().all()
        ),
        "annotation_ids_unique": not reloaded["annotation_id"].duplicated().any(),
        "conversation_ids_unique": not reloaded["conversation_id"].duplicated().any(),
        "no_empty_text": bool((reloaded["text_display"].str.strip() != "").all()),
        "text_survives_roundtrip": bool(
            (reloaded["text_display"].tolist() == expected["text_display"].tolist())
        ),
    }
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a blank annotation batch for manual labelling.")
    parser.add_argument("--batch", default="b01", help="Batch prefix, e.g. b01")
    parser.add_argument("--name", default="pilot", help="Batch name used in the filename")
    parser.add_argument("--random-quota", type=int, default=RANDOM_QUOTA)
    parser.add_argument("--targeted-quota", type=int, default=TARGETED_QUOTA)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--exclude-same-customer", dest="exclude_same_customer",
                        action="store_true", default=True)
    parser.add_argument("--allow-same-customer", dest="exclude_same_customer",
                        action="store_false")
    args = parser.parse_args()

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    output_path = GOLDEN_DIR / f"{args.batch}_{args.name}_blank.csv"
    if output_path.name.endswith("_labelled.csv"):
        raise ValueError("Refusing to write to a *_labelled.csv path")
    if output_path.exists():
        raise FileExistsError(
            f"{output_path} already exists. Delete it deliberately if you intend to redraw; "
            "regenerating would change which messages you are asked to label."
        )

    print("[1/4] Loading AmericanAir opening messages ...", flush=True)
    openings = load_openings()
    population = len(openings)

    excluded = already_labelled_conversations()
    openings = openings[~openings["conversation_id"].isin(excluded)]
    after_batch_exclusion = len(openings)

    # A customer who already appears in a labelled batch is dropped entirely: two
    # conversations from one person are not independent evaluation examples.
    same_customer_excluded = 0
    if args.exclude_same_customer and excluded:
        labelled_customers = set(
            load_openings().set_index("conversation_id")["author_id"].reindex(excluded).dropna()
        )
        keep = ~openings["author_id"].isin(labelled_customers)
        same_customer_excluded = int((~keep).sum())
        openings = openings[keep]

    print(
        f"      {len(openings):,} eligible "
        f"({len(excluded):,} already labelled, {same_customer_excluded:,} same-customer)",
        flush=True,
    )

    print("[2/4] Sampling ...", flush=True)
    random_pool = sample_random_stratified(openings, args.random_quota, args.seed)
    remaining = openings[~openings["conversation_id"].isin(set(random_pool["conversation_id"]))]
    targeted_pool = sample_targeted(remaining, args.targeted_quota, args.seed)
    rows = build_rows(random_pool, targeted_pool, args.batch)
    print(f"      {len(random_pool)} random + {len(targeted_pool)} targeted = {len(rows)}", flush=True)

    print("[3/4] Writing blank batch ...", flush=True)
    rows.to_csv(output_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)

    print("[4/4] Verifying ...", flush=True)
    checks = verify_blank_batch(output_path, rows)
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Blank batch verification failed: {', '.join(failed)}")

    month_counts = (
        pd.to_datetime(rows.loc[rows["sampling_stratum"] == "random", "created_at"])
        .dt.strftime("%Y-%m").value_counts().sort_index().to_dict()
    )
    manifest = {
        "batch": args.batch,
        "brand": BRAND,
        "seed": args.seed,
        "population_total": population,
        "excluded_already_labelled": len(excluded),
        "population_after_batch_exclusion": after_batch_exclusion,
        "exclude_same_customer": args.exclude_same_customer,
        "excluded_same_customer": same_customer_excluded,
        "eligible_population": len(openings),
        "random_quota": args.random_quota,
        "targeted_quota": args.targeted_quota,
        "random_drawn": len(random_pool),
        "targeted_drawn": len(targeted_pool),
        "random_month_distribution": month_counts,
        "targeted_probes": (
            {k: v for k, v in sorted(RARE_INTENT_PROBES.items())}
            if args.targeted_quota > 0 else {}
        ),
        "selected_conversation_ids": sorted(rows["conversation_id"].tolist()),
        "verification": checks,
        "output": str(output_path.relative_to(config.PROJECT_ROOT)),
    }
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = config.REPORTS_DIR / f"phase5_{args.batch}_sampling_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nBlank batch: {output_path}")
    print(f"Manifest:    {manifest_path}")
    print("\nLabel columns are empty. No intent has been inferred or assigned.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
