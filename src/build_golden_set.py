from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src import config

GOLDEN_DIR = config.PROJECT_ROOT / "golden"
BRAND = "AmericanAir"
TAXONOMY_VERSION = "v2"
NEAR_DUPLICATE_THRESHOLD = 0.8

BATCHES = {
    "b01": {
        "file": "b01_pilot_labelled.csv",
        "expected_rows": 148,
        "taxonomy_at_labelling": "v1_then_revised_to_v2",
        "labelling_independence": "taxonomy_refined_after_reading",
    },
    "b02": {
        "file": "b02_fresh_labelled.csv",
        "expected_rows": 100,
        "taxonomy_at_labelling": "v2_frozen",
        "labelling_independence": "independent",
    },
}

TAXONOMY_LABELS = {
    "flight_delay", "flight_cancellation_rebooking", "baggage", "seating_and_upgrade",
    "boarding_and_gate", "booking_fees_and_fare_rules", "staff_and_service_complaint",
    "loyalty_and_lounge", "praise_and_compliment", "general_dissatisfaction",
    "non_support_commentary", "OTHER", "UNCLEAR",
}
CONFIDENCE_VALUES = {"high", "medium", "low"}
FLAG_TRUE = {"yes"}

INPUT_SCHEMA = [
    "annotation_id", "conversation_id", "tweet_id", "created_at", "sampling_stratum",
    "text_display", "primary_intent", "secondary_intent", "confidence",
    "is_ambiguous", "needs_discussion", "notes",
]

# Renames text_display -> text and is_ambiguous -> ambiguous.
OUTPUT_SCHEMA = [
    "annotation_id", "batch", "sampling_stratum", "taxonomy_at_labelling",
    "labelling_independence", "conversation_id", "tweet_id", "customer_author_id",
    "created_at", "text", "primary_intent", "secondary_intent", "confidence",
    "ambiguous", "needs_discussion", "notes",
]

NEWLINE_PATTERN = re.compile(r"[\r\n]+")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


def load_population() -> pd.DataFrame:
    conversations = pd.read_parquet(config.PROCESSED_DIR / "conversations.parquet")
    clean_ids = set(
        conversations.loc[
            (conversations["status"] == "clean") & (conversations["brand"] == BRAND),
            "conversation_id",
        ]
    )
    parts = []
    for batch in pq.ParquetFile(config.PROCESSED_DIR / "conversation_turns.parquet").iter_batches(
        batch_size=250_000,
        columns=["conversation_id", "tweet_id", "brand", "speaker", "author_id", "text", "turn_index"],
    ):
        block = batch.to_pandas()
        block = block[
            (block["brand"] == BRAND)
            & (block["turn_index"] == 0)
            & (block["speaker"] == "customer")
            & (block["conversation_id"].isin(clean_ids))
        ]
        if not block.empty:
            parts.append(block[["conversation_id", "tweet_id", "author_id", "text"]])
    population = pd.concat(parts, ignore_index=True).drop_duplicates("conversation_id")
    return population.sort_values("conversation_id").reset_index(drop=True)


def normalise_text(value: str) -> str:
    stripped = MENTION_PATTERN.sub(" ", URL_PATTERN.sub(" ", str(value).lower()))
    return " ".join(TOKEN_PATTERN.findall(stripped))


def display_form(value: str) -> str:
    return NEWLINE_PATTERN.sub(" / ", str(value)).strip()


def validate_batch(name: str, frame: pd.DataFrame, spec: dict, population: pd.DataFrame) -> dict:
    source = population.set_index("conversation_id")
    checks = {
        "row_count": len(frame),
        "row_count_matches": len(frame) == spec["expected_rows"],
        "schema_matches": list(frame.columns) == INPUT_SCHEMA,
        "unique_annotation_id": not frame["annotation_id"].duplicated().any(),
        "unique_conversation_id": not frame["conversation_id"].duplicated().any(),
        "unique_tweet_id": not frame["tweet_id"].duplicated().any(),
        "no_empty_text": bool((frame["text_display"].str.strip() != "").all()),
        "primary_intent_complete": int((frame["primary_intent"].str.strip() == "").sum()) == 0,
        "confidence_complete": int((frame["confidence"].str.strip() == "").sum()) == 0,
    }
    checks["invalid_primary"] = sorted(
        {v for v in frame["primary_intent"] if v.strip() and v not in TAXONOMY_LABELS}
    )
    checks["invalid_secondary"] = sorted(
        {v for v in frame["secondary_intent"] if v.strip() and v not in TAXONOMY_LABELS}
    )
    checks["invalid_confidence"] = sorted(
        {v for v in frame["confidence"] if v.strip() and v.lower() not in CONFIDENCE_VALUES}
    )
    checks["invalid_flags"] = sorted(
        {v for column in ("is_ambiguous", "needs_discussion")
         for v in frame[column] if v.strip() and v.lower() not in FLAG_TRUE}
    )

    truth = frame["conversation_id"].map(source["text"]).fillna("").map(display_form)
    matches = frame["text_display"].str.strip() == truth.str.strip()
    checks["text_matches_source"] = int(matches.sum())
    checks["text_mismatched_ids"] = frame.loc[~matches, "annotation_id"].tolist()

    expected_tweet = frame["conversation_id"].map(source["tweet_id"]).astype("Int64").astype(str)
    checks["tweet_id_matches_source"] = bool((frame["tweet_id"] == expected_tweet).all())
    return checks


def build_golden_frame(batches: dict[str, pd.DataFrame], population: pd.DataFrame) -> pd.DataFrame:
    author_by_conversation = population.set_index("conversation_id")["author_id"]
    rows = []
    for name, frame in batches.items():
        spec = BATCHES[name]
        rows.append(
            frame.assign(
                batch=name,
                taxonomy_at_labelling=spec["taxonomy_at_labelling"],
                labelling_independence=spec["labelling_independence"],
                customer_author_id=frame["conversation_id"].map(author_by_conversation),
            ).rename(columns={"text_display": "text", "is_ambiguous": "ambiguous"})
        )
    combined = pd.concat(rows, ignore_index=True)
    # Deterministic order: batch, then annotation_id within batch.
    combined = combined.sort_values(["batch", "annotation_id"]).reset_index(drop=True)
    return combined[OUTPUT_SCHEMA]


def jaccard_overlaps(golden: pd.DataFrame, population: pd.DataFrame, threshold: float) -> list[dict]:
    """Exact near-duplicate search. A length band prunes candidates; no sampling."""
    golden_ids = set(golden["conversation_id"])
    pool = population[~population["conversation_id"].isin(golden_ids)].copy()
    pool["tokens"] = pool["text"].map(lambda t: frozenset(normalise_text(t).split()))
    pool = pool[pool["tokens"].map(len) > 0]

    pool_tokens = pool["tokens"].tolist()
    pool_convs = pool["conversation_id"].tolist()
    pool_sizes = [len(t) for t in pool_tokens]

    hits = []
    for _, row in golden.iterrows():
        tokens = frozenset(normalise_text(row["text"]).split())
        if not tokens:
            continue
        size = len(tokens)
        # Jaccard >= t forces the sizes to sit within [t*size, size/t].
        low, high = threshold * size, size / threshold
        for index, other in enumerate(pool_tokens):
            if not low <= pool_sizes[index] <= high:
                continue
            shared = len(tokens & other)
            if not shared:
                continue
            score = shared / (size + pool_sizes[index] - shared)
            if score >= threshold:
                hits.append({
                    "annotation_id": row["annotation_id"],
                    "golden_conversation_id": row["conversation_id"],
                    "population_conversation_id": pool_convs[index],
                    "jaccard": round(score, 4),
                })
    return sorted(hits, key=lambda h: (-h["jaccard"], h["annotation_id"]))


def leakage_report(golden: pd.DataFrame, population: pd.DataFrame) -> dict:
    golden_convs = set(golden["conversation_id"])
    golden_tweets = set(golden["tweet_id"])
    golden_customers = set(golden["customer_author_id"].dropna())

    rest = population[~population["conversation_id"].isin(golden_convs)]
    rest_customers = set(rest["author_id"])
    shared_customers = sorted(golden_customers & rest_customers)

    golden_norm = golden["text"].map(normalise_text)
    rest_norm = rest["text"].map(normalise_text)
    rest_lookup = {}
    for conversation_id, text in zip(rest["conversation_id"], rest_norm):
        rest_lookup.setdefault(text, []).append(conversation_id)
    exact = [
        {"annotation_id": aid, "golden_conversation_id": cid,
         "population_conversation_id": rest_lookup[text][0]}
        for aid, cid, text in zip(golden["annotation_id"], golden["conversation_id"], golden_norm)
        if text and text in rest_lookup
    ]

    near = jaccard_overlaps(golden, population, NEAR_DUPLICATE_THRESHOLD)
    return {
        "population_size": len(population),
        "conversation_id_overlap": len(golden_convs & set(rest["conversation_id"])),
        "tweet_id_overlap": len(golden_tweets & set(rest["tweet_id"].astype(str))),
        "customer_author_id_overlap_count": len(shared_customers),
        "customer_author_id_overlap_examples": shared_customers[:20],
        "conversations_sharing_a_golden_customer": int(rest["author_id"].isin(golden_customers).sum()),
        "exact_normalised_text_overlap_count": len(exact),
        "exact_normalised_text_overlap": exact[:20],
        "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
        "near_duplicate_overlap_count": len(near),
        "near_duplicate_overlap": near[:40],
    }


def write_exclusion_list(golden: pd.DataFrame, path: Path) -> int:
    lines = [
        "# Golden-set exclusion list - Phase 5",
        "#",
        "# One conversation_id per line. These 248 conversations are the human-labelled",
        "# golden evaluation set and MUST be excluded from every model-development step:",
        "#   - classifier training data",
        "#   - retrieval corpus",
        "#   - few-shot prompt examples",
        "#   - threshold tuning / dev splits",
        "#",
        "# Usage:",
        "#   excluded = {l.strip() for l in open(path) if l.strip() and not l.startswith('#')}",
        "#   corpus = corpus[~corpus['conversation_id'].isin(excluded)]",
        "#",
        "# conversation_id is the correct key: it is stable, unique per golden row, and",
        "# joins directly to data/processed/conversations.parquet and conversation_turns.",
        "# Excluding by conversation removes every turn of that thread, not just the",
        "# opening message the annotator saw.",
        "#",
        f"# rows: {len(golden)}",
    ]
    lines += sorted(golden["conversation_id"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(golden)


def distribution(frame: pd.DataFrame, column: str = "primary_intent") -> dict:
    return frame[column].value_counts().sort_index().to_dict()


def render_report(stats: dict) -> str:
    integrity = stats["integrity"]
    leak = stats["leakage"]
    lines = [
        "# Phase 5 — Golden set validation",
        "",
        "Generated by `src/build_golden_set.py`. Both labelled batches are opened "
        "read-only and are never modified.",
        "",
        "## Dataset construction",
        "",
        f"- Total rows: **{stats['total_rows']}**",
        f"- b01 (pilot): **{stats['b01_rows']}** — 120 random + 28 targeted",
        f"- b02 (fresh): **{stats['b02_rows']}** — 100 random, independently sampled",
        f"- Taxonomy version: **{stats['taxonomy_version']}** (frozen)",
        f"- Deterministic ordering: `{stats['ordering']}`",
        "",
        "| batch | taxonomy_at_labelling | labelling_independence |",
        "| --- | --- | --- |",
        "| b01 | `v1_then_revised_to_v2` | `taxonomy_refined_after_reading` |",
        "| b02 | `v2_frozen` | `independent` |",
        "",
        "## Label distribution",
        "",
        "> **b02 is the independently sampled evaluation subset; b01 contains targeted "
        "pilot examples and therefore the combined 248-example distribution is not a "
        "prevalence estimate.**",
        "",
        "| intent | all 248 | b01 (148) | b02 (100) |",
        "| --- | --- | --- | --- |",
    ]
    for intent in sorted(stats["distribution_all"], key=lambda k: -stats["distribution_all"][k]):
        lines.append(
            f"| `{intent}` | {stats['distribution_all'].get(intent, 0)} | "
            f"{stats['distribution_b01'].get(intent, 0)} | "
            f"{stats['distribution_b02'].get(intent, 0)} |"
        )

    lines += [
        "",
        "## Annotation quality",
        "",
        f"- Confidence: {stats['confidence_distribution']}",
        f"- Rows flagged ambiguous: **{stats['ambiguous_count']}**",
        f"- Rows flagged needs_discussion: **{stats['needs_discussion_count']}**",
        f"- Rows with a secondary intent: {stats['secondary_intent_count']}",
        "",
        "## Integrity",
        "",
        "| Check | Result |",
        "| --- | --- |",
        f"| Total rows == 248 | {integrity['total_is_248']} |",
        f"| b01 rows == 148 | {integrity['b01_is_148']} |",
        f"| b02 rows == 100 | {integrity['b02_is_100']} |",
        f"| Unique annotation_id | {integrity['unique_annotation_id']} |",
        f"| Unique conversation_id | {integrity['unique_conversation_id']} |",
        f"| Unique tweet_id | {integrity['unique_tweet_id']} |",
        f"| Text matches Phase 2 source | {integrity['text_matches_source']} / 248 |",
        f"| tweet_id matches source | {integrity['tweet_id_matches_source']} |",
        f"| customer_author_id resolved | {integrity['customer_author_id_resolved']} / 248 |",
        f"| Output schema as specified | {integrity['schema_matches']} |",
        f"| All labels valid under v2 | {integrity['all_labels_valid']} |",
        "",
        "## Leakage",
        "",
        f"Checked against the full population of **{leak['population_size']:,}** "
        "AmericanAir customer-rooted opening messages.",
        "",
        "| Check | Result |",
        "| --- | --- |",
        f"| conversation_id overlap | **{leak['conversation_id_overlap']}** |",
        f"| tweet_id overlap | **{leak['tweet_id_overlap']}** |",
        f"| customer_author_id overlap | **{leak['customer_author_id_overlap_count']}** |",
        f"| …conversations sharing a golden customer | {leak['conversations_sharing_a_golden_customer']} |",
        f"| exact normalised-text overlap | **{leak['exact_normalised_text_overlap_count']}** |",
        f"| near-duplicate (Jaccard >= {leak['near_duplicate_threshold']}) | **{leak['near_duplicate_overlap_count']}** |",
        f"| exclusion-list coverage | {stats['exclusion_list_rows']} / 248 |",
        "",
        "**No golden example was removed because of leakage.** Overlaps are reported and "
        "handled downstream by `golden/golden_exclusion_ids.txt`, which removes the whole "
        "conversation from any training, retrieval, prompt or tuning set.",
        "",
    ]
    if leak["near_duplicate_overlap"]:
        lines += ["Near-duplicate pairs found:", "",
                  "| annotation_id | golden conversation | population conversation | Jaccard |",
                  "| --- | --- | --- | --- |"]
        lines += [
            f"| {h['annotation_id']} | {h['golden_conversation_id']} | "
            f"{h['population_conversation_id']} | {h['jaccard']} |"
            for h in leak["near_duplicate_overlap"]
        ]
        lines.append("")
    if leak["customer_author_id_overlap_count"]:
        lines += [
            f"Golden customers who also appear elsewhere in the population: "
            f"**{leak['customer_author_id_overlap_count']}**, covering "
            f"{leak['conversations_sharing_a_golden_customer']} other conversations. "
            "Those conversations are **not** in the golden set, but they belong to the "
            "same people. Downstream code should exclude by customer as well as by "
            "conversation if it wants strict customer-level separation.",
            "",
        ]

    lines += [
        "### What cannot be deterministically detected",
        "",
        "- **One human operating multiple anonymised accounts.** `author_id` values are "
        "pseudonyms with no identity resolution, so two accounts belonging to the same "
        "person are indistinguishable from two different people.",
        "- **Semantically identical incidents from different customers.** A single "
        "cancelled flight produces many near-identical complaints from unrelated people. "
        "These are legitimately distinct conversations and no identifier links them, but "
        "they let the retrieval corpus contain a sibling of a golden example. This is "
        "*semantic* leakage: measurable after the fact, not preventable by ID exclusion.",
        "- **Cross-thread continuation.** A customer raising the same issue in a new "
        "thread from a different account is invisible to every check above.",
        "",
        "## Reproducibility",
        "",
        f"- Deterministic ordering: `{stats['ordering']}`",
        "- No randomness in construction; the output is a pure function of the two "
        "labelled CSVs and the Phase 2 Parquet.",
        "- Raw data untouched; both labelled CSVs opened read-only.",
        "- No network, LLM, embedding or model download at any point.",
        "",
        "## Limitations",
        "",
        "- **b01 was partly targeted** (28 of 148 rows drawn by keyword probe), so the "
        "combined 248-row distribution overstates rare intents. `seating_and_upgrade` is "
        "the clearest case: 14 in b01 against 2 in b02.",
        "- **b02 is the clean independent random subset** and is the only part of the "
        "golden set that supports prevalence claims.",
        "- **Per-intent metrics for rare intents will have very wide uncertainty.** With "
        "2 examples of `seating_and_upgrade` and 2 of `UNCLEAR` in b02, per-intent recall "
        "on the independent subset is close to meaningless for those classes.",
        "- **One annotator.** No inter-annotator agreement exists; a blind test-retest is "
        "planned to give an intra-annotator figure, which is an upper bound on reliability.",
        "- **Reviewer suggestions may have influenced some labels.** The annotator made "
        "every final decision, but where model or reviewer suggestions informed a call, "
        "agreement between an LLM judge and these labels would be inflated by shared "
        "influence. This must be disclosed as a limitation when LLM-judge agreement is "
        "reported, rather than presented as fully independent annotation.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    population = load_population()

    frames, batch_checks = {}, {}
    for name, spec in BATCHES.items():
        path = GOLDEN_DIR / spec["file"]
        if not path.exists():
            raise FileNotFoundError(f"Labelled batch not found at {path}")
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        checks = validate_batch(name, frame, spec, population)
        failures = [
            key for key, value in checks.items()
            if (isinstance(value, bool) and not value) or (isinstance(value, list) and value)
        ]
        if failures or checks["text_matches_source"] != spec["expected_rows"]:
            raise RuntimeError(f"{name} failed validation: {failures or 'text mismatch'}")
        frames[name], batch_checks[name] = frame, checks

    golden = build_golden_frame(frames, population)
    if len(golden) != 248:
        raise RuntimeError(f"Expected 248 golden rows, built {len(golden)}")

    golden_path = GOLDEN_DIR / "golden_set_v1.csv"
    golden.to_csv(golden_path, index=False, encoding="utf-8-sig")
    exclusion_rows = write_exclusion_list(golden, GOLDEN_DIR / "golden_exclusion_ids.txt")

    source = population.set_index("conversation_id")
    truth = golden["conversation_id"].map(source["text"]).fillna("").map(display_form)
    expected_tweet = golden["conversation_id"].map(source["tweet_id"]).astype("Int64").astype(str)
    b01, b02 = golden[golden["batch"] == "b01"], golden[golden["batch"] == "b02"]

    integrity = {
        "total_is_248": len(golden) == 248,
        "b01_is_148": len(b01) == 148,
        "b02_is_100": len(b02) == 100,
        "unique_annotation_id": not golden["annotation_id"].duplicated().any(),
        "unique_conversation_id": not golden["conversation_id"].duplicated().any(),
        "unique_tweet_id": not golden["tweet_id"].duplicated().any(),
        "text_matches_source": int((golden["text"].str.strip() == truth.str.strip()).sum()),
        "tweet_id_matches_source": bool((golden["tweet_id"] == expected_tweet).all()),
        "customer_author_id_resolved": int(golden["customer_author_id"].notna().sum()),
        "schema_matches": list(golden.columns) == OUTPUT_SCHEMA,
        "all_labels_valid": bool(set(golden["primary_intent"]) <= TAXONOMY_LABELS),
    }

    stats = {
        "taxonomy_version": TAXONOMY_VERSION,
        "ordering": "sort by batch, then annotation_id",
        "total_rows": len(golden),
        "b01_rows": len(b01),
        "b02_rows": len(b02),
        "distribution_all": distribution(golden),
        "distribution_b01": distribution(b01),
        "distribution_b02": distribution(b02),
        "confidence_distribution": golden["confidence"].value_counts().to_dict(),
        "ambiguous_count": int((golden["ambiguous"].str.strip() != "").sum()),
        "needs_discussion_count": int((golden["needs_discussion"].str.strip() != "").sum()),
        "secondary_intent_count": int((golden["secondary_intent"].str.strip() != "").sum()),
        "sampling_strata": {
            f"{batch}/{stratum}": int(count)
            for (batch, stratum), count in golden.groupby(["batch", "sampling_stratum"]).size().items()
        },
        "batch_validation": batch_checks,
        "integrity": integrity,
        "exclusion_list_rows": exclusion_rows,
        "leakage": leakage_report(golden, population),
    }

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "phase5_golden_set_validation.json").write_text(
        json.dumps(stats, indent=2, default=str), encoding="utf-8"
    )
    (config.REPORTS_DIR / "phase5_golden_set_validation.md").write_text(
        render_report(stats), encoding="utf-8"
    )

    print(f"Golden set : {golden_path} ({len(golden)} rows)")
    print(f"Exclusions : {GOLDEN_DIR / 'golden_exclusion_ids.txt'} ({exclusion_rows} ids)")
    print(f"Report     : {config.REPORTS_DIR / 'phase5_golden_set_validation.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
