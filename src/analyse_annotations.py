from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src import config

GOLDEN_DIR = config.PROJECT_ROOT / "golden"

TAXONOMY_INTENTS = [
    "flight_delay",
    "flight_cancellation_rebooking",
    "baggage",
    "seating_and_upgrade",
    "boarding_and_gate",
    "booking_fees_and_fare_rules",
    "staff_and_service_complaint",
    "loyalty_and_lounge",
    "praise_and_compliment",
    "general_dissatisfaction",
]
ESCAPE_LABELS = ["OTHER", "UNCLEAR"]
VALID_LABELS = set(TAXONOMY_INTENTS) | set(ESCAPE_LABELS)

# Applied in analysis only; the annotator's file is never modified.
INTENT_NORMALISATIONS = {
    "fligh_delay": "flight_delay",
    "baggae": "baggage",
    "booking_fee_and_fare_rules": "booking_fees_and_fare_rules",
}
CONFIDENCE_NORMALISATIONS = {"med": "medium", "medium": "medium", "high": "high", "low": "low"}
FLAG_TRUE_VALUES = {"y", "yes", "true", "1"}

# Declared in docs/DECISION_LOG.md D29 before any label existed.
TRIGGERS = {
    "other_rate_pct": 10.0,
    "unclear_rate_pct": 15.0,
    "min_intent_share_pct": 2.0,
    "max_low_confidence_pct": 30.0,
    "max_pair_cooccurrence_pct": 15.0,
}

NEWLINE_PATTERN = re.compile(r"[\r\n]+")


def load_source_texts(conversation_ids: set[str]) -> pd.Series:
    parts = []
    for batch in pq.ParquetFile(config.PROCESSED_DIR / "conversation_turns.parquet").iter_batches(
        batch_size=250_000, columns=["conversation_id", "speaker", "turn_index", "text", "created_at"]
    ):
        block = batch.to_pandas()
        block = block[
            (block["turn_index"] == 0)
            & (block["speaker"] == "customer")
            & (block["conversation_id"].isin(conversation_ids))
        ]
        if not block.empty:
            parts.append(block[["conversation_id", "text"]])
    if not parts:
        raise ValueError("No source texts found for the annotated conversations")
    return pd.concat(parts).drop_duplicates("conversation_id").set_index("conversation_id")["text"]


def normalise(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    normalised = frame.copy()
    log: list[dict] = []

    for column in ("primary_intent", "secondary_intent"):
        for row in normalised.index:
            raw = str(normalised.at[row, column]).strip()
            if raw in INTENT_NORMALISATIONS:
                log.append({
                    "annotation_id": normalised.at[row, "annotation_id"],
                    "column": column, "from": raw, "to": INTENT_NORMALISATIONS[raw],
                    "rule": "intent_typo",
                })
                normalised.at[row, column] = INTENT_NORMALISATIONS[raw]
            else:
                normalised.at[row, column] = raw

    for row in normalised.index:
        raw = str(normalised.at[row, "confidence"]).strip().lower()
        canonical = CONFIDENCE_NORMALISATIONS.get(raw, raw)
        if canonical != raw:
            log.append({
                "annotation_id": normalised.at[row, "annotation_id"],
                "column": "confidence", "from": raw, "to": canonical, "rule": "confidence_canonical",
            })
        normalised.at[row, "confidence"] = canonical

    for column in ("is_ambiguous", "needs_discussion"):
        raw_values = normalised[column].astype(str).str.strip().str.lower()
        for row in normalised.index:
            raw = raw_values.at[row]
            if raw and raw not in FLAG_TRUE_VALUES:
                log.append({
                    "annotation_id": normalised.at[row, "annotation_id"],
                    "column": column, "from": raw, "to": "ignored_non_true", "rule": "flag_unrecognised",
                })
        normalised[column] = raw_values.isin(FLAG_TRUE_VALUES)

    return normalised, log


def validate(frame: pd.DataFrame, source_texts: pd.Series) -> tuple[dict, list[str]]:
    expected_columns = [
        "annotation_id", "conversation_id", "tweet_id", "created_at", "sampling_stratum",
        "text_display", "primary_intent", "secondary_intent", "confidence",
        "is_ambiguous", "needs_discussion", "notes",
    ]
    truth = frame["conversation_id"].map(source_texts).fillna("").map(
        lambda value: NEWLINE_PATTERN.sub(" / ", str(value)).strip()
    )
    text_matches = frame["text_display"].str.strip() == truth.str.strip()
    misaligned = frame.loc[~text_matches, "annotation_id"].tolist()

    unknown_primary = sorted(set(frame["primary_intent"]) - VALID_LABELS)
    secondary_values = {value for value in frame["secondary_intent"] if value}
    unknown_secondary = sorted(secondary_values - VALID_LABELS)

    checks = {
        "row_count": len(frame),
        "columns_as_expected": list(frame.columns) == expected_columns,
        "annotation_id_unique": not frame["annotation_id"].duplicated().any(),
        "conversation_id_unique": not frame["conversation_id"].duplicated().any(),
        "no_duplicate_rows": int(frame.duplicated().sum()) == 0,
        "primary_intent_complete": int((frame["primary_intent"] == "").sum()) == 0,
        "confidence_complete": int((frame["confidence"] == "").sum()) == 0,
        "unknown_primary_labels": unknown_primary,
        "unknown_secondary_labels": unknown_secondary,
        "text_matches_source": int(text_matches.sum()),
        "text_misaligned_ids": misaligned,
        "confidence_values_seen": sorted(set(frame["confidence"])),
    }
    return checks, misaligned


def intent_table(frame: pd.DataFrame, label_column: str = "primary_intent") -> pd.DataFrame:
    counts = frame[label_column].value_counts()
    total = len(frame)
    rows = []
    for label in TAXONOMY_INTENTS + ESCAPE_LABELS:
        subset = frame[frame[label_column] == label]
        count = int(counts.get(label, 0))
        confidence = subset["confidence"].value_counts()
        rows.append({
            "intent": label,
            "n": count,
            "pct": round(100 * count / total, 1) if total else 0.0,
            "high": int(confidence.get("high", 0)),
            "medium": int(confidence.get("medium", 0)),
            "low": int(confidence.get("low", 0)),
            "pct_not_high": round(100 * (count - int(confidence.get("high", 0))) / count, 1) if count else 0.0,
            "ambiguous": int(subset["is_ambiguous"].sum()),
            "needs_discussion": int(subset["needs_discussion"].sum()),
            "with_secondary": int((subset["secondary_intent"] != "").sum()),
        })
    return pd.DataFrame(rows)


def pair_counts(frame: pd.DataFrame) -> pd.DataFrame:
    paired = frame[frame["secondary_intent"] != ""]
    pairs = Counter(
        tuple(sorted((row["primary_intent"], row["secondary_intent"])))
        for _, row in paired.iterrows()
    )
    rows = []
    for (first, second), count in pairs.most_common():
        first_total = int((frame["primary_intent"] == first).sum())
        second_total = int((frame["primary_intent"] == second).sum())
        denominator = max(first_total, second_total, 1)
        rows.append({
            "intent_a": first,
            "intent_b": second,
            "n": count,
            "pct_of_larger_intent": round(100 * count / denominator, 1),
        })
    return pd.DataFrame(rows)


def evaluate_triggers(random_frame: pd.DataFrame, table: pd.DataFrame, pairs: pd.DataFrame) -> list[dict]:
    total = len(random_frame)
    results = []

    other = int((random_frame["primary_intent"] == "OTHER").sum())
    results.append({
        "trigger": "OTHER rate above 10%",
        "observed": round(100 * other / total, 1),
        "threshold": TRIGGERS["other_rate_pct"],
        "fired": 100 * other / total > TRIGGERS["other_rate_pct"],
        "meaning": "taxonomy may have a gap",
    })

    unclear = int((random_frame["primary_intent"] == "UNCLEAR").sum())
    results.append({
        "trigger": "UNCLEAR rate above 15%",
        "observed": round(100 * unclear / total, 1),
        "threshold": TRIGGERS["unclear_rate_pct"],
        "fired": 100 * unclear / total > TRIGGERS["unclear_rate_pct"],
        "meaning": "opening messages may lack sufficient context",
    })

    rare = table[(table["intent"].isin(TAXONOMY_INTENTS)) & (table["pct"] < TRIGGERS["min_intent_share_pct"])]
    results.append({
        "trigger": "intent below 2% of random stratum",
        "observed": rare["intent"].tolist(),
        "threshold": TRIGGERS["min_intent_share_pct"],
        "fired": not rare.empty,
        "meaning": "merge-or-drop candidates",
    })

    unclear_definition = table[
        (table["intent"].isin(TAXONOMY_INTENTS))
        & (table["n"] > 0)
        & (table["pct_not_high"] > TRIGGERS["max_low_confidence_pct"])
    ]
    results.append({
        "trigger": "non-high confidence above 30% within an intent",
        "observed": unclear_definition[["intent", "pct_not_high"]].to_dict(orient="records"),
        "threshold": TRIGGERS["max_low_confidence_pct"],
        "fired": not unclear_definition.empty,
        "meaning": "intent definition may be unclear",
    })

    merge = pairs[pairs["pct_of_larger_intent"] > TRIGGERS["max_pair_cooccurrence_pct"]] if not pairs.empty else pairs
    results.append({
        "trigger": "primary/secondary pair above 15% of the larger intent",
        "observed": merge.to_dict(orient="records") if not merge.empty else [],
        "threshold": TRIGGERS["max_pair_cooccurrence_pct"],
        "fired": not merge.empty,
        "meaning": "merge candidates",
    })
    return results


def build_review_cases(frame: pd.DataFrame) -> pd.DataFrame:
    flagged = frame[
        frame["is_ambiguous"] | frame["needs_discussion"] | (frame["confidence"] != "high")
    ].copy()
    reasons = []
    for _, row in flagged.iterrows():
        parts = []
        if row["is_ambiguous"]:
            parts.append("ambiguous")
        if row["needs_discussion"]:
            parts.append("discussion")
        if row["confidence"] != "high":
            parts.append(f"confidence={row['confidence']}")
        reasons.append("+".join(parts))
    flagged["flag_reason"] = reasons
    flagged["excerpt"] = flagged["text_display"].str.slice(0, 90)
    return flagged[[
        "annotation_id", "sampling_stratum", "primary_intent", "secondary_intent",
        "confidence", "flag_reason", "excerpt", "notes",
    ]]


def markdown_table(frame: pd.DataFrame) -> str:
    def render(value):
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, int):
            return f"{value:,}"
        return str(value)

    header = "| " + " | ".join(frame.columns) + " |"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(render(value) for value in record) + " |"
        for record in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def render_report(stats: dict, random_table, targeted_table, pairs, triggers, review, source: str) -> str:
    checks = stats["validation"]
    lines = [
        "# Phase 5 Pilot Annotation Analysis",
        "",
        f"Generated by `src/analyse_annotations.py` from `{source}`. The annotator's file is "
        "opened read-only and is never modified.",
        "",
        "## 1. Dataset and annotation overview",
        "",
        f"- Rows in file: **{checks['row_count']}**",
        f"- Analysed after integrity exclusions: **{stats['analysed_rows']}** "
        f"(random {stats['random_rows']}, targeted {stats['targeted_rows']})",
        f"- `primary_intent` complete: {checks['primary_intent_complete']}",
        f"- `confidence` complete: {checks['confidence_complete']}",
        f"- Rows with a secondary intent: {stats['with_secondary']}",
        f"- Rows flagged ambiguous: {stats['ambiguous']}",
        f"- Rows flagged for discussion: {stats['needs_discussion']}",
        f"- Rows with both flags: {stats['both_flags']}",
        f"- Rows with notes: {stats['with_notes']}",
        "",
        "### Integrity checks",
        "",
        "| Check | Result |",
        "| --- | --- |",
        f"| Columns as expected | {checks['columns_as_expected']} |",
        f"| `annotation_id` unique | {checks['annotation_id_unique']} |",
        f"| `conversation_id` unique | {checks['conversation_id_unique']} |",
        f"| No duplicate rows | {checks['no_duplicate_rows']} |",
        f"| Unknown primary labels (after normalisation) | {checks['unknown_primary_labels'] or 'none'} |",
        f"| Unknown secondary labels (after normalisation) | {checks['unknown_secondary_labels'] or 'none'} |",
        f"| `text_display` matches source | {checks['text_matches_source']} / {checks['row_count']} |",
        f"| Misaligned rows excluded | {checks['text_misaligned_ids'] or 'none'} |",
        "",
        "### Normalisations applied in analysis only",
        "",
        "The annotator's CSV is unchanged on disk. These rules are deterministic and every "
        "affected row is listed.",
        "",
    ]
    if stats["normalisations"]:
        lines += ["| annotation_id | column | from | to | rule |", "| --- | --- | --- | --- | --- |"]
        lines += [
            f"| {item['annotation_id']} | {item['column']} | `{item['from']}` | `{item['to']}` | {item['rule']} |"
            for item in stats["normalisations"]
            if item["rule"] != "confidence_canonical"
        ]
        confidence_fixes = sum(1 for item in stats["normalisations"] if item["rule"] == "confidence_canonical")
        if confidence_fixes:
            lines += [f"| _{confidence_fixes} rows_ | confidence | `med` | `medium` | confidence_canonical |"]
    else:
        lines.append("_No normalisations were required._")

    lines += [
        "",
        f"Flag values `{sorted(stats['flag_values_seen'])}` were treated as true. "
        "The guidelines asked for `y`; `yes` is the natural thing to type and was accepted.",
        "",
        "## 2. Primary intent distribution",
        "",
        "> Prevalence claims use the **random stratum only**. The targeted stratum deliberately "
        "over-samples rare intents and is reported separately (decision D28).",
        "",
        f"### Random stratum (n={stats['random_rows']}) — representative",
        "",
        markdown_table(random_table),
        "",
        f"### Targeted stratum (n={stats['targeted_rows']}) — NOT representative",
        "",
        markdown_table(targeted_table),
        "",
        "## 3. Secondary intent distribution",
        "",
        f"{stats['with_secondary']} of {stats['analysed_rows']} rows carry a secondary intent.",
        "",
    ]
    lines += [markdown_table(pairs)] if not pairs.empty else ["_No secondary intents recorded._"]

    lines += [
        "",
        "## 4. Confidence / ambiguity / discussion statistics",
        "",
        "| Metric | Random | Targeted | All |",
        "| --- | --- | --- | --- |",
    ]
    for label, key in [("high confidence", "high"), ("medium confidence", "medium"), ("low confidence", "low")]:
        lines.append(
            f"| {label} | {stats['confidence_random'].get(key, 0)} | "
            f"{stats['confidence_targeted'].get(key, 0)} | {stats['confidence_all'].get(key, 0)} |"
        )
    lines += [
        f"| ambiguous | {stats['ambiguous_random']} | {stats['ambiguous'] - stats['ambiguous_random']} | {stats['ambiguous']} |",
        f"| needs discussion | {stats['discussion_random']} | {stats['needs_discussion'] - stats['discussion_random']} | {stats['needs_discussion']} |",
        "",
        "## 5. Taxonomy health",
        "",
        "See the per-intent tables in section 2. Trigger evaluation, against thresholds declared "
        "in `docs/DECISION_LOG.md` D29 **before any label existed**:",
        "",
        "| Trigger | Threshold | Observed | Fired |",
        "| --- | --- | --- | --- |",
    ]
    for item in triggers:
        observed = item["observed"]
        rendered = observed if isinstance(observed, (int, float, str)) else json.dumps(observed)
        if len(str(rendered)) > 90:
            rendered = str(rendered)[:87] + "..."
        lines.append(
            f"| {item['trigger']} | {item['threshold']} | {rendered} | "
            f"**{'YES' if item['fired'] else 'no'}** |"
        )

    lines += [
        "",
        "## 6. Ambiguous-case analysis",
        "",
        f"{len(review)} rows carry at least one of: ambiguity flag, discussion flag, or "
        "non-high confidence. Full list with excerpts in `reports/phase5_ambiguous_cases.csv`.",
        "",
        "Flag-reason breakdown:",
        "",
        "| Reason combination | n |",
        "| --- | --- |",
    ]
    for reason, count in Counter(review["flag_reason"]).most_common():
        lines.append(f"| {reason} | {count} |")

    lines += [
        "",
        "## 7. OTHER and UNCLEAR analysis",
        "",
        f"- `OTHER` in random stratum: **{stats['other_random']}** "
        f"({round(100 * stats['other_random'] / max(stats['random_rows'], 1), 1)}%)",
        f"- `UNCLEAR` in random stratum: **{stats['unclear_random']}** "
        f"({round(100 * stats['unclear_random'] / max(stats['random_rows'], 1), 1)}%)",
        "",
        "Notes recorded against `OTHER` rows (verbatim, deduplicated):",
        "",
    ]
    lines += [f"- {note}" for note in stats["other_notes"]] or ["_none recorded_"]
    lines += [
        "",
        "Notes recorded against `UNCLEAR` rows:",
        "",
    ]
    lines += [f"- {note}" for note in stats["unclear_notes"]] or ["_none recorded_"]

    lines += [
        "",
        "## 8. Taxonomy recommendations",
        "",
        "_Authored separately after reading the evidence above; see the Phase 5d section of "
        "`docs/DEVELOPMENT_LOG.md`. This generated report deliberately stops at evidence._",
        "",
        "## 9. Sampling observations",
        "",
        f"- Seed: {stats['manifest']['seed']}",
        f"- Population: {stats['manifest']['population']:,}",
        f"- Random drawn: {stats['manifest']['random_drawn']} "
        f"(month split {stats['manifest']['random_month_distribution']})",
        f"- Targeted drawn: {stats['manifest']['targeted_drawn']}",
        f"- Manifest matches labelled file: **{stats['manifest_matches']}**",
        "",
        "## 10. Annotation limitations",
        "",
        "- **One annotator, one pass.** Neither inter-annotator nor intra-annotator agreement "
        "can be computed from this pilot. No agreement score is reported, because none exists.",
        "- A blind test-retest would require the same annotator relabelling a shuffled subset "
        "(40-50 rows) after a gap, with prior labels hidden. Cohen's kappa would then be "
        "computable and would still be an **upper bound** on reliability, since self-agreement "
        "exceeds agreement between different people.",
        "- Opening messages only; context from later turns was deliberately withheld.",
        "- The targeted stratum is keyword-probed and cannot support prevalence claims.",
        "",
        "## 11. Decision: freeze taxonomy or revise it",
        "",
        "_Recorded in `docs/DECISION_LOG.md` after review with the annotator._",
        "",
        "## 12. Recommended next step",
        "",
        "_See the Phase 5d section of `docs/DEVELOPMENT_LOG.md`._",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyse a completed annotation batch.")
    parser.add_argument("--batch", default="b01")
    args = parser.parse_args()

    labelled_path = GOLDEN_DIR / f"{args.batch}_pilot_labelled.csv"
    if not labelled_path.exists():
        raise FileNotFoundError(f"Completed annotation file not found at {labelled_path}")

    print(f"[1/5] Reading {labelled_path.name} (read-only) ...", flush=True)
    frame = pd.read_csv(labelled_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    flag_values_seen = sorted(
        {value.strip().lower() for column in ("is_ambiguous", "needs_discussion")
         for value in frame[column] if value.strip()}
    )

    print("[2/5] Normalising (analysis only) and validating ...", flush=True)
    normalised, normalisation_log = normalise(frame)
    source_texts = load_source_texts(set(normalised["conversation_id"]))
    checks, misaligned = validate(normalised, source_texts)

    if checks["unknown_primary_labels"]:
        raise ValueError(f"Unrecognised primary labels after normalisation: {checks['unknown_primary_labels']}")

    analysed = normalised[~normalised["annotation_id"].isin(misaligned)].copy()
    random_frame = analysed[analysed["sampling_stratum"] == "random"]
    targeted_frame = analysed[analysed["sampling_stratum"] == "targeted"]
    print(f"      {len(analysed)} analysed ({len(misaligned)} excluded for text misalignment)", flush=True)

    print("[3/5] Building tables ...", flush=True)
    random_table = intent_table(random_frame)
    targeted_table = intent_table(targeted_frame)
    pairs = pair_counts(analysed)
    triggers = evaluate_triggers(random_frame, random_table, pairs)
    review = build_review_cases(analysed)

    manifest = json.loads(
        (config.REPORTS_DIR / f"phase5_{args.batch}_sampling_manifest.json").read_text(encoding="utf-8")
    )

    print("[4/5] Assembling statistics ...", flush=True)
    stats = {
        "source": str(labelled_path.relative_to(config.PROJECT_ROOT)),
        "validation": checks,
        "normalisations": normalisation_log,
        "flag_values_seen": flag_values_seen,
        "analysed_rows": len(analysed),
        "random_rows": len(random_frame),
        "targeted_rows": len(targeted_frame),
        "with_secondary": int((analysed["secondary_intent"] != "").sum()),
        "ambiguous": int(analysed["is_ambiguous"].sum()),
        "ambiguous_random": int(random_frame["is_ambiguous"].sum()),
        "needs_discussion": int(analysed["needs_discussion"].sum()),
        "discussion_random": int(random_frame["needs_discussion"].sum()),
        "both_flags": int((analysed["is_ambiguous"] & analysed["needs_discussion"]).sum()),
        "with_notes": int((analysed["notes"] != "").sum()),
        "confidence_all": analysed["confidence"].value_counts().to_dict(),
        "confidence_random": random_frame["confidence"].value_counts().to_dict(),
        "confidence_targeted": targeted_frame["confidence"].value_counts().to_dict(),
        "other_random": int((random_frame["primary_intent"] == "OTHER").sum()),
        "unclear_random": int((random_frame["primary_intent"] == "UNCLEAR").sum()),
        "other_notes": sorted({n for n in analysed.loc[analysed["primary_intent"] == "OTHER", "notes"] if n}),
        "unclear_notes": sorted({n for n in analysed.loc[analysed["primary_intent"] == "UNCLEAR", "notes"] if n}),
        "random_intent_table": random_table.to_dict(orient="records"),
        "targeted_intent_table": targeted_table.to_dict(orient="records"),
        "secondary_pairs": pairs.to_dict(orient="records"),
        "triggers": triggers,
        "manifest": manifest,
        "manifest_matches": bool(
            manifest["random_drawn"] == int((normalised["sampling_stratum"] == "random").sum())
            and manifest["targeted_drawn"] == int((normalised["sampling_stratum"] == "targeted").sum())
        ),
        "agreement": "not computable: one annotator, one pass",
    }

    print("[5/5] Writing reports ...", flush=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "phase5_pilot_stats.json").write_text(
        json.dumps(stats, indent=2, default=str), encoding="utf-8"
    )
    review.to_csv(config.REPORTS_DIR / "phase5_ambiguous_cases.csv", index=False, encoding="utf-8-sig")
    (config.REPORTS_DIR / "phase5_pilot_analysis.md").write_text(
        render_report(stats, random_table, targeted_table, pairs, triggers, review, labelled_path.name),
        encoding="utf-8",
    )

    print(f"\nReport:    {config.REPORTS_DIR / 'phase5_pilot_analysis.md'}")
    print(f"Stats:     {config.REPORTS_DIR / 'phase5_pilot_stats.json'}")
    print(f"Review:    {config.REPORTS_DIR / 'phase5_ambiguous_cases.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
