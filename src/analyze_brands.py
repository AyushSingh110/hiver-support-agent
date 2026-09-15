from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src import config

SELECTED_BRAND = "AmericanAir"

MIN_CLEAN_CONVERSATIONS = 8_000
MAX_DM_DEFLECTION_PCT = 40.0
MIN_ENGLISH_PCT = 95.0
MAX_BRAND_ROOTED_PCT = 5.0

SCORING_WEIGHTS = {
    "public_resolution": 0.30,
    "depth": 0.20,
    "volume": 0.20,
    "diversity": 0.15,
    "language": 0.10,
    "support_purity": 0.05,
}

DM_PATTERN = re.compile(r"\b(?:dm|direct message|private message|pm us|inbox)\b", re.I)
URL_PATTERN = re.compile(r"https?://")
MENTION_PATTERN = re.compile(r"@\w+")
TOKEN_PATTERN = re.compile(r"[a-z']+")
CJK_PATTERN = re.compile(r"[぀-ヿ一-鿿가-힯]")
ACCENTED_PATTERN = re.compile(r"[À-ɏ]")
NON_ENGLISH_WORD_PATTERN = re.compile(
    r"\b(?:que|por|para|nicht|ich|und|ist|der|die|das|une|les|pour|est|nao|"
    r"obrigado|gracias|bitte|danke|merci)\b",
    re.I,
)

# TTR shrinks as sample size grows, so every brand is measured on the same budget.
TTR_SAMPLE_SIZE = 5_000


def load_clean_conversations(conversations_path: Path) -> pd.DataFrame:
    conversations = pd.read_parquet(conversations_path)
    clean = conversations[conversations["status"] == "clean"]
    if clean.empty:
        raise ValueError(f"No clean conversations found in {conversations_path}")
    return clean


def collect_turn_signals(turns_path: Path, brands: set[str]):
    # A conversation counts as DM-deflected if ANY brand turn redirects to DM.
    per_conversation = []
    openings = {brand: Counter() for brand in brands}
    opening_counts = {brand: 0 for brand in brands}
    language = {
        brand: {"openings": 0, "cjk": 0, "accented": 0, "non_english_word": 0}
        for brand in brands
    }

    for batch in pq.ParquetFile(turns_path).iter_batches(
        batch_size=250_000,
        columns=["conversation_id", "brand", "speaker", "text", "turn_index"],
    ):
        block = batch.to_pandas()
        block = block[block["brand"].isin(brands)]
        if block.empty:
            continue

        brand_turns = block[block["speaker"] == "brand"]
        if not brand_turns.empty:
            text = brand_turns["text"].fillna("")
            per_conversation.append(
                pd.DataFrame(
                    {
                        "conversation_id": brand_turns["conversation_id"],
                        "dm": text.map(lambda value: bool(DM_PATTERN.search(value))),
                        "url": text.map(lambda value: bool(URL_PATTERN.search(value))),
                        "length": text.str.len(),
                    }
                )
            )

        roots = block[block["turn_index"] == 0]
        per_conversation.append(
            pd.DataFrame(
                {
                    "conversation_id": roots["conversation_id"],
                    "root_speaker": roots["speaker"],
                }
            )
        )

        customer_roots = roots[roots["speaker"] == "customer"]
        for brand, group in customer_roots.groupby("brand"):
            stripped = group["text"].fillna("").map(
                lambda value: MENTION_PATTERN.sub(" ", URL_PATTERN.sub(" ", str(value)))
            )
            counters = language[brand]
            counters["openings"] += len(group)
            counters["cjk"] += int(stripped.map(lambda v: bool(CJK_PATTERN.search(v))).sum())
            counters["accented"] += int(stripped.map(lambda v: bool(ACCENTED_PATTERN.search(v))).sum())
            counters["non_english_word"] += int(
                stripped.map(lambda v: bool(NON_ENGLISH_WORD_PATTERN.search(v))).sum()
            )

            budget = TTR_SAMPLE_SIZE - opening_counts[brand]
            if budget > 0:
                for value in stripped.head(budget):
                    openings[brand].update(TOKEN_PATTERN.findall(value.lower()))
                opening_counts[brand] += min(budget, len(group))

    combined = pd.concat(per_conversation, ignore_index=True)
    signals = combined.groupby("conversation_id").agg(
        any_dm=("dm", "any"),
        any_url=("url", "any"),
        mean_brand_length=("length", "mean"),
        root_speaker=("root_speaker", "first"),
    )
    return signals, openings, opening_counts, language


def brand_metrics(clean: pd.DataFrame, signals, openings, opening_counts, language) -> pd.DataFrame:
    # Signals are only collected for the analysed brands; others would join to nulls.
    analysed = clean[clean["brand"].isin(language.keys())]
    joined = analysed.set_index("conversation_id").join(signals)
    rows = []
    for brand, group in joined.groupby("brand"):
        counters = language[brand]
        openings_seen = max(counters["openings"], 1)
        tokens = openings[brand]
        total_tokens = sum(tokens.values())

        customer_rooted = group["root_speaker"] == "customer"
        retrieval_pool = group[
            customer_rooted
            & (~group["root_truncated"])
            & (group["n_turns"] >= 2)
            & (group["n_brand_turns"] >= 1)
        ]
        non_english = counters["cjk"] + counters["non_english_word"]

        rows.append(
            {
                "brand": brand,
                "clean_conversations": len(group),
                "clean_turns": int(group["n_turns"].sum()),
                "median_turns": float(group["n_turns"].median()),
                "pct_4plus_turns": round(100 * float((group["n_turns"] >= 4).mean()), 1),
                "pct_2plus_brand_turns": round(100 * float((group["n_brand_turns"] >= 2).mean()), 1),
                "pct_dm_deflected": round(100 * float(group["any_dm"].fillna(False).mean()), 1),
                "pct_with_url": round(100 * float(group["any_url"].fillna(False).mean()), 1),
                "mean_brand_reply_chars": round(float(group["mean_brand_length"].mean()), 1),
                "pct_brand_rooted": round(100 * float((group["root_speaker"] == "brand").mean()), 2),
                "pct_truncated_roots": round(100 * float(group["root_truncated"].mean()), 2),
                "pct_non_english_signal": round(100 * non_english / openings_seen, 1),
                "pct_accented_signal": round(100 * counters["accented"] / openings_seen, 1),
                "type_token_ratio": round(len(tokens) / max(total_tokens, 1), 4),
                "ttr_sample_openings": opening_counts[brand],
                "retrieval_pool": len(retrieval_pool),
                "median_duration_minutes": round(float(group["duration_minutes"].median()), 1),
                "months_covered": int(group["started_at"].dt.strftime("%Y-%m").nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values("clean_conversations", ascending=False).reset_index(drop=True)


def apply_gates(metrics: pd.DataFrame) -> pd.DataFrame:
    gated = metrics.copy()
    gated["pct_english"] = (100 - gated["pct_non_english_signal"]).round(1)
    gated["g1_volume"] = gated["clean_conversations"] >= MIN_CLEAN_CONVERSATIONS
    gated["g2_public_resolution"] = gated["pct_dm_deflected"] < MAX_DM_DEFLECTION_PCT
    gated["g3_language"] = gated["pct_english"] >= MIN_ENGLISH_PCT
    gated["g4_support_purity"] = gated["pct_brand_rooted"] < MAX_BRAND_ROOTED_PCT
    gate_columns = ["g1_volume", "g2_public_resolution", "g3_language", "g4_support_purity"]
    gated["passes_all_gates"] = gated[gate_columns].all(axis=1)
    gated["failed_gates"] = gated[gate_columns].apply(
        lambda row: ", ".join(name for name, passed in row.items() if not passed) or "-", axis=1
    )
    return gated


def score_candidates(gated: pd.DataFrame) -> pd.DataFrame:
    survivors = gated[gated["passes_all_gates"]].copy()
    if survivors.empty:
        return survivors

    criteria = {
        "public_resolution": 100 - survivors["pct_dm_deflected"],
        "depth": survivors["pct_4plus_turns"],
        "volume": survivors["clean_conversations"],
        "diversity": survivors["type_token_ratio"],
        "language": survivors["pct_english"],
        "support_purity": 100 - survivors["pct_brand_rooted"],
    }

    # Min-max within the surviving set, so the score reflects measurements rather
    # than hand-assigned points. Meaningful only as a relative ranking.
    score = pd.Series(0.0, index=survivors.index)
    for name, values in criteria.items():
        spread = values.max() - values.min()
        normalised = (values - values.min()) / spread if spread else pd.Series(1.0, index=values.index)
        survivors[f"norm_{name}"] = normalised.round(3)
        score += SCORING_WEIGHTS[name] * normalised
    survivors["score"] = score.round(4)
    return survivors.sort_values("score", ascending=False).reset_index(drop=True)


def summarise_selected_brand(clean: pd.DataFrame, signals, brand: str) -> dict:
    joined = clean[clean["brand"] == brand].set_index("conversation_id").join(signals)
    if joined.empty:
        return {}

    customer_rooted = joined["root_speaker"] == "customer"
    retrieval_pool = joined[
        customer_rooted
        & (~joined["root_truncated"])
        & (joined["n_turns"] >= 2)
        & (joined["n_brand_turns"] >= 1)
    ]
    monthly = joined["started_at"].dt.strftime("%Y-%m").value_counts().sort_index()

    return {
        "brand": brand,
        "clean_conversations": len(joined),
        "clean_turns": int(joined["n_turns"].sum()),
        "customer_turns": int(joined["n_customer_turns"].sum()),
        "brand_turns": int(joined["n_brand_turns"].sum()),
        "customer_rooted_conversations": int(customer_rooted.sum()),
        "truncated_conversations": int(joined["root_truncated"].sum()),
        "multi_turn_conversations": int((joined["n_turns"] >= 4).sum()),
        "two_turn_conversations": int((joined["n_turns"] == 2).sum()),
        "retrieval_pool": len(retrieval_pool),
        "candidate_opening_messages": int(customer_rooted.sum()),
        "unique_customers": int(joined["n_customers"].sum()),
        "median_turns": float(joined["n_turns"].median()),
        "median_duration_minutes": round(float(joined["duration_minutes"].median()), 1),
        "earliest": str(joined["started_at"].min()),
        "latest": str(joined["started_at"].max()),
        "monthly_distribution": {month: int(count) for month, count in monthly.items()},
    }


def markdown_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(frame.columns) + " |"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
    def render(value):
        # bool is a subclass of int, so it must be checked first or it prints as 0/1.
        if isinstance(value, (bool, np.bool_)):
            return "yes" if value else "no"
        if isinstance(value, (int, np.integer)):
            return f"{value:,}"
        return str(value)

    rows = [
        "| " + " | ".join(render(value) for value in record) + " |"
        for record in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def render_report(metrics, gated, scored, selected, source: str) -> str:
    shortlist_columns = [
        "brand", "clean_conversations", "retrieval_pool", "median_turns",
        "pct_4plus_turns", "pct_2plus_brand_turns", "pct_dm_deflected",
        "pct_with_url", "type_token_ratio", "pct_english", "months_covered",
    ]
    gate_columns = ["brand", "clean_conversations", "pct_dm_deflected", "pct_english",
                    "pct_brand_rooted", "passes_all_gates", "failed_gates"]

    lines = [
        f"# Phase 3 — Brand selection (`{source}`)",
        "",
        "Generated by `src/analyze_brands.py`. Intent discovery is **not** part of this phase.",
        "",
        "## Objective",
        "",
        "Choose one brand for the support agent using measured evidence rather than dataset size.",
        "",
        "## Method",
        "",
        "Four screening gates, then a weighted score over the survivors. Gates come first "
        "because some weaknesses are disqualifying no matter how strong a brand looks elsewhere.",
        "",
        f"- **G1 Volume** — at least {MIN_CLEAN_CONVERSATIONS:,} clean conversations",
        f"- **G2 Public resolution** — conversation-level DM deflection below {MAX_DM_DEFLECTION_PCT}%",
        f"- **G3 Language** — at least {MIN_ENGLISH_PCT}% English signal",
        f"- **G4 Support purity** — fewer than {MAX_BRAND_ROOTED_PCT}% brand-rooted conversations",
        "",
        "### Measurement definitions",
        "",
        "| Metric | Type | Definition |",
        "| --- | --- | --- |",
        "| DM deflection | Derived | A clean conversation counts as deflected if **any** brand "
        f"turn matches `{DM_PATTERN.pattern}`. Conversation-level, not per tweet. |",
        "| Non-English signal | **Heuristic** | Opening message contains CJK characters or a "
        "common non-English function word. Not a language detector. |",
        "| Accented signal | **Heuristic** | Reported separately and *not* used in the gate, "
        "because English tweets routinely contain accented characters and typographic marks. |",
        f"| Type-token ratio | Heuristic | Unique / total tokens over the first {TTR_SAMPLE_SIZE:,} "
        "opening messages per brand. A fixed budget is used because TTR falls as sample size "
        "grows, so unequal samples would not be comparable. |",
        "| Retrieval pool | Derived | Customer-rooted, untruncated, 2+ turns, at least one brand reply. |",
        "",
        "## Gate results",
        "",
        markdown_table(gated[gate_columns]),
        "",
        "## Full brand metrics",
        "",
        markdown_table(metrics[shortlist_columns] if "pct_english" in metrics.columns
                       else gated[shortlist_columns]),
        "",
        "## Weighted score (gate survivors only)",
        "",
        "Each criterion is min-max normalised **within the surviving set**, then weighted. "
        "The score is a relative ranking aid, not the basis of the decision.",
        "",
        "| Criterion | Weight | Why it matters |",
        "| --- | --- | --- |",
        "| Public resolution | 30% | If the fix happens in DM there is no visible evidence to ground a reply on |",
        "| Depth | 20% | Multi-turn threads carry retrievable resolution patterns |",
        "| Volume | 20% | Golden set, retrieval corpus and held-out evaluation must all be disjoint |",
        "| Diversity | 15% | A taxonomy needs 8-12 distinguishable intents |",
        "| Language | 10% | Mixed languages would confound classification, retrieval and judging at once |",
        "| Support purity | 5% | Low variance across candidates, so a small weight |",
        "",
    ]
    if scored.empty:
        lines += ["_No brand passed all gates in this source._", ""]
    else:
        lines += [markdown_table(scored[["brand", "score"] + shortlist_columns[1:6]]), ""]

    lines += [
        "## Selected brand",
        "",
    ]
    if selected:
        lines += [
            f"**{selected['brand']}**",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Clean conversations | {selected['clean_conversations']:,} |",
            f"| Clean turns | {selected['clean_turns']:,} |",
            f"| Customer turns | {selected['customer_turns']:,} |",
            f"| Brand turns | {selected['brand_turns']:,} |",
            f"| Customer-rooted conversations | {selected['customer_rooted_conversations']:,} |",
            f"| Retrieval pool | {selected['retrieval_pool']:,} |",
            f"| Candidate opening messages | {selected['candidate_opening_messages']:,} |",
            f"| Multi-turn (4+) conversations | {selected['multi_turn_conversations']:,} |",
            f"| Two-turn conversations | {selected['two_turn_conversations']:,} |",
            f"| Truncated conversations | {selected['truncated_conversations']:,} |",
            f"| Median turns | {selected['median_turns']} |",
            f"| Median duration (minutes) | {selected['median_duration_minutes']} |",
            f"| Date range | {selected['earliest']} to {selected['latest']} |",
            "",
            "### Monthly distribution",
            "",
            "| Month | Conversations |",
            "| --- | --- |",
            *[f"| {month} | {count:,} |" for month, count in selected["monthly_distribution"].items()],
            "",
        ]
    else:
        lines += [f"_`{SELECTED_BRAND}` is not present in this source._", ""]

    lines += [
        "## Limitations",
        "",
        "- Language measurement is a heuristic, not a language detector. It will miss "
        "unaccented non-English text and may flag English tweets quoting foreign words.",
        "- Type-token ratio is a crude proxy for issue diversity and is sensitive to templated text.",
        "- DM deflection detects the *redirect*, not whether the issue was actually resolved privately.",
        "- The weighted score depends on chosen weights; different weights reorder close candidates.",
        "- Metrics cover `clean` conversations only, which is 83.94% of all tweets.",
        "",
    ]
    return "\n".join(lines)


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if value is pd.NA or value is pd.NaT:
        return None
    raise TypeError(f"Cannot serialise {type(value)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Select the support brand from reconstructed data.")
    parser.add_argument("--source", choices=["sample", "full"], required=True)
    args = parser.parse_args()

    suffix = "_sample" if args.source == "sample" else ""
    conversations_path = config.PROCESSED_DIR / f"conversations{suffix}.parquet"
    turns_path = config.PROCESSED_DIR / f"conversation_turns{suffix}.parquet"
    for path in (conversations_path, turns_path):
        if not path.exists():
            raise FileNotFoundError(f"Phase 2 output missing at {path}; run src.reconstruct first")

    print("[1/3] Loading clean conversations ...", flush=True)
    clean = load_clean_conversations(conversations_path)
    brands = set(clean["brand"].value_counts().head(config.TOP_BRAND_COUNT).index)
    print(f"      {len(clean):,} clean conversations, analysing {len(brands)} brands", flush=True)

    print("[2/3] Scanning turns for DM, URL and language signals ...", flush=True)
    signals, openings, opening_counts, language = collect_turn_signals(turns_path, brands)

    print("[3/3] Scoring and reporting ...", flush=True)
    metrics = brand_metrics(clean, signals, openings, opening_counts, language)
    gated = apply_gates(metrics)
    scored = score_candidates(gated)
    selected = summarise_selected_brand(clean, signals, SELECTED_BRAND)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stats = {
        "source": str(conversations_path),
        "selected_brand": SELECTED_BRAND,
        "gates": {
            "min_clean_conversations": MIN_CLEAN_CONVERSATIONS,
            "max_dm_deflection_pct": MAX_DM_DEFLECTION_PCT,
            "min_english_pct": MIN_ENGLISH_PCT,
            "max_brand_rooted_pct": MAX_BRAND_ROOTED_PCT,
        },
        "scoring_weights": SCORING_WEIGHTS,
        "dm_pattern": DM_PATTERN.pattern,
        "ttr_sample_size": TTR_SAMPLE_SIZE,
        "brand_metrics": gated.to_dict(orient="records"),
        "scored_candidates": scored.to_dict(orient="records"),
        "selected_brand_corpus": selected,
    }

    stats_path = config.REPORTS_DIR / f"phase3_brand_stats{suffix}.json"
    report_path = config.REPORTS_DIR / f"phase3_brand_analysis{suffix}.md"
    stats_path.write_text(json.dumps(stats, indent=2, default=json_default), encoding="utf-8")
    report_path.write_text(
        render_report(gated, gated, scored, selected, conversations_path.name), encoding="utf-8"
    )

    print(f"\nReport: {report_path}")
    print(f"Stats:  {stats_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
