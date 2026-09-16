from __future__ import annotations

import json
import sys

import pandas as pd
import pyarrow.parquet as pq

from src import config
from src.leakage import assert_isolated, exclude_golden
from src.taxonomy import weak_label


def load_conversations() -> pd.DataFrame:
    conversations = pd.read_parquet(config.PROCESSED_DIR / "conversations.parquet")
    return conversations[
        (conversations["status"] == "clean") & (conversations["brand"] == config.BRAND)
    ]


def load_turns(conversation_ids: set[str]) -> pd.DataFrame:
    parts = []
    for batch in pq.ParquetFile(config.PROCESSED_DIR / "conversation_turns.parquet").iter_batches(
        batch_size=250_000,
        columns=["conversation_id", "tweet_id", "brand", "speaker", "author_id",
                 "text", "turn_index", "created_at"],
    ):
        block = batch.to_pandas()
        block = block[
            (block["brand"] == config.BRAND) & (block["conversation_id"].isin(conversation_ids))
        ]
        if not block.empty:
            parts.append(block.drop(columns="brand"))
    return pd.concat(parts, ignore_index=True).sort_values(["conversation_id", "turn_index"])


def build_records(conversations: pd.DataFrame, turns: pd.DataFrame) -> pd.DataFrame:
    """One row per conversation: the opening issue plus the brand's reply text."""
    openings = turns[(turns["turn_index"] == 0) & (turns["speaker"] == "customer")]
    brand_turns = turns[turns["speaker"] == "brand"].sort_values(["conversation_id", "turn_index"])
    brand_reply = brand_turns.groupby("conversation_id")["text"].apply(lambda s: "\n".join(s))
    customer_followups = (
        turns[(turns["speaker"] == "customer") & (turns["turn_index"] > 0)]
        .groupby("conversation_id").size()
    )

    records = openings[["conversation_id", "tweet_id", "author_id", "text", "created_at"]].rename(
        columns={"author_id": "customer_author_id", "text": "customer_message"}
    )
    records["brand_reply"] = records["conversation_id"].map(brand_reply).fillna("")
    records["customer_followups"] = (
        records["conversation_id"].map(customer_followups).fillna(0).astype(int)
    )
    meta = conversations.set_index("conversation_id")
    for column in ("n_turns", "n_brand_turns", "root_truncated", "duration_minutes"):
        records[column] = records["conversation_id"].map(meta[column])

    # `text` is what leakage control inspects; keep it aligned with the golden column.
    records["text"] = records["customer_message"]
    return records.reset_index(drop=True)


def apply_eligibility(records: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    steps = {"start": len(records)}

    eligible = records[~records["root_truncated"].astype(bool)]
    steps["after_dropping_truncated_roots"] = len(eligible)

    eligible = eligible[eligible["brand_reply"].str.strip() != ""]
    steps["after_requiring_a_brand_reply"] = len(eligible)

    eligible, leakage = exclude_golden(eligible)
    steps["after_leakage_exclusion"] = len(eligible)

    assert_isolated(eligible)
    return eligible.reset_index(drop=True), {"steps": steps, "leakage": leakage}


def attach_weak_labels(records: pd.DataFrame) -> pd.DataFrame:
    labels, scores = zip(*(weak_label(text) for text in records["customer_message"]))
    labelled = records.copy()
    labelled["weak_label"] = labels
    labelled["weak_label_matched_intents"] = [len(s) for s in scores]
    return labelled


def main() -> int:
    print("[1/4] Loading conversations and turns ...", flush=True)
    conversations = load_conversations()
    turns = load_turns(set(conversations["conversation_id"]))
    records = build_records(conversations, turns)
    print(f"      {len(records):,} {config.BRAND} conversations", flush=True)

    print("[2/4] Applying eligibility and leakage exclusion ...", flush=True)
    corpus, audit = apply_eligibility(records)
    print(f"      {len(corpus):,} eligible after exclusions", flush=True)

    print("[3/4] Weak-labelling for classifier training ...", flush=True)
    corpus = attach_weak_labels(corpus)
    labelled = corpus[corpus["weak_label"].notna()]
    print(f"      {len(labelled):,} weakly labelled ({100 * len(labelled) / len(corpus):.1f}%)",
          flush=True)

    print("[4/4] Writing artifacts ...", flush=True)
    config.CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    corpus_path = config.CORPUS_DIR / "retrieval_corpus.parquet"
    training_path = config.CORPUS_DIR / "weak_training_set.parquet"
    corpus.to_parquet(corpus_path, index=False)
    labelled.to_parquet(training_path, index=False)

    stats = {
        "brand": config.BRAND,
        "eligibility": audit["steps"],
        "leakage_control": audit["leakage"],
        "retrieval_corpus_rows": len(corpus),
        "weak_training_rows": len(labelled),
        "weak_label_coverage_pct": round(100 * len(labelled) / len(corpus), 2),
        "weak_label_distribution": labelled["weak_label"].value_counts().to_dict(),
        "turn_count_distribution": corpus["n_turns"].value_counts().sort_index().head(10).to_dict(),
        "conversations_with_customer_followup": int((corpus["customer_followups"] > 0).sum()),
        "outputs": {
            "retrieval_corpus": str(corpus_path.relative_to(config.PROJECT_ROOT)),
            "weak_training_set": str(training_path.relative_to(config.PROJECT_ROOT)),
        },
    }
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "phase6_corpus_stats.json").write_text(
        json.dumps(stats, indent=2, default=str), encoding="utf-8"
    )

    print(f"\nCorpus   : {corpus_path} ({len(corpus):,} rows)")
    print(f"Training : {training_path} ({len(labelled):,} rows)")
    print(f"Stats    : {config.REPORTS_DIR / 'phase6_corpus_stats.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
