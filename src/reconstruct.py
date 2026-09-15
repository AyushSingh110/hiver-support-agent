from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src import config

GRAPH_COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "response_tweet_id",
    "in_response_to_tweet_id",
]

TURNS_SCHEMA = pa.schema(
    [
        ("conversation_id", pa.string()),
        ("tweet_id", pa.int64()),
        ("author_id", pa.string()),
        ("speaker", pa.string()),
        ("brand", pa.string()),
        ("created_at", pa.timestamp("us", tz="UTC")),
        ("text", pa.string()),
        ("parent_tweet_id", pa.int64()),
        ("turn_index", pa.int32()),
    ]
)


def build_parent_index(tweet_id: np.ndarray, in_response_to: pd.Series):
    if pd.Series(tweet_id).duplicated().any():
        raise ValueError("duplicate tweet_id in source; parent lookup would be ambiguous")

    position = pd.Series(np.arange(len(tweet_id)), index=tweet_id)
    parent_tweet_id = pd.to_numeric(in_response_to, errors="coerce")
    located = position.reindex(parent_tweet_id).to_numpy(dtype="float64")

    resolved = ~np.isnan(located)
    parent_idx = np.where(resolved, located, -1).astype(np.int64)
    declared = parent_tweet_id.notna().to_numpy()
    return parent_idx, declared, resolved


def find_roots(parent_idx: np.ndarray, max_rounds: int = 64) -> np.ndarray:
    # Pointer doubling: each round replaces a pointer with its parent's pointer,
    # so reach doubles. Roots point at themselves, which halts the walk.
    pointer = np.where(parent_idx >= 0, parent_idx, np.arange(len(parent_idx)))
    for _ in range(max_rounds):
        following = pointer[pointer]
        if np.array_equal(following, pointer):
            return pointer
        pointer = following
    raise RuntimeError("root finding did not converge; the parent graph may contain a cycle")


def compute_depth(parent_idx: np.ndarray) -> np.ndarray:
    depth = np.zeros(len(parent_idx), dtype=np.int32)
    active = np.flatnonzero(parent_idx >= 0)
    current = parent_idx[active]
    while active.size:
        depth[active] += 1
        following = parent_idx[current]
        keep = following >= 0
        active, current = active[keep], following[keep]
    return depth


def classify_participants(root_idx: np.ndarray, author_id: np.ndarray, is_customer: np.ndarray):
    frame = pd.DataFrame({"root": root_idx, "author": author_id, "is_customer": is_customer})
    customers = frame[frame["is_customer"]]
    brands = frame[~frame["is_customer"]]

    n_turns = frame.groupby("root").size()
    n_customers = customers.groupby("root")["author"].nunique().reindex(n_turns.index, fill_value=0)
    n_brands = brands.groupby("root")["author"].nunique().reindex(n_turns.index, fill_value=0)

    # Total partition with explicit precedence; "clean" is only ever the strict dyad.
    status = pd.Series(
        np.select(
            [
                n_customers == 0,
                n_brands == 0,
                n_customers >= 2,
                n_brands >= 2,
            ],
            ["no_customer", "no_brand", "multi_customer", "multi_brand"],
            default="clean",
        ),
        index=n_turns.index,
        dtype="object",
    )

    # Dominant brand handle, ties broken alphabetically so the choice is deterministic.
    brand_rank = (
        brands.groupby(["root", "author"]).size().reset_index(name="turns")
        .sort_values(["root", "turns", "author"], ascending=[True, False, True])
        .drop_duplicates("root")
        .set_index("root")["author"]
    )

    thread_count = customers.groupby("author")["root"].nunique()
    customer_thread_count = (
        customers.assign(count=customers["author"].map(thread_count))
        .groupby("root")["count"].max()
        .reindex(n_turns.index, fill_value=0)
        .astype("int64")
    )

    return pd.DataFrame(
        {
            "n_turns": n_turns,
            "n_customer_turns": customers.groupby("root").size().reindex(n_turns.index, fill_value=0),
            "n_brand_turns": brands.groupby("root").size().reindex(n_turns.index, fill_value=0),
            "n_customers": n_customers,
            "n_brands": n_brands,
            "status": status,
            "brand": brand_rank.reindex(n_turns.index),
            "customer_thread_count": customer_thread_count,
        }
    )


def assign_turn_index(root_idx: np.ndarray, timestamp: np.ndarray, tweet_id: np.ndarray) -> np.ndarray:
    # created_at orders the conversation; tweet_id only breaks exact ties.
    order = np.lexsort((tweet_id, timestamp, root_idx))
    grouped = root_idx[order]
    starts = np.flatnonzero(np.r_[True, grouped[1:] != grouped[:-1]])
    lengths = np.diff(np.r_[starts, len(order)])
    position = np.arange(len(order)) - np.repeat(starts, lengths)

    turn_index = np.empty(len(order), dtype=np.int32)
    turn_index[order] = position
    return turn_index


def cross_check_response_field(tweet_id, response_tweet_id, parent_idx, resolved) -> dict:
    # response_tweet_id is not used to build edges; it only corroborates them.
    declared = (
        pd.DataFrame({"parent": tweet_id, "child": response_tweet_id.fillna("").str.split(",")})
        .explode("child")
        .assign(child=lambda d: pd.to_numeric(d["child"], errors="coerce"))
        .dropna(subset=["child"])
    )
    declared_edges = set(zip(declared["parent"].astype("int64"), declared["child"].astype("int64")))
    built_edges = set(zip(tweet_id[parent_idx[resolved]], tweet_id[resolved]))
    agreed = declared_edges & built_edges
    return {
        "edges_built_from_in_response_to": len(built_edges),
        "edges_declared_by_response_tweet_id": len(declared_edges),
        "edges_confirmed_by_both": len(agreed),
        "built_edges_confirmed_pct": round(100 * len(agreed) / max(len(built_edges), 1), 2),
        "declared_edges_not_built": len(declared_edges - built_edges),
    }


def validate(turns: pd.DataFrame, conversations: pd.DataFrame, source_tweet_ids: np.ndarray) -> dict:
    checks = {}
    checks["turn_count_matches_source"] = len(turns) == len(source_tweet_ids)
    checks["tweet_id_unique"] = not turns["tweet_id"].duplicated().any()
    checks["all_source_tweets_present"] = set(turns["tweet_id"]) == set(source_tweet_ids.tolist())

    per_tweet_conversations = turns.groupby("tweet_id")["conversation_id"].nunique()
    checks["one_conversation_per_tweet"] = bool((per_tweet_conversations == 1).all())

    conversation_of = dict(zip(turns["tweet_id"], turns["conversation_id"]))
    linked = turns[turns["parent_tweet_id"].notna()]
    parent_conversation = linked["parent_tweet_id"].astype("int64").map(conversation_of)
    checks["parent_in_same_conversation"] = bool((parent_conversation == linked["conversation_id"]).all())
    checks["no_parent_outside_source"] = bool(parent_conversation.notna().all())

    ordered = turns.sort_values(["conversation_id", "turn_index"])
    grouped = ordered.groupby("conversation_id")["turn_index"]
    checks["turn_index_starts_at_zero"] = bool((grouped.min() == 0).all())
    checks["turn_index_has_no_gaps"] = bool((grouped.max() + 1 == grouped.size()).all())

    parent_time = linked["parent_tweet_id"].astype("int64").map(
        dict(zip(turns["tweet_id"], turns["created_at"]))
    )
    checks["parent_never_after_child"] = bool((parent_time <= linked["created_at"]).all())

    clean = conversations[conversations["status"] == "clean"]
    checks["clean_conversations_are_dyads"] = bool(
        ((clean["n_customers"] == 1) & (clean["n_brands"] == 1)).all()
    )
    checks["conversation_ids_unique"] = not conversations["conversation_id"].duplicated().any()
    checks["turn_counts_match_conversation_table"] = (
        int(conversations["n_turns"].sum()) == len(turns)
    )
    return checks


def check_free_disk(target_dir: Path) -> float:
    free_gb = shutil.disk_usage(target_dir).free / 1024**3
    if free_gb < config.MIN_FREE_DISK_GB:
        raise RuntimeError(
            f"Only {free_gb:.2f} GB free; need at least {config.MIN_FREE_DISK_GB} GB "
            "to write the processed conversation tables."
        )
    return free_gb


def write_turns(cache_path: Path, turns_path: Path, columns: dict) -> None:
    writer = pq.ParquetWriter(turns_path, TURNS_SCHEMA, compression="snappy")
    offset = 0
    try:
        for batch in pq.ParquetFile(cache_path).iter_batches(
            batch_size=200_000, columns=["tweet_id", "author_id", "text"]
        ):
            block = batch.to_pandas()
            window = slice(offset, offset + len(block))
            block = pd.DataFrame(
                {
                    "conversation_id": columns["conversation_id"][window],
                    "tweet_id": block["tweet_id"].to_numpy(),
                    "author_id": block["author_id"],
                    "speaker": columns["speaker"][window],
                    "brand": columns["brand"][window],
                    "created_at": columns["created_at"].iloc[window].reset_index(drop=True),
                    "text": block["text"],
                    "parent_tweet_id": columns["parent_tweet_id"][window],
                    "turn_index": columns["turn_index"][window],
                }
            )
            writer.write_table(pa.Table.from_pandas(block, schema=TURNS_SCHEMA, preserve_index=False))
            offset += len(block)
    finally:
        writer.close()


def reconstruct(cache_path: Path):
    frame = pq.read_table(cache_path, columns=GRAPH_COLUMNS).to_pandas()
    tweet_id = frame["tweet_id"].to_numpy()

    parent_idx, declared_parent, resolved = build_parent_index(
        tweet_id, frame["in_response_to_tweet_id"]
    )
    root_idx = find_roots(parent_idx)
    depth = compute_depth(parent_idx)

    is_customer = (frame["inbound"] == "True").to_numpy()
    author_id = frame["author_id"].to_numpy()
    timestamp = pd.to_datetime(
        frame["created_at"], format=config.TWITTER_TIME_FORMAT, errors="coerce", utc=True
    )
    if timestamp.isna().any():
        raise ValueError(f"{int(timestamp.isna().sum())} timestamps failed to parse; cannot order turns")

    summary = classify_participants(root_idx, author_id, is_customer)
    summary["root_tweet_id"] = tweet_id[summary.index.to_numpy()]
    summary["conversation_id"] = "conv_" + summary["root_tweet_id"].astype(str)
    summary["root_truncated"] = (declared_parent & ~resolved)[summary.index.to_numpy()]

    turn_index = assign_turn_index(root_idx, timestamp.astype("int64").to_numpy(), tweet_id)

    timing = pd.DataFrame({"root": root_idx, "ts": timestamp, "depth": depth}).groupby("root")
    summary["started_at"] = timing["ts"].min()
    summary["ended_at"] = timing["ts"].max()
    summary["duration_minutes"] = (
        (summary["ended_at"] - summary["started_at"]).dt.total_seconds() / 60
    ).round(2)
    summary["max_depth"] = timing["depth"].max()

    conversation_id_per_row = summary["conversation_id"].reindex(root_idx).to_numpy()
    brand_per_row = summary["brand"].reindex(root_idx).to_numpy()
    parent_tweet_id = np.where(resolved, tweet_id[parent_idx], np.nan)

    columns = {
        "conversation_id": conversation_id_per_row,
        "speaker": np.where(is_customer, "customer", "brand"),
        "brand": brand_per_row,
        "created_at": timestamp,
        "parent_tweet_id": pd.array(parent_tweet_id, dtype="Int64"),
        "turn_index": turn_index,
    }

    conversations = summary.reset_index(drop=True)[
        [
            "conversation_id", "brand", "status", "root_tweet_id", "root_truncated",
            "n_turns", "n_customer_turns", "n_brand_turns", "n_customers", "n_brands",
            "customer_thread_count", "started_at", "ended_at", "duration_minutes", "max_depth",
        ]
    ]

    cross_check = cross_check_response_field(
        tweet_id, frame["response_tweet_id"], parent_idx, resolved
    )
    graph_facts = {
        "source_rows": len(frame),
        "roots_without_parent_field": int((~declared_parent).sum()),
        "truncated_roots": int((declared_parent & ~resolved).sum()),
        "resolved_parent_edges": int(resolved.sum()),
        "max_depth": int(depth.max()),
    }
    return conversations, columns, cross_check, graph_facts, tweet_id


def sample_conversations(turns: pd.DataFrame, conversations: pd.DataFrame, seed: int) -> dict:
    picked = {}
    rng = np.random.default_rng(seed)
    for status in ["clean", "multi_customer", "multi_brand", "no_customer", "no_brand"]:
        pool = conversations.loc[conversations["status"] == status, "conversation_id"].to_numpy()
        if len(pool) == 0:
            continue
        chosen = pool[rng.integers(0, len(pool))]
        thread = turns[turns["conversation_id"] == chosen].sort_values("turn_index")
        picked[status] = thread.head(12).to_dict(orient="records")
    return picked


def render_report(stats: dict, examples: dict, known_thread, source: str) -> str:
    facts, checks, cross = stats["graph_facts"], stats["validation"], stats["cross_check"]
    status_counts = stats["status_counts"]
    total_conversations = sum(status_counts.values())

    lines = [
        f"# Phase 2 — Conversation reconstruction (`{source}`)",
        "",
        "Generated by `src/reconstruct.py`. No brand is selected here.",
        "",
        "## Objective",
        "",
        "Turn tweet-level reply links into conversation structures that later phases can use "
        "for intent discovery, retrieval, golden-set construction and evaluation.",
        "",
        "## Methodology",
        "",
        "- Edges are built **only** from `in_response_to_tweet_id`.",
        "- `response_tweet_id` is used **only** to cross-check, never to build edges.",
        "- Components are found by pointer doubling over the parent array.",
        "- Turns are ordered by `created_at`; `tweet_id` breaks exact ties.",
        "- A parent absent from the dataset makes its child a **truncated root**.",
        "",
        "## Graph facts",
        "",
        f"- Source rows: **{facts['source_rows']:,}**",
        f"- Roots with no parent field: {facts['roots_without_parent_field']:,}",
        f"- Truncated roots (parent missing from dataset): {facts['truncated_roots']:,}",
        f"- Resolved parent edges: {facts['resolved_parent_edges']:,}",
        f"- Maximum depth: {facts['max_depth']}",
        "",
        "## Cross-check against `response_tweet_id`",
        "",
        "This field is *not* used for reconstruction. It is compared afterwards as independent "
        "corroboration of the edges actually built.",
        "",
        f"- Edges built from `in_response_to_tweet_id`: {cross['edges_built_from_in_response_to']:,}",
        f"- Edges declared by `response_tweet_id`: {cross['edges_declared_by_response_tweet_id']:,}",
        f"- Confirmed by both: {cross['edges_confirmed_by_both']:,} "
        f"(**{cross['built_edges_confirmed_pct']}%** of built edges)",
        f"- Declared but not buildable: {cross['declared_edges_not_built']:,}",
        "",
        "## Conversation classification",
        "",
        "| Status | Conversations | % | Tweets |",
        "| --- | --- | --- | --- |",
    ]
    for status, count in status_counts.items():
        tweets = stats["tweets_by_status"][status]
        lines.append(
            f"| `{status}` | {count:,} | {100 * count / total_conversations:.2f}% | {tweets:,} |"
        )

    lines += [
        "",
        f"Total conversations: **{total_conversations:,}**",
        "",
        "### Why participant structure and not a fan-out threshold",
        "",
        "Fan-out is a proxy. Participant count measures the thing itself: a support interaction "
        "is one customer talking to one brand. The largest components here are rooted in "
        "announcements that drew hundreds of unrelated customers, which a dyad rule excludes "
        "directly.",
        "",
        "Fan-out measured from `response_tweet_id` (Phase 1) and fan-out measured from resolvable "
        "parent edges (Phase 2) are different quantities. Both are reported; neither supersedes "
        "the other. See the Phase 2 section of `docs/DEVELOPMENT_LOG.md`.",
        "",
        "## Clean conversation shape",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for key, value in stats["clean_shape"].items():
        rendered = f"{value:,}" if isinstance(value, int) else str(value)
        lines.append(f"| {key.replace('_', ' ')} | {rendered} |")

    lines += ["", "## Validation", "", "| Invariant | Result |", "| --- | --- |"]
    for name, passed in checks.items():
        lines.append(f"| {name.replace('_', ' ')} | {'PASS' if passed else 'FAIL'} |")

    lines += ["", "## Manual sanity check - the known thread", ""]
    if known_thread is not None and len(known_thread):
        lines += [
            "| turn | tweet_id | speaker | author | parent | created_at |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in known_thread.to_dict(orient="records"):
            parent = "-" if pd.isna(row["parent_tweet_id"]) else int(row["parent_tweet_id"])
            lines.append(
                f"| {row['turn_index']} | {row['tweet_id']} | {row['speaker']} | "
                f"{row['author_id']} | {parent} | {row['created_at']} |"
            )
    else:
        lines.append("_Not present in this source._")

    lines += ["", "## Seeded random examples (one per status)", ""]
    for status, thread in examples.items():
        lines += [f"### `{status}` - `{thread[0]['conversation_id']}`", ""]
        for row in thread:
            text = str(row["text"]).replace("\n", " ")[:110]
            lines.append(f"- **{row['turn_index']}** [{row['speaker']}] `{row['author_id']}`: {text}")
        lines.append("")

    lines += [
        "## Limitations",
        "",
        "- Speaker role is inherited from the dataset's `inbound` field. Some brand accounts are "
        "anonymised as numeric IDs and are therefore counted as customers.",
        "- `customer_thread_count` is recorded without a threshold; downstream phases decide.",
        "- Multi-customer components are kept intact, not split into per-customer threads.",
        "- Truncated roots are missing the customer's opening message.",
        "- Conversations resolved over DM end abruptly; the resolution is not in the data.",
        "",
        "## What Phase 3 depends on",
        "",
        "`data/processed/conversations.parquet` and `data/processed/conversation_turns.parquet`, "
        "joined on `conversation_id`.",
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
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA or value is pd.NaT:
        return None
    raise TypeError(f"Cannot serialise {type(value)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconstruct conversations from the TWCS cache.")
    parser.add_argument("--source", choices=["sample", "full"], required=True)
    args = parser.parse_args()

    if args.source == "sample":
        cache_path = config.INTERIM_DIR / "sample.parquet"
        suffix = "_sample"
    else:
        cache_path = config.INTERIM_DIR / "twcs.parquet"
        suffix = ""

    if not cache_path.exists():
        raise FileNotFoundError(f"Phase 1 cache not found at {cache_path}; run src.profile_raw first")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    free_gb = check_free_disk(config.PROCESSED_DIR)

    print(f"[1/4] Reconstructing from {cache_path.name} ...", flush=True)
    conversations, columns, cross_check, graph_facts, source_tweet_ids = reconstruct(cache_path)
    print(f"      {len(conversations):,} conversations", flush=True)

    print("[2/4] Writing processed tables ...", flush=True)
    turns_path = config.PROCESSED_DIR / f"conversation_turns{suffix}.parquet"
    conversations_path = config.PROCESSED_DIR / f"conversations{suffix}.parquet"
    write_turns(cache_path, turns_path, columns)
    conversations.to_parquet(conversations_path, index=False)

    print("[3/4] Validating ...", flush=True)
    turns = pd.read_parquet(turns_path)
    validation = validate(turns, conversations, source_tweet_ids)
    failed = [name for name, passed in validation.items() if not passed]
    if failed:
        raise RuntimeError(f"Reconstruction validation failed: {', '.join(failed)}")

    print("[4/4] Reporting ...", flush=True)
    status_counts = conversations["status"].value_counts().to_dict()
    tweets_by_status = conversations.groupby("status")["n_turns"].sum().to_dict()
    clean = conversations[conversations["status"] == "clean"]
    clean_shape = {
        "conversations": int(len(clean)),
        "tweets": int(clean["n_turns"].sum()),
        "pct_of_all_tweets": round(100 * clean["n_turns"].sum() / len(turns), 2),
        "median_turns": float(clean["n_turns"].median()),
        "mean_turns": round(float(clean["n_turns"].mean()), 2),
        "max_turns": int(clean["n_turns"].max()),
        "truncated_roots": int(clean["root_truncated"].sum()),
        "median_duration_minutes": float(clean["duration_minutes"].median()),
    }

    known = turns[turns["tweet_id"].isin([8, 6, 5, 4, 3, 1, 2])].sort_values("turn_index")
    examples = sample_conversations(turns, conversations, config.RANDOM_SEED)

    stats = {
        "source": str(cache_path),
        "graph_facts": graph_facts,
        "cross_check": cross_check,
        "status_counts": status_counts,
        "tweets_by_status": {key: int(value) for key, value in tweets_by_status.items()},
        "clean_shape": clean_shape,
        "validation": validation,
        "free_disk_gb_before_write": round(free_gb, 2),
        "turns_parquet_mb": round(turns_path.stat().st_size / 1024**2, 1),
        "conversations_parquet_mb": round(conversations_path.stat().st_size / 1024**2, 1),
    }

    stats_path = config.REPORTS_DIR / f"phase2_reconstruction_stats{suffix}.json"
    report_path = config.REPORTS_DIR / f"phase2_reconstruction{suffix}.md"
    stats_path.write_text(json.dumps(stats, indent=2, default=json_default), encoding="utf-8")
    report_path.write_text(render_report(stats, examples, known, cache_path.name), encoding="utf-8")

    print(f"\nTurns:         {turns_path}")
    print(f"Conversations: {conversations_path}")
    print(f"Report:        {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
