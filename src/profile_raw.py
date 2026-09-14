from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src import config

# Read as text so malformed rows can be counted rather than crash the parser.
RAW_DTYPES = {column: "string" for column in config.RAW_COLUMNS}

PARQUET_SCHEMA = pa.schema(
    [
        ("tweet_id", pa.int64()),
        ("author_id", pa.string()),
        ("inbound", pa.string()),
        ("created_at", pa.string()),
        ("text", pa.string()),
        ("response_tweet_id", pa.string()),
        ("in_response_to_tweet_id", pa.string()),
    ]
)

URL_PATTERN = re.compile(r"https?://")
MENTION_PATTERN = re.compile(r"@\w+")
EMOJI_PATTERN = re.compile("[\U0001f300-\U0001faff☀-➿⬀-⯿]")
AGENT_SIGNATURE_PATTERN = re.compile(r"\^[A-Za-z]{1,4}\s*$")
DM_DEFLECTION_PATTERN = re.compile(r"\b(?:dm|direct message|private message)\b", re.I)
NUMERIC_AUTHOR_PATTERN = re.compile(r"^\d+$")
TOKEN_PATTERN = re.compile(r"[a-z']+")

def count_physical_lines(path: Path) -> int:
    # Compared against parsed rows to detect newlines embedded in tweet text.
    total = 0
    with open(path, "rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            total += block.count(b"\n")
    return total


def check_free_disk(target_dir: Path) -> float:
    free_gb = shutil.disk_usage(target_dir).free / 1024**3
    if free_gb < config.MIN_FREE_DISK_GB:
        raise RuntimeError(
            f"Only {free_gb:.2f} GB free on {target_dir.drive or target_dir}; "
            f"need at least {config.MIN_FREE_DISK_GB} GB to write the Parquet cache."
        )
    return free_gb


def scan_and_cache(csv_path: Path, parquet_path: Path) -> dict:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    free_gb_before = check_free_disk(parquet_path.parent)

    stats = {
        "rows": 0,
        "null_counts": Counter(),
        "inbound_counts": Counter(),
        "malformed_tweet_id": 0,
        "unparsed_timestamps": 0,
        "empty_text": 0,
        "with_url": 0,
        "with_mention": 0,
        "with_emoji": 0,
        "with_agent_signature": 0,
        "malformed_tweet_id_examples": [],
        "unparsed_timestamp_examples": [],
    }
    text_lengths: list[np.ndarray] = []
    earliest = latest = None
    month_counts: Counter = Counter()

    writer = pq.ParquetWriter(parquet_path, PARQUET_SCHEMA, compression="snappy")
    try:
        reader = pd.read_csv(
            csv_path,
            dtype=RAW_DTYPES,
            chunksize=config.CHUNK_SIZE,
            encoding="utf-8",
        )
        for chunk in reader:
            stats["rows"] += len(chunk)
            for column in config.RAW_COLUMNS:
                stats["null_counts"][column] += int(chunk[column].isna().sum())
            stats["inbound_counts"].update(
                chunk["inbound"].fillna("<missing>").value_counts().to_dict()
            )

            tweet_id = pd.to_numeric(chunk["tweet_id"], errors="coerce").astype("Int64")
            bad_id = tweet_id.isna() & chunk["tweet_id"].notna()
            stats["malformed_tweet_id"] += int(bad_id.sum())
            if bad_id.any() and len(stats["malformed_tweet_id_examples"]) < 5:
                stats["malformed_tweet_id_examples"].extend(
                    chunk.loc[bad_id, "tweet_id"].head(5).tolist()
                )

            timestamps = pd.to_datetime(
                chunk["created_at"],
                format=config.TWITTER_TIME_FORMAT,
                errors="coerce",
                utc=True,
            )
            bad_time = timestamps.isna() & chunk["created_at"].notna()
            stats["unparsed_timestamps"] += int(bad_time.sum())
            if bad_time.any() and len(stats["unparsed_timestamp_examples"]) < 5:
                stats["unparsed_timestamp_examples"].extend(
                    chunk.loc[bad_time, "created_at"].head(5).tolist()
                )
            valid_times = timestamps.dropna()
            if not valid_times.empty:
                chunk_min, chunk_max = valid_times.min(), valid_times.max()
                earliest = chunk_min if earliest is None else min(earliest, chunk_min)
                latest = chunk_max if latest is None else max(latest, chunk_max)
                month_counts.update(
                    valid_times.dt.strftime("%Y-%m").value_counts().to_dict()
                )

            text = chunk["text"].fillna("")
            text_lengths.append(text.str.len().to_numpy(dtype=np.int32))
            stats["empty_text"] += int((text.str.strip() == "").sum())
            stats["with_url"] += int(text.str.contains(URL_PATTERN, regex=True).sum())
            stats["with_mention"] += int(
                text.str.contains(MENTION_PATTERN, regex=True).sum()
            )
            stats["with_emoji"] += int(
                text.str.contains(EMOJI_PATTERN, regex=True).sum()
            )
            stats["with_agent_signature"] += int(
                text.str.contains(AGENT_SIGNATURE_PATTERN, regex=True).sum()
            )

            cached = pd.DataFrame(
                {
                    "tweet_id": tweet_id,
                    "author_id": chunk["author_id"],
                    "inbound": chunk["inbound"],
                    "created_at": chunk["created_at"],
                    "text": chunk["text"],
                    "response_tweet_id": chunk["response_tweet_id"],
                    "in_response_to_tweet_id": chunk["in_response_to_tweet_id"],
                }
            )
            writer.write_table(
                pa.Table.from_pandas(cached, schema=PARQUET_SCHEMA, preserve_index=False)
            )
    finally:
        writer.close()

    lengths = np.concatenate(text_lengths)
    stats["text_length"] = {
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "p25": float(np.percentile(lengths, 25)),
        "p75": float(np.percentile(lengths, 75)),
        "p95": float(np.percentile(lengths, 95)),
    }
    stats["earliest_timestamp"] = None if earliest is None else str(earliest)
    stats["latest_timestamp"] = None if latest is None else str(latest)
    stats["tweets_per_month"] = dict(sorted(month_counts.items()))
    stats["null_counts"] = dict(stats["null_counts"])
    stats["inbound_counts"] = dict(stats["inbound_counts"])
    stats["physical_lines"] = count_physical_lines(csv_path)
    stats["free_disk_gb_before_cache"] = round(free_gb_before, 2)
    stats["parquet_size_mb"] = round(parquet_path.stat().st_size / 1024**2, 1)
    return stats


def verify_cache(csv_path: Path, parquet_path: Path, expected_rows: int) -> dict:
    cached_ids = pq.read_table(parquet_path, columns=["tweet_id"])["tweet_id"]
    checks = {
        "row_count_matches": cached_ids.length() == expected_rows,
        "cached_rows": cached_ids.length(),
        "expected_rows": expected_rows,
    }

    head = pd.read_csv(csv_path, dtype=RAW_DTYPES, nrows=1000)
    cached_head = pq.read_table(parquet_path, columns=["tweet_id"]).slice(0, 1000)
    checks["first_1000_ids_match"] = (
        pd.to_numeric(head["tweet_id"]).tolist() == cached_head["tweet_id"].to_pylist()
    )
    return checks


def split_child_ids(series: pd.Series) -> pd.Series:
    # response_tweet_id may hold several comma-separated IDs.
    return series.fillna("").str.split(",")


def analyze_relationships(frame: pd.DataFrame) -> dict:
    stats: dict = {}

    duplicate_mask = frame["tweet_id"].duplicated(keep=False)
    stats["duplicate_tweet_id_rows"] = int(duplicate_mask.sum())
    stats["duplicate_tweet_ids"] = int(frame.loc[duplicate_mask, "tweet_id"].nunique())
    if duplicate_mask.any():
        duplicated_rows = frame[duplicate_mask].sort_values("tweet_id")
        stats["duplicates_are_identical"] = bool(
            duplicated_rows.duplicated(keep=False).all()
        )

    # Graph analysis needs one row per ID; duplicates are counted above, then dropped.
    graph = frame.drop_duplicates(subset="tweet_id").reset_index(drop=True)
    known_ids = pd.Index(graph["tweet_id"])
    stats["unique_tweets_in_graph"] = len(graph)

    parent = pd.to_numeric(graph["in_response_to_tweet_id"], errors="coerce")
    has_parent = parent.notna()
    stats["tweets_with_parent"] = int(has_parent.sum())
    stats["tweets_without_parent"] = int(len(graph) - has_parent.sum())
    dangling_parent = has_parent & ~parent.isin(known_ids)
    stats["dangling_parent_references"] = int(dangling_parent.sum())

    children = split_child_ids(graph["response_tweet_id"])
    child_counts = children.apply(lambda ids: sum(1 for i in ids if i.strip()))
    stats["children_per_tweet"] = {
        "none": int((child_counts == 0).sum()),
        "one": int((child_counts == 1).sum()),
        "two": int((child_counts == 2).sum()),
        "three_or_more": int((child_counts >= 3).sum()),
        "max": int(child_counts.max()),
    }
    stats["branching_rate_pct"] = round(100 * float((child_counts >= 2).mean()), 3)

    exploded = (
        pd.DataFrame({"parent": graph["tweet_id"], "child": children})
        .explode("child")
        .assign(child=lambda d: pd.to_numeric(d["child"], errors="coerce"))
        .dropna(subset=["child"])
    )
    stats["declared_child_edges"] = len(exploded)
    stats["dangling_child_references"] = int((~exploded["child"].isin(known_ids)).sum())

    child_claims = exploded["child"].value_counts()
    stats["tweets_claimed_by_multiple_parents"] = int((child_claims > 1).sum())

    # Both columns describe the same edges; comparing them shows which to trust.
    edges_from_children = set(
        zip(exploded["parent"].astype("int64"), exploded["child"].astype("int64"))
    )
    parent_edges = pd.DataFrame(
        {"parent": parent[has_parent], "child": graph.loc[has_parent, "tweet_id"]}
    )
    edges_from_parents = set(
        zip(parent_edges["parent"].astype("int64"), parent_edges["child"].astype("int64"))
    )
    both = edges_from_children & edges_from_parents
    stats["edge_agreement"] = {
        "edges_from_response_tweet_id": len(edges_from_children),
        "edges_from_in_response_to_tweet_id": len(edges_from_parents),
        "edges_in_both": len(both),
        "only_in_response_tweet_id": len(edges_from_children - edges_from_parents),
        "only_in_in_response_to": len(edges_from_parents - edges_from_children),
        "agreement_pct": round(
            100 * len(both) / max(len(edges_from_children | edges_from_parents), 1), 2
        ),
    }

    # High fan-out marks broadcasts (promotions, outage notices), not support threads.
    fanout = pd.DataFrame(
        {"tweet_id": graph["tweet_id"], "author_id": graph["author_id"], "children": child_counts}
    )
    stats["high_fanout"] = {
        "tweets_with_10_or_more_children": int((child_counts >= 10).sum()),
        "tweets_with_50_or_more_children": int((child_counts >= 50).sum()),
        "replies_under_50plus_fanout": int(child_counts[child_counts >= 50].sum()),
        "top_examples": fanout.nlargest(10, "children").to_dict(orient="records"),
    }

    stats["orphan_tweets"] = int((~has_parent & (child_counts == 0)).sum())
    stats["cycles"] = detect_cycles(graph["tweet_id"].to_numpy(), parent.to_numpy())
    stats.update(analyze_chronology(graph, parent, has_parent))
    return stats


def detect_cycles(tweet_ids: np.ndarray, parent_ids: np.ndarray) -> dict:
    # At most one parent per tweet makes this a functional graph, so one linear
    # scan with three-state marking suffices. Iterative: recursion would overflow.
    position = pd.Series(np.arange(len(tweet_ids)), index=tweet_ids)
    parent_index = position.reindex(parent_ids).to_numpy(dtype="float64")
    parent_index = np.where(np.isnan(parent_index), -1, parent_index).astype(np.int64)

    UNVISITED, IN_PROGRESS, DONE = 0, 1, 2
    state = np.zeros(len(tweet_ids), dtype=np.int8)
    cycle_entry_points: list[int] = []

    for start in range(len(tweet_ids)):
        if state[start] != UNVISITED:
            continue
        path = []
        node = start
        while node != -1 and state[node] == UNVISITED:
            state[node] = IN_PROGRESS
            path.append(node)
            node = parent_index[node]
        if node != -1 and state[node] == IN_PROGRESS:
            cycle_entry_points.append(int(tweet_ids[node]))
        for visited in path:
            state[visited] = DONE

    return {
        "count": len(cycle_entry_points),
        "examples": cycle_entry_points[:10],
    }


def analyze_chronology(
    graph: pd.DataFrame, parent: pd.Series, has_parent: pd.Series
) -> dict:
    timestamps = pd.to_datetime(
        graph["created_at"], format=config.TWITTER_TIME_FORMAT, errors="coerce", utc=True
    )

    valid = timestamps.notna()
    sample = graph.loc[valid].sample(
        n=min(200_000, int(valid.sum())), random_state=config.RANDOM_SEED
    )
    sample_times = timestamps.loc[sample.index]
    # Spearman = Pearson on ranks; computing it this way avoids a scipy dependency.
    spearman = (
        sample["tweet_id"].rank().corr(sample_times.rank())
        if len(sample) > 1
        else float("nan")
    )

    time_by_id = pd.Series(timestamps.to_numpy(), index=graph["tweet_id"])
    edges = pd.DataFrame(
        {"parent": parent[has_parent], "child": graph.loc[has_parent, "tweet_id"]}
    )
    edges["parent_time"] = edges["parent"].map(time_by_id)
    edges["child_time"] = edges["child"].map(time_by_id)
    comparable = edges.dropna(subset=["parent_time", "child_time"])

    # Ties are reported separately because a rounded 100% would hide real violations.
    earlier = int((comparable["parent_time"] < comparable["child_time"]).sum())
    simultaneous = int((comparable["parent_time"] == comparable["child_time"]).sum())
    later = int((comparable["parent_time"] > comparable["child_time"]).sum())

    return {
        "chronology": {
            "spearman_tweet_id_vs_time": round(float(spearman), 4),
            "comparable_edges": len(comparable),
            "parent_earlier_in_time": earlier,
            "parent_same_second": simultaneous,
            "parent_later_in_time": later,
            "pct_parent_earlier_in_time": round(100 * earlier / max(len(comparable), 1), 6),
            "pct_parent_lower_id": round(
                100 * float((comparable["parent"] < comparable["child"]).mean()), 2
            ),
        }
    }


def analyze_authors(frame: pd.DataFrame) -> dict:
    is_numeric = frame["author_id"].str.match(NUMERIC_AUTHOR_PATTERN).fillna(False)
    crosstab = pd.crosstab(is_numeric, frame["inbound"].fillna("<missing>"))

    per_author_inbound = frame.groupby("author_id", observed=True)["inbound"].nunique()
    mixed_authors = per_author_inbound[per_author_inbound > 1]

    outbound_by_author = (
        frame[frame["inbound"] == "False"]
        .groupby("author_id", observed=True)
        .size()
        .sort_values(ascending=False)
    )

    return {
        "unique_authors": int(frame["author_id"].nunique()),
        "numeric_author_ids": int(is_numeric.sum()),
        "non_numeric_author_ids": int((~is_numeric).sum()),
        "numeric_vs_inbound_crosstab": {
            str(index): row.to_dict() for index, row in crosstab.iterrows()
        },
        "authors_with_both_directions": int(len(mixed_authors)),
        "authors_with_both_directions_examples": mixed_authors.index[:10].tolist(),
        "support_account_candidates": int(len(outbound_by_author)),
        "outbound_by_author": outbound_by_author,
    }


def compare_brands(frame: pd.DataFrame, outbound_by_author: pd.Series) -> pd.DataFrame:
    # Built from 1-hop links only, so conversation columns are approximations.
    brands = outbound_by_author.head(config.TOP_BRAND_COUNT).index.tolist()
    author_by_tweet = pd.Series(
        frame["author_id"].to_numpy(), index=frame["tweet_id"]
    )
    author_by_tweet = author_by_tweet[~author_by_tweet.index.duplicated()]

    parent = pd.to_numeric(frame["in_response_to_tweet_id"], errors="coerce")
    edges = pd.DataFrame(
        {"parent": parent, "child": frame["tweet_id"], "child_author": frame["author_id"]}
    ).dropna(subset=["parent"])
    edges["parent_author"] = edges["parent"].map(author_by_tweet)

    rows = []
    for brand in brands:
        brand_tweets = frame[frame["author_id"] == brand]
        brand_ids = set(brand_tweets["tweet_id"])

        replies_to_brand = edges[edges["parent_author"] == brand]
        replies_by_brand = edges[edges["child_author"] == brand]
        customers = set(replies_to_brand["child_author"]) | set(
            replies_by_brand["parent"].map(author_by_tweet).dropna()
        )
        customers.discard(brand)

        linked = brand_tweets["tweet_id"].isin(
            set(edges["parent"]) | set(edges["child"])
        )
        has_parent = brand_tweets["in_response_to_tweet_id"].notna()

        rows.append(
            {
                "brand": brand,
                "total_tweets": len(brand_tweets),
                "outbound_tweets": int((brand_tweets["inbound"] == "False").sum()),
                "inbound_tweets": int((brand_tweets["inbound"] == "True").sum()),
                "customer_replies_received": len(replies_to_brand),
                "brand_replies_sent": len(replies_by_brand),
                "unique_customers": len(customers),
                "conversation_candidates": int(
                    replies_by_brand["parent"].nunique()
                ),
                "avg_brand_tweets_per_customer": round(
                    len(brand_tweets) / max(len(customers), 1), 2
                ),
                "pct_tweets_linked": round(100 * float(linked.mean()), 1),
                "pct_tweets_with_parent": round(100 * float(has_parent.mean()), 1),
                "brand_ids_count": len(brand_ids),
            }
        )

    return pd.DataFrame(rows).sort_values("total_tweets", ascending=False)


def analyze_brand_text(parquet_path: Path, brands: list[str]) -> dict:
    # Text is the largest column, so it is streamed and only shortlisted brands kept.
    wanted = set(brands)
    brand_texts: dict[str, list[str]] = {brand: [] for brand in brands}
    brand_outbound_stats: dict[str, Counter] = {brand: Counter() for brand in brands}

    parquet_file = pq.ParquetFile(parquet_path)
    for batch in parquet_file.iter_batches(
        batch_size=200_000, columns=["author_id", "inbound", "text"]
    ):
        block = batch.to_pandas()
        block = block[block["author_id"].isin(wanted)]
        if block.empty:
            continue
        for brand, group in block.groupby("author_id", observed=True):
            text = group["text"].fillna("")
            counters = brand_outbound_stats[brand]
            counters["tweets"] += len(group)
            counters["total_length"] += int(text.str.len().sum())
            counters["with_url"] += int(text.str.contains(URL_PATTERN, regex=True).sum())
            counters["dm_deflection"] += int(
                text.str.contains(DM_DEFLECTION_PATTERN, regex=True).sum()
            )
            counters["with_signature"] += int(
                text.str.contains(AGENT_SIGNATURE_PATTERN, regex=True).sum()
            )
            room = config.BRAND_TEXT_SAMPLE - len(brand_texts[brand])
            if room > 0:
                brand_texts[brand].extend(text.head(room).tolist())

    results = {}
    for brand in brands:
        counters = brand_outbound_stats[brand]
        tweets = max(counters["tweets"], 1)
        tokens = []
        for text in brand_texts[brand]:
            cleaned = MENTION_PATTERN.sub(" ", URL_PATTERN.sub(" ", text.lower()))
            tokens.extend(TOKEN_PATTERN.findall(cleaned))
        results[brand] = {
            "tweets_seen": counters["tweets"],
            "avg_text_length": round(counters["total_length"] / tweets, 1),
            "pct_with_url": round(100 * counters["with_url"] / tweets, 1),
            "pct_dm_deflection": round(100 * counters["dm_deflection"] / tweets, 1),
            "pct_with_agent_signature": round(100 * counters["with_signature"] / tweets, 1),
            "type_token_ratio": round(len(set(tokens)) / max(len(tokens), 1), 4),
            "tokens_sampled": len(tokens),
            "top_terms": [word for word, _ in Counter(tokens).most_common(15)],
        }
    return results


def markdown_table(frame: pd.DataFrame) -> str:
    # pandas' to_markdown() would pull in tabulate, which nothing else needs.
    header = "| " + " | ".join(frame.columns) + " |"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(f"{value:,}" if isinstance(value, (int, np.integer)) else str(value)
                          for value in record) + " |"
        for record in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Series):
        return value.to_dict()
    if value is pd.NA or value is pd.NaT:
        return None
    raise TypeError(f"Cannot serialise {type(value)}")


def render_report(stats: dict, brand_table: pd.DataFrame, source: str) -> str:
    basic = stats["basic"]
    rel = stats["relationships"]
    authors = stats["authors"]
    chrono = rel["chronology"]

    lines = [
        f"# Phase 1 — Raw dataset profile (`{source}`)",
        "",
        "Generated by `src/profile_raw.py`. Conversations are **not** reconstructed "
        "here and **no brand is selected**.",
        "",
        "## 1. General",
        "",
        f"- Rows parsed: **{basic['rows']:,}**",
        f"- Physical lines in file: {basic['physical_lines']:,} "
        f"(difference of {basic['physical_lines'] - basic['rows'] - 1:,} indicates "
        "newlines embedded inside tweet text)",
        f"- Malformed `tweet_id`: {basic['malformed_tweet_id']:,}",
        f"- Unparsed timestamps: {basic['unparsed_timestamps']:,}",
        "",
        "### Missing values per column",
        "",
        "| Column | Nulls | % |",
        "| --- | --- | --- |",
    ]
    for column, count in basic["null_counts"].items():
        lines.append(f"| `{column}` | {count:,} | {100 * count / basic['rows']:.2f}% |")

    lines += [
        "",
        "## 2. Direction (`inbound`)",
        "",
        "| Value | Count | % |",
        "| --- | --- | --- |",
    ]
    for value, count in sorted(basic["inbound_counts"].items()):
        lines.append(f"| `{value}` | {count:,} | {100 * count / basic['rows']:.2f}% |")

    lines += [
        "",
        "## 3. Time",
        "",
        f"- Earliest: `{basic['earliest_timestamp']}`",
        f"- Latest: `{basic['latest_timestamp']}`",
        "",
        "| Month | Tweets |",
        "| --- | --- |",
    ]
    for month, count in basic["tweets_per_month"].items():
        lines.append(f"| {month} | {count:,} |")

    length = basic["text_length"]
    lines += [
        "",
        "## 4. Text",
        "",
        f"- Mean length: {length['mean']:.1f} characters",
        f"- Median length: {length['median']:.0f}",
        f"- Range: {length['min']}–{length['max']}",
        f"- p25 / p75 / p95: {length['p25']:.0f} / {length['p75']:.0f} / {length['p95']:.0f}",
        f"- Empty or whitespace-only: {basic['empty_text']:,}",
        f"- Containing a URL: {basic['with_url']:,} "
        f"({100 * basic['with_url'] / basic['rows']:.1f}%)",
        f"- Containing an @mention: {basic['with_mention']:,} "
        f"({100 * basic['with_mention'] / basic['rows']:.1f}%)",
        f"- Containing emoji: {basic['with_emoji']:,} "
        f"({100 * basic['with_emoji'] / basic['rows']:.1f}%)",
        f"- Ending with an agent signature (`^XX`): {basic['with_agent_signature']:,} "
        f"({100 * basic['with_agent_signature'] / basic['rows']:.1f}%)",
        "",
        "## 5. Relationships",
        "",
        f"- Tweets with a parent: {rel['tweets_with_parent']:,}",
        f"- Tweets without a parent: {rel['tweets_without_parent']:,}",
        f"- Dangling parent references: {rel['dangling_parent_references']:,}",
        f"- Declared child edges: {rel['declared_child_edges']:,}",
        f"- Dangling child references: {rel['dangling_child_references']:,}",
        f"- Duplicate `tweet_id` rows: {rel['duplicate_tweet_id_rows']:,}",
        f"- Tweets claimed as a child by more than one parent: "
        f"{rel['tweets_claimed_by_multiple_parents']:,}",
        f"- Orphans (no parent, no children): {rel['orphan_tweets']:,}",
        f"- Cycles detected: {rel['cycles']['count']:,}",
        "",
        "### Branching",
        "",
        "| Children | Tweets |",
        "| --- | --- |",
        f"| 0 | {rel['children_per_tweet']['none']:,} |",
        f"| 1 | {rel['children_per_tweet']['one']:,} |",
        f"| 2 | {rel['children_per_tweet']['two']:,} |",
        f"| 3+ | {rel['children_per_tweet']['three_or_more']:,} |",
        "",
        f"Branching rate (2+ children): **{rel['branching_rate_pct']}%**, "
        f"max children on one tweet: {rel['children_per_tweet']['max']}.",
        "",
        "### Broadcast tweets (high fan-out)",
        "",
        "Tweets with very many direct replies are promotions or outage notices, not "
        "support threads. Phase 2 must not treat their reply storms as conversations.",
        "",
        f"- Tweets with 10+ replies: {rel['high_fanout']['tweets_with_10_or_more_children']:,}",
        f"- Tweets with 50+ replies: {rel['high_fanout']['tweets_with_50_or_more_children']:,}",
        f"- Replies sitting under those 50+ fan-out tweets: "
        f"{rel['high_fanout']['replies_under_50plus_fanout']:,}",
        "",
        "| tweet_id | author | replies |",
        "| --- | --- | --- |",
        *[
            f"| {row['tweet_id']} | {row['author_id']} | {row['children']:,} |"
            for row in rel["high_fanout"]["top_examples"]
        ],
        "",
        "### Do the two relationship columns agree?",
        "",
    ]
    agreement = rel["edge_agreement"]
    for key, value in agreement.items():
        lines.append(f"- {key.replace('_', ' ')}: {value:,}" if isinstance(value, int)
                     else f"- {key.replace('_', ' ')}: {value}")

    lines += [
        "",
        "### Is `tweet_id` chronological?",
        "",
        f"- Spearman correlation between `tweet_id` and timestamp: "
        f"**{chrono['spearman_tweet_id_vs_time']}**",
        f"- Comparable parent→child edges: {chrono['comparable_edges']:,}",
        f"- Parent **earlier in time**: {chrono['parent_earlier_in_time']:,} "
        f"(**{chrono['pct_parent_earlier_in_time']}%**)",
        f"- Parent at the **same second** (tie): {chrono['parent_same_second']:,}",
        f"- Parent **later in time** (violations): "
        f"**{chrono['parent_later_in_time']:,}**",
        f"- Edges where the parent has a **lower tweet_id**: "
        f"**{chrono['pct_parent_lower_id']}%**",
        "",
        "## 6. Authors",
        "",
        f"- Unique authors: {authors['unique_authors']:,}",
        f"- Numeric-looking `author_id`: {authors['numeric_author_ids']:,}",
        f"- Non-numeric `author_id`: {authors['non_numeric_author_ids']:,}",
        f"- Authors appearing with **both** inbound directions: "
        f"{authors['authors_with_both_directions']:,}",
        f"- Support-account candidates (any outbound tweet): "
        f"{authors['support_account_candidates']:,}",
        "",
        "### Does numeric `author_id` really mean 'customer'?",
        "",
        "| author_id is numeric | " + " | ".join(
            sorted(next(iter(authors["numeric_vs_inbound_crosstab"].values())).keys())
        ) + " |",
        "| --- " * (1 + len(next(iter(authors["numeric_vs_inbound_crosstab"].values()))))
        + "|",
    ]
    for is_numeric, counts in authors["numeric_vs_inbound_crosstab"].items():
        ordered = [f"{counts[key]:,}" for key in sorted(counts)]
        lines.append(f"| {is_numeric} | " + " | ".join(ordered) + " |")

    lines += [
        "",
        "## 7. Brand comparison",
        "",
        "> Conversation-shaped columns are **approximations from 1-hop links only**. "
        "Real conversations do not exist until Phase 3. Use this to shortlist, "
        "not to decide.",
        "",
        markdown_table(brand_table),
        "",
        "## 8. Per-brand text characteristics",
        "",
        "| Brand | Avg len | % URL | % DM deflection | % signed | Type-token ratio |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for brand, values in stats["brand_text"].items():
        lines.append(
            f"| {brand} | {values['avg_text_length']} | {values['pct_with_url']} "
            f"| {values['pct_dm_deflection']} | {values['pct_with_agent_signature']} "
            f"| {values['type_token_ratio']} |"
        )

    lines += ["", "### Most frequent terms per brand", ""]
    for brand, values in stats["brand_text"].items():
        lines.append(f"- **{brand}**: {', '.join(values['top_terms'])}")

    lines += [
        "",
        "## 9. Cache and integrity",
        "",
        f"- Parquet cache: {basic['parquet_size_mb']} MB",
        f"- Free disk before write: {basic['free_disk_gb_before_cache']} GB",
        f"- Cache row count matches CSV: {stats['cache_checks']['row_count_matches']}",
        f"- First 1000 tweet IDs match: {stats['cache_checks']['first_1000_ids_match']}",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile the raw TWCS dataset.")
    parser.add_argument("--source", choices=["sample", "full"], required=True)
    args = parser.parse_args()

    if args.source == "sample":
        csv_path = config.RAW_SAMPLE_CSV
        parquet_path = config.INTERIM_DIR / "sample.parquet"
        report_path = config.REPORTS_DIR / "phase1_profile_sample.md"
        stats_path = config.REPORTS_DIR / "phase1_stats_sample.json"
    else:
        csv_path = config.RAW_TWCS_CSV
        parquet_path = config.INTERIM_DIR / "twcs.parquet"
        report_path = config.REPORTS_DIR / "phase1_profile.md"
        stats_path = config.REPORTS_DIR / "phase1_stats.json"

    if not csv_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {csv_path}")

    print(f"[1/4] Scanning {csv_path.name} and writing cache ...", flush=True)
    basic = scan_and_cache(csv_path, parquet_path)
    cache_checks = verify_cache(csv_path, parquet_path, basic["rows"])
    if not cache_checks["row_count_matches"]:
        raise RuntimeError(
            f"Parquet cache has {cache_checks['cached_rows']:,} rows but the CSV "
            f"parsed {cache_checks['expected_rows']:,}."
        )
    print(f"      {basic['rows']:,} rows, cache {basic['parquet_size_mb']} MB", flush=True)

    print("[2/4] Analysing relationships ...", flush=True)
    graph_columns = [
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "response_tweet_id",
        "in_response_to_tweet_id",
    ]
    frame = pq.read_table(parquet_path, columns=graph_columns).to_pandas()
    frame = frame.dropna(subset=["tweet_id"])
    relationships = analyze_relationships(frame)

    print("[3/4] Analysing authors and brands ...", flush=True)
    authors = analyze_authors(frame)
    outbound_by_author = authors.pop("outbound_by_author")
    brand_table = compare_brands(frame, outbound_by_author)

    print("[4/4] Analysing per-brand text ...", flush=True)
    brand_text = analyze_brand_text(parquet_path, brand_table["brand"].tolist())

    stats = {
        "source": str(csv_path),
        "basic": basic,
        "relationships": relationships,
        "authors": authors,
        "brand_table": brand_table.to_dict(orient="records"),
        "brand_text": brand_text,
        "cache_checks": cache_checks,
    }

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2, default=json_default), encoding="utf-8")
    report_path.write_text(render_report(stats, brand_table, csv_path.name), encoding="utf-8")

    print(f"\nReport: {report_path}")
    print(f"Stats:  {stats_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
