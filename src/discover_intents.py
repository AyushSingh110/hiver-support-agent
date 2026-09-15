from __future__ import annotations

import os

# Must precede the sklearn import: avoids the MKL KMeans leak on Windows and keeps
# float summation order stable across runs.
os.environ.setdefault("OMP_NUM_THREADS", "2")

import argparse
import html
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from src import config

BRAND = "AmericanAir"
K_VALUES = [6, 8, 10, 12, 15, 20]
SVD_COMPONENTS = 100
SAMPLE_SIZE = 500
SILHOUETTE_SAMPLE = 5_000

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
WHITESPACE_PATTERN = re.compile(r"\s+")
SIGNATURE_PATTERN = re.compile(r"\^[A-Za-z]{1,4}\s*$")
CAPS_PATTERN = re.compile(r"[A-Z]")

# The brand's own name appears in nearly every message and cannot discriminate.
BRAND_STOP_WORDS = {"americanair", "american", "aa", "americanairlines"}
STOP_WORDS = list(ENGLISH_STOP_WORDS | BRAND_STOP_WORDS)

TFIDF_SETTINGS = {
    "ngram_range": (1, 2),
    "min_df": 5,
    "max_df": 0.5,
    "sublinear_tf": True,
    "max_features": 50_000,
}


def normalise_message(text: str, keep_mentions: bool = False) -> str:
    text = html.unescape(str(text))
    text = URL_PATTERN.sub(" ", text)
    if not keep_mentions:
        text = MENTION_PATTERN.sub(" ", text)
    text = HASHTAG_PATTERN.sub(r"\1", text)
    return WHITESPACE_PATTERN.sub(" ", text).strip().lower()


def load_openings(sample: bool) -> pd.DataFrame:
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
        columns=["conversation_id", "brand", "speaker", "text", "turn_index", "created_at"],
    ):
        block = batch.to_pandas()
        block = block[
            (block["brand"] == BRAND)
            & (block["turn_index"] == 0)
            & (block["speaker"] == "customer")
        ]
        block = block[block["conversation_id"].isin(clean_ids)]
        if not block.empty:
            parts.append(block[["conversation_id", "text", "created_at"]])

    openings = pd.concat(parts, ignore_index=True)
    if sample:
        openings = openings.head(SAMPLE_SIZE)
    return openings


def build_features(messages: list[str], keep_mentions: bool = False, use_bigrams: bool = True):
    settings = dict(TFIDF_SETTINGS)
    if not use_bigrams:
        settings["ngram_range"] = (1, 1)
    vectoriser = TfidfVectorizer(stop_words=STOP_WORDS, **settings)
    matrix = vectoriser.fit_transform(messages)
    if matrix.shape[1] == 0:
        raise ValueError("TF-IDF produced an empty vocabulary; check preprocessing and min_df")
    return vectoriser, matrix


def reduce_dimensions(matrix, n_components: int):
    components = min(n_components, matrix.shape[1] - 1, matrix.shape[0] - 1)
    svd = TruncatedSVD(n_components=components, random_state=config.RANDOM_SEED)
    reduced = svd.fit_transform(matrix)
    return svd, reduced


def sweep_k(reduced: np.ndarray, k_values: list[int]) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(config.RANDOM_SEED)
    sample_idx = (
        rng.choice(len(reduced), SILHOUETTE_SAMPLE, replace=False)
        if len(reduced) > SILHOUETTE_SAMPLE
        else np.arange(len(reduced))
    )
    for k in k_values:
        if k >= len(reduced):
            continue
        model = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_SEED).fit(reduced)
        labels = model.labels_
        sizes = np.bincount(labels, minlength=k)
        centroid_similarity = cosine_similarity(model.cluster_centers_)
        np.fill_diagonal(centroid_similarity, 0.0)
        rows.append(
            {
                "k": k,
                "inertia": round(float(model.inertia_), 2),
                "silhouette": round(float(silhouette_score(reduced[sample_idx], labels[sample_idx])), 4),
                "largest_cluster_pct": round(100 * float(sizes.max() / len(labels)), 1),
                "smallest_cluster_pct": round(100 * float(sizes.min() / len(labels)), 2),
                "clusters_under_1pct": int((sizes / len(labels) < 0.01).sum()),
                "max_centroid_similarity": round(float(centroid_similarity.max()), 3),
            }
        )
    return pd.DataFrame(rows)


def distinctive_terms(matrix, labels: np.ndarray, cluster: int, vocabulary: np.ndarray, top: int = 15):
    # Ranked by lift, not raw weight: raw weight surfaces common words in every cluster.
    inside = labels == cluster
    inside_mean = np.asarray(matrix[inside].mean(axis=0)).ravel()
    outside_mean = np.asarray(matrix[~inside].mean(axis=0)).ravel()
    lift = inside_mean - outside_mean
    return [
        {"term": vocabulary[i], "lift": round(float(lift[i]), 4), "weight": round(float(inside_mean[i]), 4)}
        for i in np.argsort(-lift)[:top]
    ]


def describe_cluster(cluster, labels, matrix, reduced, centres, vocabulary, messages, raw, seed):
    members = np.flatnonzero(labels == cluster)
    distances = np.linalg.norm(reduced[members] - centres[cluster], axis=1)
    nearest = members[np.argsort(distances)[:8]]

    rng = np.random.default_rng(seed + cluster)
    pool = np.setdiff1d(members, nearest)
    random_members = rng.choice(pool, min(3, len(pool)), replace=False) if len(pool) else np.array([], dtype=int)

    normalised = [messages[i] for i in members]
    return {
        "cluster": int(cluster),
        "size": int(len(members)),
        "pct_of_corpus": round(100 * len(members) / len(labels), 2),
        "cohesion": round(float(np.mean(1 / (1 + distances))), 4),
        "mean_chars": round(float(np.mean([len(text) for text in normalised])), 1),
        "top_terms": distinctive_terms(matrix, labels, cluster, vocabulary),
        "representative_messages": [str(raw[i])[:220] for i in nearest],
        "random_messages": [str(raw[i])[:220] for i in random_members],
    }


def quality_checks(clusters, labels, raw, corpus_top_terms) -> dict:
    sizes = np.array([c["size"] for c in clusters])
    url_flag = np.array([bool(URL_PATTERN.search(str(text))) for text in raw])
    caps_ratio = np.array([
        len(CAPS_PATTERN.findall(str(text))) / max(len(str(text)), 1) for text in raw
    ])
    lengths = np.array([len(str(text)) for text in raw])

    generic = []
    for cluster in clusters:
        terms = {entry["term"] for entry in cluster["top_terms"][:8]}
        if len(terms & corpus_top_terms) >= 5:
            generic.append(cluster["cluster"])

    url_driven, format_driven = [], []
    for cluster in clusters:
        members = labels == cluster["cluster"]
        if url_flag[members].mean() > 0.6 and url_flag.mean() < 0.3:
            url_driven.append(cluster["cluster"])
        if abs(lengths[members].mean() - lengths.mean()) > 2 * lengths.std():
            format_driven.append(cluster["cluster"])

    return {
        "generic_clusters": generic,
        "url_driven_clusters": url_driven,
        "format_driven_clusters": format_driven,
        "clusters_under_1pct": [c["cluster"] for c in clusters if c["pct_of_corpus"] < 1.0],
        "clusters_over_25pct": [c["cluster"] for c in clusters if c["pct_of_corpus"] > 25.0],
        "signature_matches_in_openings": int(
            sum(bool(SIGNATURE_PATTERN.search(str(text))) for text in raw)
        ),
        "corpus_url_rate_pct": round(100 * float(url_flag.mean()), 1),
        "mean_caps_ratio": round(float(caps_ratio.mean()), 3),
        "size_ratio_largest_to_smallest": round(float(sizes.max() / max(sizes.min(), 1)), 1),
    }


def normalisation_sensitivity(raw_messages, labels, chosen_k: int) -> dict:
    variant = [normalise_message(text, keep_mentions=True) for text in raw_messages]
    _, matrix = build_features(variant, keep_mentions=True, use_bigrams=False)
    _, reduced = reduce_dimensions(matrix, SVD_COMPONENTS)
    variant_labels = KMeans(
        n_clusters=chosen_k, n_init=10, random_state=config.RANDOM_SEED
    ).fit_predict(reduced)
    return {
        "variant": "mentions kept, unigrams only",
        "adjusted_rand_index": round(float(adjusted_rand_score(labels, variant_labels)), 4),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    def render(value):
        if isinstance(value, (bool, np.bool_)):
            return "yes" if value else "no"
        if isinstance(value, (int, np.integer)):
            return f"{value:,}"
        return str(value)

    header = "| " + " | ".join(frame.columns) + " |"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(render(value) for value in record) + " |"
        for record in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def render_report(stats: dict, sweep: pd.DataFrame, clusters: list[dict], source: str) -> str:
    features = stats["features"]
    checks = stats["quality_checks"]

    lines = [
        f"# Phase 4 — Intent discovery for {BRAND} (`{source}`)",
        "",
        "Generated by `src/discover_intents.py`.",
        "",
        "> **The clusters below are discovery artifacts, not intents.** The proposed taxonomy "
        "at the end is a **human interpretation** of this evidence and is **UNVALIDATED** — "
        "no labelled data exists yet.",
        "",
        "## Objective",
        "",
        "Find the natural issue structure in AmericanAir customer opening messages so a small, "
        "human-authored intent taxonomy can be written from evidence rather than intuition.",
        "",
        "## Data scope",
        "",
        f"- Opening messages analysed: **{stats['messages']:,}**",
        "- Definition: first turn of a `clean`, customer-rooted AmericanAir conversation",
        f"- Empty after normalisation: {stats['empty_after_normalisation']:,}",
        f"- Date range: {stats['earliest']} to {stats['latest']}",
        "",
        "## Preprocessing",
        "",
        "HTML unescape, lowercase, URLs removed, `@mentions` stripped, `#` removed but the word "
        "kept, whitespace collapsed. No stemming. English stopwords plus "
        f"`{sorted(BRAND_STOP_WORDS)}`. `flight` was deliberately **not** hand-removed; `max_df` "
        "decides whether it survives.",
        "",
        "## TF-IDF",
        "",
        f"- Settings: `{TFIDF_SETTINGS}`",
        f"- Matrix shape: **{features['n_documents']:,} x {features['vocabulary_size']:,}**",
        f"- Sparsity: {features['sparsity_pct']}% zero entries",
        f"- Mean non-zero terms per message: {features['mean_terms_per_document']}",
        f"- `flight` retained in vocabulary: **{features['flight_retained']}**",
        "",
        "Highest document-frequency terms retained: "
        + ", ".join(f"`{term}`" for term in features["top_document_frequency_terms"]),
        "",
        "## SVD",
        "",
        f"- Components: {features['svd_components']}",
        f"- Cumulative explained variance: **{features['explained_variance_pct']}%**",
        "",
        "## K sweep",
        "",
        markdown_table(sweep),
        "",
        f"**Chosen K = {stats['chosen_k']}.** {stats['k_rationale']}",
        "",
        "## Clusters",
        "",
    ]

    for cluster in clusters:
        terms = ", ".join(f"`{entry['term']}`" for entry in cluster["top_terms"][:12])
        lines += [
            f"### Cluster {cluster['cluster']} — {cluster['size']:,} messages "
            f"({cluster['pct_of_corpus']}%)",
            "",
            f"**Distinctive terms:** {terms}",
            "",
            f"**Cohesion:** {cluster['cohesion']} · **Mean length:** {cluster['mean_chars']} chars"
            + (f" · **Nearest cluster:** {cluster['nearest_cluster']} "
               f"(similarity {cluster['nearest_similarity']})" if "nearest_cluster" in cluster else ""),
            "",
            "**Representative messages (nearest to centroid):**",
            "",
        ]
        lines += [f"- {text}" for text in cluster["representative_messages"]]
        if cluster["random_messages"]:
            lines += ["", "**Random members (seeded, not hand-picked):**", ""]
            lines += [f"- {text}" for text in cluster["random_messages"]]
        lines.append("")

    lines += [
        "## Quality checks",
        "",
        "| Check | Result |",
        "| --- | --- |",
        f"| Generic clusters (top terms overlap corpus-wide terms) | {checks['generic_clusters'] or 'none'} |",
        f"| URL-driven clusters | {checks['url_driven_clusters'] or 'none'} |",
        f"| Format/length-driven clusters | {checks['format_driven_clusters'] or 'none'} |",
        f"| Clusters under 1% of corpus | {checks['clusters_under_1pct'] or 'none'} |",
        f"| Clusters over 25% of corpus | {checks['clusters_over_25pct'] or 'none'} |",
        f"| Agent signatures found in customer openings | {checks['signature_matches_in_openings']:,} |",
        f"| Corpus URL rate | {checks['corpus_url_rate_pct']}% |",
        f"| Largest:smallest cluster ratio | {checks['size_ratio_largest_to_smallest']} |",
        "",
        "## Normalisation sensitivity",
        "",
        f"Re-clustered with a different preprocessing variant ({stats['sensitivity']['variant']}) "
        f"at the same K. Adjusted Rand Index against the main partition: "
        f"**{stats['sensitivity']['adjusted_rand_index']}**.",
        "",
        "ARI of 1.0 means identical partitions and 0.0 means no better than chance. A low value "
        "means the cluster structure depends materially on preprocessing choices, which is a "
        "limitation on how much any single partition should be trusted.",
        "",
        "## Limitations",
        "",
        "- TF-IDF is **lexical, not semantic**: it cannot see that `bag` and `luggage` mean the "
        "same thing, and it will split paraphrases of one issue across clusters.",
        "- KMeans assigns **every** message to a cluster, including genuine noise, and prefers "
        "roughly spherical, similar-sized groups. Real intent distributions are neither.",
        "- Silhouette scores on sparse text are weak and are reported for comparison across K "
        "only — they are not evidence that clusters are meaningful.",
        "- No ground truth exists, so nothing here is validated.",
        "- Opening messages only; issues revealed later in a conversation are invisible.",
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
    parser = argparse.ArgumentParser(description="Discover intent structure in AmericanAir openings.")
    parser.add_argument("--source", choices=["sample", "full"], required=True)
    parser.add_argument("--k", type=int, help="Override the chosen K for cluster description")
    args = parser.parse_args()

    suffix = "_sample" if args.source == "sample" else ""
    print(f"[1/5] Loading {BRAND} opening messages ...", flush=True)
    openings = load_openings(sample=args.source == "sample")
    raw_messages = openings["text"].tolist()
    normalised = [normalise_message(text) for text in raw_messages]

    keep = [i for i, text in enumerate(normalised) if text]
    empty = len(normalised) - len(keep)
    raw_messages = [raw_messages[i] for i in keep]
    normalised = [normalised[i] for i in keep]
    print(f"      {len(normalised):,} messages ({empty:,} empty after normalisation)", flush=True)

    print("[2/5] Building TF-IDF and reducing ...", flush=True)
    vectoriser, matrix = build_features(normalised)
    svd, reduced = reduce_dimensions(matrix, SVD_COMPONENTS)
    vocabulary = vectoriser.get_feature_names_out()

    document_frequency = np.asarray((matrix > 0).sum(axis=0)).ravel()
    features = {
        "n_documents": matrix.shape[0],
        "vocabulary_size": matrix.shape[1],
        "sparsity_pct": round(100 * (1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])), 3),
        "mean_terms_per_document": round(matrix.nnz / matrix.shape[0], 1),
        "svd_components": svd.n_components,
        "explained_variance_pct": round(100 * float(svd.explained_variance_ratio_.sum()), 2),
        "flight_retained": bool("flight" in set(vocabulary)),
        "top_document_frequency_terms": [
            str(vocabulary[i]) for i in np.argsort(-document_frequency)[:20]
        ],
    }
    print(f"      {features['n_documents']:,} x {features['vocabulary_size']:,}, "
          f"{features['explained_variance_pct']}% variance", flush=True)

    print("[3/5] Sweeping K ...", flush=True)
    k_values = [k for k in K_VALUES if k < len(normalised)]
    sweep = sweep_k(reduced, k_values)

    chosen_k = args.k or pick_k(sweep)
    rationale = (
        "Chosen by hand from the sweep, not by a single metric: the largest K whose clusters "
        "stay readable without fragmenting into near-duplicates. Silhouette on sparse text is "
        "too weak to decide this on its own."
    )

    print(f"[4/5] Clustering at K={chosen_k} and describing ...", flush=True)
    model = KMeans(n_clusters=chosen_k, n_init=10, random_state=config.RANDOM_SEED).fit(reduced)
    labels = model.labels_
    similarity = cosine_similarity(model.cluster_centers_)
    np.fill_diagonal(similarity, 0.0)

    clusters = []
    for cluster in range(chosen_k):
        described = describe_cluster(
            cluster, labels, matrix, reduced, model.cluster_centers_,
            vocabulary, normalised, raw_messages, config.RANDOM_SEED,
        )
        described["nearest_cluster"] = int(np.argmax(similarity[cluster]))
        described["nearest_similarity"] = round(float(similarity[cluster].max()), 3)
        clusters.append(described)
    clusters.sort(key=lambda item: -item["size"])

    corpus_top_terms = set(features["top_document_frequency_terms"])
    checks = quality_checks(clusters, labels, raw_messages, corpus_top_terms)

    print("[5/5] Checking normalisation sensitivity and writing reports ...", flush=True)
    sensitivity = normalisation_sensitivity(raw_messages, labels, chosen_k)

    stats = {
        "brand": BRAND,
        "source": args.source,
        "messages": len(normalised),
        "empty_after_normalisation": empty,
        "earliest": str(openings["created_at"].min()),
        "latest": str(openings["created_at"].max()),
        "features": features,
        "k_values_tested": k_values,
        "k_sweep": sweep.to_dict(orient="records"),
        "chosen_k": chosen_k,
        "k_rationale": rationale,
        "clusters": clusters,
        "quality_checks": checks,
        "sensitivity": sensitivity,
        "taxonomy_status": "UNVALIDATED - human interpretation, no labelled data yet",
    }

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = config.REPORTS_DIR / f"phase4_intent_stats{suffix}.json"
    report_path = config.REPORTS_DIR / f"phase4_intent_discovery{suffix}.md"
    stats_path.write_text(json.dumps(stats, indent=2, default=json_default), encoding="utf-8")
    report_path.write_text(render_report(stats, sweep, clusters, args.source), encoding="utf-8")

    print(f"\nReport: {report_path}")
    print(f"Stats:  {stats_path}")
    return 0


def pick_k(sweep: pd.DataFrame) -> int:
    # Prefer a K without near-duplicate centroids or a dominant cluster; ties go to more detail.
    usable = sweep[(sweep["max_centroid_similarity"] < 0.8) & (sweep["largest_cluster_pct"] < 40)]
    if usable.empty:
        return int(sweep.loc[sweep["silhouette"].idxmax(), "k"])
    return int(usable["k"].max())


if __name__ == "__main__":
    sys.exit(main())
