from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from src import config
from src.leakage import assert_isolated, exclude_golden
from src.taxonomy import normalise

TFIDF_SETTINGS = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.6,
    "sublinear_tf": True,
    "max_features": 100_000,
}

EVIDENCE_FIELDS = [
    "rank", "conversation_id", "tweet_id", "similarity", "historical_customer_message",
    "historical_brand_reply", "historical_weak_intent", "n_turns", "n_brand_turns",
    "customer_followups", "evidence_shape", "retrieved_by",
]


def load_corpus() -> pd.DataFrame:
    path = config.CORPUS_DIR / "retrieval_corpus.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Retrieval corpus missing at {path}; run src.build_corpus first")
    corpus = pd.read_parquet(path)
    assert_isolated(corpus)
    return corpus.reset_index(drop=True)


class Retriever:
    """TF-IDF cosine retrieval over historical conversations.

    Only `customer_message` is indexed: queries are customer openings, so matching
    them against brand replies would reward answers that echo the question.
    """

    def __init__(self, corpus: pd.DataFrame, settings: dict | None = None):
        if corpus.empty:
            raise ValueError("Cannot build a retriever over an empty corpus")
        self.corpus = corpus.reset_index(drop=True)
        self.vectoriser = TfidfVectorizer(**(settings or TFIDF_SETTINGS))
        self.matrix = self.vectoriser.fit_transform(self.corpus["customer_message"].map(normalise))
        self.conversation_ids = self.corpus["conversation_id"].to_numpy()
        self.customer_ids = self.corpus["customer_author_id"].astype(str).to_numpy()

    def _eligible(self, exclude_conversation_id: str | None,
                  exclude_customer_id: str | None) -> np.ndarray:
        eligible = np.ones(len(self.corpus), dtype=bool)
        if exclude_conversation_id is not None:
            eligible &= self.conversation_ids != exclude_conversation_id
        if exclude_customer_id is not None:
            eligible &= self.customer_ids != str(exclude_customer_id)
        return eligible

    def _format(self, positions: np.ndarray, scores: np.ndarray, source: str) -> list[dict]:
        evidence = []
        for rank, (position, score) in enumerate(zip(positions, scores), start=1):
            row = self.corpus.iloc[int(position)]
            evidence.append({
                "rank": rank,
                "conversation_id": row["conversation_id"],
                "tweet_id": int(row["tweet_id"]),
                "similarity": round(float(score), 6),
                "historical_customer_message": row["customer_message"],
                "historical_brand_reply": row["brand_reply"],
                "historical_weak_intent": None if pd.isna(row["weak_label"]) else row["weak_label"],
                "n_turns": int(row["n_turns"]),
                "n_brand_turns": int(row["n_brand_turns"]),
                "customer_followups": int(row["customer_followups"]),
                "evidence_shape": "two_turn" if int(row["n_turns"]) == 2 else "multi_turn",
                "retrieved_by": source,
            })
        return evidence

    def retrieve(self, query: str, k: int = 5, *, exclude_conversation_id: str | None = None,
                 exclude_customer_id: str | None = None) -> list[dict]:
        normalised = normalise(query)
        eligible = self._eligible(exclude_conversation_id, exclude_customer_id)
        if not eligible.any() or not normalised:
            return []

        scores = linear_kernel(self.vectoriser.transform([normalised]), self.matrix).ravel()
        scores[~eligible] = -1.0
        # Sort by descending similarity, then conversation_id, so ties are stable.
        order = np.lexsort((self.conversation_ids, -scores))[:k]
        order = np.array([p for p in order if scores[p] > 0], dtype=int)
        return self._format(order, scores[order], "tfidf")

    def retrieve_random(self, k: int = 5, *, seed: int, exclude_conversation_id: str | None = None,
                        exclude_customer_id: str | None = None) -> list[dict]:
        eligible = np.flatnonzero(self._eligible(exclude_conversation_id, exclude_customer_id))
        if eligible.size == 0:
            return []
        rng = np.random.default_rng(seed)
        chosen = rng.choice(eligible, size=min(k, eligible.size), replace=False)
        chosen = np.sort(chosen)
        return self._format(chosen, np.zeros(len(chosen)), "random")


def summarise(results: list[list[dict]], gold_intents: list[str], k: int, system: str) -> dict:
    available = [r for r in results if r]
    top1 = [r[0]["similarity"] for r in available]
    all_scores = [item["similarity"] for r in available for item in r]

    labelled_hits, labelled_total = 0, 0
    any_match_queries = 0
    for evidence, gold in zip(results, gold_intents):
        labelled = [e for e in evidence if e["historical_weak_intent"]]
        labelled_total += len(labelled)
        matches = sum(1 for e in labelled if e["historical_weak_intent"] == gold)
        labelled_hits += matches
        if matches:
            any_match_queries += 1

    shapes = [e["evidence_shape"] for r in available for e in r]
    two_turn = shapes.count("two_turn")

    return {
        "system": system,
        "k": k,
        "queries": len(results),
        "availability_pct": round(100 * len(available) / len(results), 2),
        "queries_with_no_evidence": len(results) - len(available),
        "top1_similarity_mean": round(float(np.mean(top1)), 4) if top1 else None,
        "top1_similarity_median": round(float(np.median(top1)), 4) if top1 else None,
        "topk_similarity_mean": round(float(np.mean(all_scores)), 4) if all_scores else None,
        "evidence_items_returned": len(all_scores),
        "evidence_with_weak_label": labelled_total,
        "weak_label_coverage_pct": round(100 * labelled_total / len(all_scores), 2) if all_scores else None,
        "intent_consistency_pct": round(100 * labelled_hits / labelled_total, 2) if labelled_total else None,
        "queries_with_any_intent_match_pct": round(100 * any_match_queries / len(results), 2),
        "two_turn_evidence_pct": round(100 * two_turn / len(shapes), 2) if shapes else None,
        "multi_turn_evidence_pct": round(100 * (len(shapes) - two_turn) / len(shapes), 2) if shapes else None,
    }


def similarity_by_shape(results: list[list[dict]]) -> dict:
    by_shape = {"two_turn": [], "multi_turn": []}
    for evidence in results:
        for item in evidence:
            by_shape[item["evidence_shape"]].append(item["similarity"])
    return {
        shape: {
            "n": len(scores),
            "mean_similarity": round(float(np.mean(scores)), 4) if scores else None,
        }
        for shape, scores in by_shape.items()
    }


def render_report(stats: dict) -> str:
    lines = [
        "# Phase 6B — Retrieval",
        "",
        "Generated by `src/retrieve.py`. No LLM, no embeddings, no network.",
        "",
        "## Question",
        "",
        "Does lexical retrieval from historical AmericanAir conversations surface more "
        "topically consistent evidence than random selection from the same pool?",
        "",
        "## Design",
        "",
        f"- **Corpus unit:** one conversation. {stats['corpus_rows']:,} rows, one per "
        "`conversation_id`, each pairing the customer's opening with all brand turns.",
        "- **Retrieval unit:** the conversation, never a detached brand reply.",
        "- **Indexed field:** `customer_message` only. Queries are customer openings, so "
        "indexing brand replies would reward answers that echo the question's vocabulary.",
        "- **Returned:** similarity, both sides of the exchange, turn structure, and the "
        "historical weak intent — enough to inspect why an item was retrieved.",
        f"- **Method:** TF-IDF cosine, settings `{stats['tfidf_settings']}`.",
        "- **Tie-breaking:** descending similarity, then `conversation_id`, so ordering is "
        "deterministic.",
        "",
        "## Historical evidence definition",
        "",
        "Retrieved items are **historical response evidence**, not confirmed resolutions. "
        "The dataset contains no outcome labels, and the corpus structure makes the reason "
        "concrete:",
        "",
        "| | two-turn | 3+ turns |",
        "| --- | --- | --- |",
        f"| Share of corpus | **{stats['corpus_two_turn_pct']}%** | {stats['corpus_multi_turn_pct']}% |",
        f"| Customer responded after the brand | **0.0%** | 99.8% |",
        "",
        "For a clear majority of the corpus the record is one customer message and one brand "
        "reply, with no customer response at all. Even where a follow-up exists it is as "
        "likely to be renewed anger as thanks. **No resolution metric is reported, because "
        "none can be computed honestly.**",
        "",
        "## Leakage controls",
        "",
        "`src/leakage.py` is reused unchanged — there is no second leakage implementation. "
        "The corpus was already filtered at build time (248 golden + 207 same-customer + 1 "
        "exact duplicate = 456 removed). Verified again here:",
        "",
        f"- Golden conversations in corpus: **{stats['leakage_check']['golden_conversations_in_corpus']}**",
        f"- Golden customers in corpus: **{stats['leakage_check']['golden_customers_in_corpus']}**",
        "- `assert_isolated()` is called on every corpus load and raises on any leak.",
        "- Retrieval additionally supports self-exclusion by `conversation_id` and "
        "`customer_author_id` for queries traceable to a corpus row.",
        "",
        "## Results",
        "",
        "> The **only** difference between the two systems is the selection rule. Identical "
        "pool, identical k, identical exclusions. Random selection never consults intent "
        "labels.",
        "",
        "| system | k | availability | top-1 sim (mean) | top-k sim (mean) | intent consistency | weak-label coverage |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in stats["results"]:
        consistency = f"{row['intent_consistency_pct']}%" if row["intent_consistency_pct"] is not None else "n/a"
        top1 = row["top1_similarity_mean"] if row["top1_similarity_mean"] is not None else "n/a"
        lines.append(
            f"| {row['system']} | {row['k']} | {row['availability_pct']}% | {top1} | "
            f"{row['topk_similarity_mean']} | {consistency} | {row['weak_label_coverage_pct']}% |"
        )

    lines += [
        "",
        "### Queries with at least one intent-matching item",
        "",
        "| system | k=1 | k=3 | k=5 |",
        "| --- | --- | --- | --- |",
    ]
    for system in ("tfidf", "random"):
        cells = [
            str(next(r["queries_with_any_intent_match_pct"] for r in stats["results"]
                     if r["system"] == system and r["k"] == k)) + "%"
            for k in (1, 3, 5)
        ]
        lines.append(f"| {system} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Two-turn evidence",
        "",
        "| system | k | two-turn evidence | multi-turn evidence |",
        "| --- | --- | --- | --- |",
    ]
    for row in stats["results"]:
        lines.append(
            f"| {row['system']} | {row['k']} | {row['two_turn_evidence_pct']}% | "
            f"{row['multi_turn_evidence_pct']}% |"
        )
    lines += [
        "",
        "Mean similarity split by evidence shape (TF-IDF, k=5): "
        f"two-turn {stats['similarity_by_shape']['two_turn']['mean_similarity']} "
        f"(n={stats['similarity_by_shape']['two_turn']['n']}), multi-turn "
        f"{stats['similarity_by_shape']['multi_turn']['mean_similarity']} "
        f"(n={stats['similarity_by_shape']['multi_turn']['n']}).",
        "",
        "Two-turn conversations are **kept**. They are real historical responses; the "
        "limitation is that they carry no confirmation, and that is reported rather than "
        "engineered around.",
        "",
        "## Chosen k for generation",
        "",
        f"**k = {stats['selected_k']}.** {stats['k_selection_rationale']}",
        "",
        "## Failure cases",
        "",
        f"- Queries returning no evidence: **{stats['queries_with_no_evidence']}**",
        f"- Golden queries whose text normalises to empty: {stats['degenerate_queries']['empty_after_normalisation']}",
        f"- Golden queries with fewer than three tokens: {stats['degenerate_queries']['fewer_than_three_tokens']}",
        "",
        "## What these results do NOT prove",
        "",
        "- **Not that retrieved evidence resolved anything.** No outcome labels exist.",
        "- **Not that the evidence is useful to a generator.** Usefulness is a property of "
        "the reply, measured later, not of cosine similarity.",
        "- **Not that intent consistency means correctness.** Query intents are **gold** "
        "human labels; retrieved intents are **weak rule-based** labels. Agreement with a "
        "noisy proxy is not accuracy.",
        f"- **Intent consistency is computed over a partial view.** Only "
        f"{stats['corpus_weak_label_pct']}% of the corpus carries a weak label at all, and "
        "the rules cover 10 of 13 intents. `OTHER`, `UNCLEAR` and `non_support_commentary` "
        "are unrepresentable, so a golden query carrying one of those can never register a "
        "match by construction.",
        "- **Higher similarity is not higher quality.** It measures lexical overlap.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    print("[1/4] Loading corpus and golden queries ...", flush=True)
    corpus = load_corpus()
    golden = pd.read_csv(config.GOLDEN_SET, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    retriever = Retriever(corpus)
    print(f"      {len(corpus):,} conversations indexed, {len(golden)} golden queries", flush=True)

    print("[2/4] Running TF-IDF and random retrieval for k in (1, 3, 5) ...", flush=True)
    gold_intents = golden["primary_intent"].tolist()
    results, per_k_tfidf = [], {}
    for k in (1, 3, 5):
        tfidf = [
            retriever.retrieve(
                row["text"], k,
                exclude_conversation_id=row["conversation_id"],
                exclude_customer_id=row["customer_author_id"],
            )
            for _, row in golden.iterrows()
        ]
        random_ = [
            retriever.retrieve_random(
                k, seed=config.RANDOM_SEED + index,
                exclude_conversation_id=row["conversation_id"],
                exclude_customer_id=row["customer_author_id"],
            )
            for index, (_, row) in enumerate(golden.iterrows())
        ]
        per_k_tfidf[k] = tfidf
        results.append(summarise(tfidf, gold_intents, k, "tfidf"))
        results.append(summarise(random_, gold_intents, k, "random"))

    print("[3/4] Selecting k by the pre-declared rule ...", flush=True)
    tfidf_rows = {row["k"]: row for row in results if row["system"] == "tfidf"}
    k3, k5 = tfidf_rows[3], tfidf_rows[5]
    better = (
        k3["intent_consistency_pct"] is not None
        and k5["intent_consistency_pct"] is not None
        and k3["intent_consistency_pct"] - k5["intent_consistency_pct"] >= 5.0
    )
    selected_k = 3 if better else 5
    rationale = (
        "Rule declared before running: k=5 unless k=3 shows materially higher intent "
        f"consistency. k=3 scored {k3['intent_consistency_pct']}% against k=5's "
        f"{k5['intent_consistency_pct']}%, a difference of "
        f"{round((k3['intent_consistency_pct'] or 0) - (k5['intent_consistency_pct'] or 0), 2)} "
        "points, so " + ("k=3 is selected." if better else "k=5 stands.")
    )

    print("[4/4] Writing reports ...", flush=True)
    normalised_golden = golden["text"].map(normalise)
    stats = {
        "corpus_rows": len(corpus),
        "corpus_two_turn_pct": round(100 * float((corpus["n_turns"] == 2).mean()), 1),
        "corpus_multi_turn_pct": round(100 * float((corpus["n_turns"] >= 3).mean()), 1),
        "corpus_weak_label_pct": round(100 * float(corpus["weak_label"].notna().mean()), 1),
        "tfidf_settings": TFIDF_SETTINGS,
        "golden_queries": len(golden),
        "leakage_check": {
            "golden_conversations_in_corpus": len(
                set(golden["conversation_id"]) & set(corpus["conversation_id"])
            ),
            "golden_customers_in_corpus": len(
                set(golden["customer_author_id"]) & set(corpus["customer_author_id"].astype(str))
            ),
        },
        "degenerate_queries": {
            "empty_after_normalisation": int((normalised_golden.str.strip() == "").sum()),
            "fewer_than_three_tokens": int((normalised_golden.str.split().str.len() < 3).sum()),
        },
        "queries_with_no_evidence": max(row["queries_with_no_evidence"] for row in results),
        "results": results,
        "similarity_by_shape": similarity_by_shape(per_k_tfidf[5]),
        "selected_k": selected_k,
        "k_selection_rationale": rationale,
    }

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "phase6_retrieval_stats.json").write_text(
        json.dumps(stats, indent=2, default=str), encoding="utf-8"
    )
    (config.REPORTS_DIR / "phase6_retrieval.md").write_text(render_report(stats), encoding="utf-8")

    for row in results:
        consistency = row["intent_consistency_pct"]
        print(f"  {row['system']:<7} k={row['k']}  avail {row['availability_pct']:>6}%  "
              f"top1 {str(row['top1_similarity_mean']):>7}  "
              f"intent-consistency {str(consistency):>6}%")
    print(f"\nSelected k = {selected_k}")
    print(f"Report: {config.REPORTS_DIR / 'phase6_retrieval.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
