"""Size the escalation thresholds on a fixed, golden-free development pool.

Rule, declared before any value was computed (docs/DECISION_LOG.md):
    T_conf = 20th percentile of intent_confidence on the pool
    T_sim  = 20th percentile of top1_similarity on the pool

There are no escalation outcome labels, so these are traffic-sizing choices, not
accuracy-tuned values. Golden labels are never read; golden identifiers are read only
to prove they are absent.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src import config
from src.classify_intent import build_pipeline, load_training
from src.escalate import (
    HUMAN_REVIEW_POLICY_INTENTS,
    LOW_INTENT_CONFIDENCE,
    WEAK_GROUNDING_FOR_SUBSTANTIVE_REPLY,
    EscalationSignals,
    Thresholds,
    decide,
)
from src.leakage import golden_conversation_ids, golden_customer_ids
from src.retrieve import Retriever, load_corpus
from src.taxonomy import normalise

POOL_SIZE = 2_000
POOL_SEED = config.RANDOM_SEED
PERCENTILE = 20
PERCENTILE_METHOD = "linear"
OOF_FOLDS = 5
OUTPUT = config.REPORTS_DIR / "phase6_escalation_calibration.json"


def golden_identifiers() -> dict[str, frozenset[str]]:
    # Identifier columns only; labels are never loaded.
    golden = pd.read_csv(config.GOLDEN_SET, dtype=str, keep_default_na=False,
                         encoding="utf-8-sig", usecols=["conversation_id", "customer_author_id"])
    return {
        "golden_set_conversation_ids": frozenset(golden["conversation_id"]),
        "golden_set_customer_ids": frozenset(golden["customer_author_id"]),
        "conversation_exclusion_ids": golden_conversation_ids(),
        "customer_exclusion_ids": golden_customer_ids(),
    }


def verify_no_golden(frame: pd.DataFrame, identifiers: dict[str, frozenset[str]]) -> dict[str, int]:
    conversations = set(frame["conversation_id"].astype(str))
    customers = set(frame["customer_author_id"].astype(str))
    hits = {
        name: len((customers if "customer" in name else conversations) & ids)
        for name, ids in identifiers.items()
    }
    if any(hits.values()):
        raise RuntimeError(f"Golden identifiers found in the development pool: {hits}")
    return hits


def sample_pool(corpus: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(POOL_SEED)
    positions = np.sort(rng.choice(len(corpus), size=POOL_SIZE, replace=False))
    return corpus.iloc[positions].reset_index(drop=True)


def intent_confidence(pool: pd.DataFrame, training: pd.DataFrame) -> pd.DataFrame:
    """Out-of-fold for rows the classifier was trained on, fitted-model otherwise."""
    texts = training["customer_message"].map(normalise)
    labels = training["weak_label"].to_numpy()

    folds = StratifiedKFold(n_splits=OOF_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    oof = cross_val_predict(build_pipeline(config.RANDOM_SEED), texts, labels,
                            cv=folds, method="predict_proba")
    fitted = build_pipeline(config.RANDOM_SEED).fit(texts, labels)
    classes = fitted.classes_
    oof_by_id = dict(zip(training["conversation_id"], range(len(training))))

    in_training = pool["conversation_id"].isin(oof_by_id).to_numpy()
    probabilities = np.empty((len(pool), len(classes)))
    if in_training.any():
        rows = [oof_by_id[c] for c in pool.loc[in_training, "conversation_id"]]
        probabilities[in_training] = oof[rows]
    if (~in_training).any():
        probabilities[~in_training] = fitted.predict_proba(
            pool.loc[~in_training, "customer_message"].map(normalise))

    return pd.DataFrame({
        "conversation_id": pool["conversation_id"].to_numpy(),
        "in_weak_training": in_training,
        "confidence_source": np.where(in_training, "out_of_fold", "fitted_model"),
        "predicted_intent": classes[probabilities.argmax(axis=1)],
        "intent_confidence": probabilities.max(axis=1),
    })


def retrieval_signals(pool: pd.DataFrame, corpus: pd.DataFrame) -> pd.DataFrame:
    retriever = Retriever(corpus)
    rows = []
    for row in pool.itertuples(index=False):
        evidence = retriever.retrieve(
            row.customer_message, config.RETRIEVAL_TOP_K,
            exclude_conversation_id=row.conversation_id,
            exclude_customer_id=row.customer_author_id,
        )
        rows.append({
            "evidence_count": len(evidence),
            "top1_similarity": evidence[0]["similarity"] if evidence else np.nan,
        })
    return pd.DataFrame(rows)


def describe(values: np.ndarray) -> dict:
    return {
        "n": int(values.size),
        "min": float(values.min()),
        "median": float(np.median(values)),
        "max": float(values.max()),
        f"p{PERCENTILE}": float(np.percentile(values, PERCENTILE, method=PERCENTILE_METHOD)),
    }


def rate(count: int, total: int) -> dict:
    return {"count": int(count), "pct": round(100.0 * count / total, 2)}


def confirm_strict_comparison(thresholds: Thresholds) -> dict[str, bool]:
    """Exercise decide() at the exact thresholds: equal must not fire, below must."""
    base = dict(predicted_intent="baggage", evidence_count=5, reply="synthetic",
                parse_error=None, grounding_flag_codes=(), needs_more_information=False,
                evidence_used=())
    at = decide(EscalationSignals(**base, intent_confidence=thresholds.intent_confidence,
                                  top1_similarity=thresholds.top1_similarity), thresholds)
    below = decide(EscalationSignals(
        **base,
        intent_confidence=float(np.nextafter(thresholds.intent_confidence, 0)),
        top1_similarity=float(np.nextafter(thresholds.top1_similarity, 0)),
    ), thresholds)
    return {
        "equal_confidence_fires": LOW_INTENT_CONFIDENCE in at["reason_codes"],
        "equal_similarity_fires": WEAK_GROUNDING_FOR_SUBSTANTIVE_REPLY in at["reason_codes"],
        "below_confidence_fires": LOW_INTENT_CONFIDENCE in below["reason_codes"],
        "below_similarity_fires": WEAK_GROUNDING_FOR_SUBSTANTIVE_REPLY in below["reason_codes"],
    }


def calibrate() -> dict:
    identifiers = golden_identifiers()
    corpus = load_corpus()
    training = load_training()

    corpus_hits = verify_no_golden(corpus, identifiers)
    verify_no_golden(training, identifiers)
    if not set(training["conversation_id"]) <= set(corpus["conversation_id"]):
        raise RuntimeError("Weak training set is not a subset of the retrieval corpus")

    pool = sample_pool(corpus)
    pool_hits = verify_no_golden(pool, identifiers)
    pool_ids = sorted(pool["conversation_id"])

    signals = pd.concat([intent_confidence(pool, training),
                         retrieval_signals(pool, corpus)], axis=1)

    confidence = signals["intent_confidence"].to_numpy()
    has_evidence = signals["evidence_count"] > 0
    similarity = signals.loc[has_evidence, "top1_similarity"].to_numpy()

    t_conf = float(np.percentile(confidence, PERCENTILE, method=PERCENTILE_METHOD))
    t_sim = float(np.percentile(similarity, PERCENTILE, method=PERCENTILE_METHOD))
    thresholds = Thresholds(intent_confidence=t_conf, top1_similarity=t_sim,
                            min_evidence_count=config.ESCALATION_MIN_EVIDENCE_COUNT)

    # The same strict comparisons escalate.py applies; no evidence counts as weak.
    low_conf = signals["intent_confidence"] < t_conf
    weak_sim = ~has_evidence | (signals["top1_similarity"] < t_sim)
    insufficient = signals["evidence_count"] < thresholds.min_evidence_count
    policy = signals["predicted_intent"].isin(HUMAN_REVIEW_POLICY_INTENTS)
    measurable = low_conf | weak_sim | insufficient | policy
    n = len(signals)

    patterns = (
        pd.DataFrame({"low_intent_confidence": low_conf, "weak_similarity": weak_sim,
                      "insufficient_evidence": insufficient,
                      "explicit_human_review_policy": policy})
        .apply(lambda r: "+".join(k for k, v in r.items() if v) or "none", axis=1)
        .value_counts()
    )

    return {
        "rule": {
            "T_conf": f"{PERCENTILE}th percentile of intent_confidence on the development pool",
            "T_sim": f"{PERCENTILE}th percentile of top1_similarity on the development pool "
                     "(rows with at least one retrieved item)",
            "percentile_method": f"numpy.percentile(method='{PERCENTILE_METHOD}')",
            "comparison": "strict '<' in src/escalate.py",
            "interpretation": "traffic-sizing rule; no escalation outcome labels exist, so the "
                              "thresholds are not optimised, validated or accuracy-tuned",
        },
        "thresholds": {
            "T_conf": t_conf,
            "T_sim": t_sim,
            "min_evidence_count": thresholds.min_evidence_count,
            "min_evidence_count_source": "config.ESCALATION_MIN_EVIDENCE_COUNT (unchanged)",
        },
        "development_pool": {
            "source": "data/processed/phase6/retrieval_corpus.parquet (golden-free)",
            "source_population": len(corpus),
            "source_build_exclusions": json.loads(
                (config.REPORTS_DIR / "phase6_corpus_stats.json").read_text(encoding="utf-8")
            )["leakage_control"]["removed"],
            "size": n,
            "seed": POOL_SEED,
            "sampling": "numpy.random.default_rng(seed).choice(len(corpus), size, "
                        "replace=False), positions sorted; uniform, not by intent or cluster",
            "conversation_ids_sha256": hashlib.sha256("\n".join(pool_ids).encode()).hexdigest(),
            "retrieval_exclusions_per_query": ["own conversation_id", "own customer_author_id"],
            "retrieval_k": config.RETRIEVAL_TOP_K,
            "golden_identifier_hits": {"corpus": corpus_hits, "pool": pool_hits},
            "golden_labels_read": False,
        },
        "confidence": {
            "classifier": "src.classify_intent.build_pipeline(seed=42), weak labels only",
            "rows_out_of_fold": int(signals["in_weak_training"].sum()),
            "rows_fitted_model": int((~signals["in_weak_training"]).sum()),
            "oof": f"StratifiedKFold(n_splits={OOF_FOLDS}, shuffle=True, random_state=42) "
                   "over the full weak training set",
            "predicted_intent_source": "argmax of the same probabilities as the confidence",
            "distribution": describe(confidence),
            "distribution_out_of_fold": describe(confidence[signals["in_weak_training"].to_numpy()]),
            "distribution_fitted_model": describe(confidence[~signals["in_weak_training"].to_numpy()]),
            "rows_exactly_equal_T_conf": int((signals["intent_confidence"] == t_conf).sum()),
        },
        "similarity": {
            "rows_with_evidence": int(has_evidence.sum()),
            "rows_without_evidence": int((~has_evidence).sum()),
            "evidence_count_distribution": {
                str(k): int(v) for k, v in signals["evidence_count"].value_counts().sort_index().items()
            },
            "distribution": describe(similarity),
            "rows_exactly_equal_T_sim": int((signals["top1_similarity"] == t_sim).sum()),
        },
        "rule_firing_on_pool": {
            "insufficient_evidence": rate(insufficient.sum(), n),
            "explicit_human_review_policy": rate(policy.sum(), n),
            "low_intent_confidence": rate(low_conf.sum(), n),
            "weak_similarity_condition_only": rate(weak_sim.sum(), n),
            "note": "weak_grounding_for_substantive_reply also requires "
                    "needs_more_information == false; the pool has no generated replies, so "
                    "only its similarity condition is measured here",
            "not_measurable_without_generation": [
                "generation_unusable", "grounding_check_failed", "invalid_evidence_citation",
                "weak_grounding_for_substantive_reply (full rule)",
            ],
        },
        "overlap": {
            "confidence_only": rate((low_conf & ~weak_sim).sum(), n),
            "similarity_only": rate((weak_sim & ~low_conf).sum(), n),
            "confidence_and_similarity": rate((low_conf & weak_sim).sum(), n),
            "explicit_human_review_policy": rate(policy.sum(), n),
            "insufficient_evidence": rate(insufficient.sum(), n),
            "any_measurable_rule": rate(measurable.sum(), n),
            "none": rate((~measurable).sum(), n),
            "patterns": {k: int(v) for k, v in patterns.sort_index().items()},
        },
        "predicted_intent_distribution": {
            k: int(v) for k, v in signals["predicted_intent"].value_counts().sort_index().items()
        },
        "strict_comparison_check": confirm_strict_comparison(thresholds),
    }


def main() -> int:
    started = time.perf_counter()
    result = calibrate()
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"written {OUTPUT} in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
