"""Phase 6E deterministic evaluation harness.

Three layers, kept apart on purpose:

1. Evaluation functions: take arrays and records, never read files.
2. Integrity and isolation checks: raise `EvaluationIntegrityError` on any violation.
3. Golden execution: the only code that reads gold labels, and only when called with
   `confirm=True` (from the CLI: `--confirm-golden-run`).

No LLM is called. Generation is evaluated from the recorded Phase 6C outputs.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src import config
from src.escalate import (
    POLICY_VERSION,
    PREDICTABLE_INTENTS,
    WEAK_GROUNDING_FOR_SUBSTANTIVE_REPLY,
    build_signals,
    decide,
    default_thresholds,
)
from src.generate_reply import (
    PROMPT_TEMPLATE,
    PROMPT_VERSION,
    check_grounding,
    copy_similarity,
    generate_baseline_echo,
    generate_baseline_template,
)
from src.taxonomy import ALL_LABELS, TAXONOMY_VERSION, normalise

SEED = config.RANDOM_SEED
BOOTSTRAP_RESAMPLES = 2_000
K_RULE_MARGIN_POINTS = 5.0
PREDICTABLE_LABELS = [label for label in ALL_LABELS if label in PREDICTABLE_INTENTS]

# The configuration the golden run was produced under. Any drift invalidates the evaluation.
FROZEN = {
    "generation_model": "llama3.1:8b",
    "prompt_version": "p2",
    "retrieval_k": 5,
    "temperature": 0.0,
    "seed": 42,
    "escalation_min_intent_confidence": 0.19128133486537377,
    "escalation_min_retrieval_similarity": 0.198143,
    "escalation_min_evidence_count": 2,
    "escalation_policy_version": "esc-v1",
    "taxonomy_version": "v2",
}

FIRST_PASS_RECORDS = config.CORPUS_DIR / "golden_generation_records.jsonl"
RETRY_RECORDS = config.CORPUS_DIR / "golden_generation_retry1.jsonl"
FINAL_RECORDS = config.CORPUS_DIR / "golden_generation_final.jsonl"
CALIBRATION_REPORT = config.REPORTS_DIR / "phase6_escalation_calibration.json"
RETEST_REPORT = config.REPORTS_DIR / "phase5_retest_agreement.json"
EVALUATION_ROWS = config.CORPUS_DIR / "evaluation_rows.jsonl"
EVALUATION_STATS = config.REPORTS_DIR / "phase6_evaluation_stats.json"
EVALUATION_REPORT = config.REPORTS_DIR / "phase6_evaluation.md"

BASELINE_A_CAVEAT = (
    "Baseline A returns the top-1 historical reply verbatim, so G7 is expected to flag it "
    "by construction. Its G7 rate is not a safety comparison with the LLM system."
)


class EvaluationIntegrityError(RuntimeError):
    """An invariant the evaluation depends on does not hold."""


class GoldenAccessError(RuntimeError):
    """Gold labels were requested without explicit confirmation."""


def rate(count: int, total: int) -> dict:
    return {"count": int(count), "total": int(total),
            "pct": round(100.0 * count / total, 2) if total else None}


def distribution(values) -> dict:
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if values.size == 0:
        return {"n": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "n": int(values.size),
        "min": round(float(values.min()), 6),
        "median": round(float(np.median(values)), 6),
        "mean": round(float(values.mean()), 6),
        "max": round(float(values.max()), 6),
    }


# --------------------------------------------------------------------------- layer 1
# Classifier

def majority_label(training_labels) -> str:
    # Same expression as Phase 6A, so the baseline is identical.
    return pd.Series(list(training_labels)).value_counts().idxmax()


def classification_metrics(truth, predicted, labels: list[str]) -> dict:
    truth = np.asarray(truth, dtype=object)
    predicted = np.asarray(predicted, dtype=object)
    if len(truth) != len(predicted):
        raise ValueError("truth and predicted differ in length")
    if len(truth) == 0:
        raise ValueError("cannot score an empty set")

    report = classification_report(truth, predicted, labels=labels,
                                   output_dict=True, zero_division=0)
    matrix = confusion_matrix(truth, predicted, labels=labels)
    return {
        "n": int(len(truth)),
        "accuracy": round(float((truth == predicted).mean()), 4),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 4),
        "per_intent": {
            label: {
                "support": int(report[label]["support"]),
                "precision": round(float(report[label]["precision"]), 4),
                "recall": round(float(report[label]["recall"]), 4),
                "f1": round(float(report[label]["f1-score"]), 4),
            }
            for label in labels
        },
        "confusion_matrix": {"labels": list(labels), "rows_true_columns_predicted": matrix.tolist()},
    }


def bootstrap_ci(truth, predicted, labels: list[str], *,
                 resamples: int = BOOTSTRAP_RESAMPLES, seed: int = SEED) -> dict:
    """Percentile bootstrap over rows; macro-F1 uses the same fixed label set as the point estimate."""
    truth = np.asarray(truth, dtype=object)
    predicted = np.asarray(predicted, dtype=object)
    rng = np.random.default_rng(seed)
    accuracy, macro_f1 = [], []
    for _ in range(resamples):
        index = rng.integers(0, len(truth), len(truth))
        t, p = truth[index], predicted[index]
        accuracy.append(float((t == p).mean()))
        macro_f1.append(float(f1_score(t, p, labels=labels, average="macro", zero_division=0)))
    return {
        "method": "percentile bootstrap over rows, 95% interval",
        "resamples": resamples,
        "seed": seed,
        "accuracy": [round(float(np.percentile(accuracy, q)), 4) for q in (2.5, 97.5)],
        "macro_f1": [round(float(np.percentile(macro_f1, q)), 4) for q in (2.5, 97.5)],
    }


def predictable_intent_view(truth, predicted) -> dict:
    """SECONDARY: rows whose gold label the classifier can predict, scored over those 10 labels."""
    truth = np.asarray(truth, dtype=object)
    predicted = np.asarray(predicted, dtype=object)
    keep = np.isin(truth, PREDICTABLE_LABELS)
    return {
        "view": "secondary: gold label in the 10 predictable intents",
        "rows_included": int(keep.sum()),
        "rows_outside_view": int((~keep).sum()),
        "metrics": classification_metrics(truth[keep], predicted[keep], PREDICTABLE_LABELS),
    }


# Retrieval

def summarise_evidence(evidence: list[dict]) -> list[dict]:
    return [
        {"rank": e["rank"], "conversation_id": e["conversation_id"],
         "similarity": e["similarity"], "weak_intent": e["historical_weak_intent"]}
        for e in evidence
    ]


def _intent_vs_weak_label(evidence_lists: list[list[dict]], reference_intents: list[str]) -> dict:
    if len(evidence_lists) != len(reference_intents):
        raise ValueError("one reference intent is required per query")
    items = labelled = matching = queries_with_match = 0
    for evidence, reference in zip(evidence_lists, reference_intents):
        items += len(evidence)
        weak = [e["weak_intent"] for e in evidence if e["weak_intent"]]
        labelled += len(weak)
        hits = sum(1 for label in weak if label == reference)
        matching += hits
        queries_with_match += bool(hits)
    return {
        "queries": len(evidence_lists),
        "evidence_items": items,
        "evidence_with_weak_label": labelled,
        "weak_label_coverage_pct": round(100 * labelled / items, 2) if items else None,
        "matching_items": matching,
        "agreement_pct_of_weakly_labelled_items": round(100 * matching / labelled, 2) if labelled else None,
        "queries_with_any_match": rate(queries_with_match, len(evidence_lists)),
    }


def predicted_intent_vs_weak_label_consistency(evidence_lists, predicted_intents) -> dict:
    return {
        "measure": "predicted_intent_vs_weak_label_consistency",
        "definition": "the query's predicted intent against each evidence item's rule-based weak label",
        **_intent_vs_weak_label(evidence_lists, list(predicted_intents)),
    }


def gold_intent_vs_weak_label_agreement(evidence_lists, gold_intents) -> dict:
    return {
        "measure": "gold_intent_vs_weak_label_agreement",
        "definition": "the query's human gold intent against each evidence item's rule-based weak "
                      "label; evidence has no human labels, so this is not retrieval accuracy",
        **_intent_vs_weak_label(evidence_lists, list(gold_intents)),
    }


def similarity_summary(evidence_lists: list[list[dict]]) -> dict:
    return {
        "queries": len(evidence_lists),
        "queries_without_evidence": sum(1 for e in evidence_lists if not e),
        "evidence_count_distribution": dict(sorted(Counter(len(e) for e in evidence_lists).items())),
        "top1_similarity": distribution(e[0]["similarity"] for e in evidence_lists if e),
        "all_topk_similarity": distribution(i["similarity"] for e in evidence_lists for i in e),
    }


def random_retrieval(retriever, queries: list[dict], k: int) -> list[list[dict]]:
    """Frozen Phase 6B baseline: per-query seed 42 + query index, same exclusions as TF-IDF."""
    return [
        retriever.retrieve_random(
            k, seed=SEED + index,
            exclude_conversation_id=query["conversation_id"],
            exclude_customer_id=query["customer_author_id"],
        )
        for index, query in enumerate(queries)
    ]


def k_selection_rule(consistency_k3: float | None, consistency_k5: float | None) -> int:
    """The Phase 6B rule: k=5 unless k=3 is at least 5 points more intent-consistent."""
    if consistency_k3 is None or consistency_k5 is None:
        return 5
    return 3 if consistency_k3 - consistency_k5 >= K_RULE_MARGIN_POINTS else 5


# Generation records

REQUIRED_GENERATION_FIELDS = ("reply", "parse_error", "grounding_flags", "grounding_passed",
                              "needs_more_information", "evidence_used")


def parse_error_kind(error: str | None) -> str | None:
    if error is None:
        return None
    for prefix, kind in (
        ("model did not return valid JSON", "invalid_json"),
        ("model returned JSON that is not an object", "non_object"),
        ("model response missing required fields", "missing_fields"),
        ("'evidence_used'", "invalid_evidence_used"),
        ("'reply' must", "field_type"),
        ("'needs_more_information' must", "field_type"),
    ):
        if error.startswith(prefix):
            return kind
    return "other"


def citation_validity(evidence_used: list[int], evidence_count: int, parsed: bool) -> bool | None:
    if not parsed:
        return None
    return all(1 <= rank <= evidence_count for rank in evidence_used)


def max_copy_similarity(reply: str, evidence: list[dict]) -> float | None:
    if not reply or not evidence:
        return None
    return round(max(copy_similarity(reply, e["historical_brand_reply"]) for e in evidence), 4)


def flag_codes(flags: list[dict]) -> list[str]:
    return sorted({flag["code"] for flag in flags})


def generation_measures(llm: dict, customer_message: str, evidence: list[dict], *,
                        attempt: int, first_attempt_error: str | None) -> dict:
    missing = [field for field in REQUIRED_GENERATION_FIELDS if field not in llm]
    if missing:
        raise EvaluationIntegrityError(f"generation record missing fields: {missing}")

    recorded = flag_codes(llm["grounding_flags"])
    recomputed = flag_codes(check_grounding(llm["reply"], customer_message, evidence))
    if recorded != recomputed:
        raise EvaluationIntegrityError(
            f"recorded grounding flags {recorded} differ from recomputed {recomputed}")

    parsed = llm["parse_error"] is None
    latency = llm.get("latency_seconds")
    return {
        "parsed": parsed,
        "parse_error_kind": parse_error_kind(llm["parse_error"]),
        "reply": llm["reply"],
        "grounding_flag_codes": recorded,
        "grounding_passed": bool(llm["grounding_passed"]),
        "needs_more_information": llm["needs_more_information"] if parsed else None,
        "evidence_used": list(llm["evidence_used"]) if parsed else [],
        "evidence_used_coerced": bool(llm.get("evidence_used_coerced", False)),
        "citation_valid": citation_validity(llm["evidence_used"], len(evidence), parsed),
        "max_copy_similarity": max_copy_similarity(llm["reply"], evidence),
        "latency_seconds": latency,
        "latency_missing": latency is None,
        "attempt": attempt,
        "first_attempt_error": first_attempt_error,
    }


def _flag_counts(rows: list[dict]) -> dict:
    counts = Counter(code for row in rows for code in row["grounding_flag_codes"])
    codes = ["G1_empty_reply", "G2_unsupported_url", "G3_unsupported_number", "G4_action_claim",
             "G5_resolution_claim", "G6_leaked_identifier", "G7_excessive_copying"]
    return {code: rate(counts.get(code, 0), len(rows)) for code in codes}


def generation_summary(rows: list[dict]) -> dict:
    n = len(rows)
    parsed = [r for r in rows if r["parsed"]]
    return {
        "note": "grounding-check behaviour, not reply correctness",
        "rows": n,
        "parsed": rate(len(parsed), n),
        "parse_failures": rate(n - len(parsed), n),
        "parse_failure_kinds": dict(sorted(Counter(r["parse_error_kind"] for r in rows
                                                   if not r["parsed"]).items())),
        "flags": _flag_counts(rows),
        "all_checks_passed": rate(sum(r["grounding_passed"] for r in rows), n),
        "needs_more_information_true": rate(sum(r["needs_more_information"] is True for r in parsed), len(parsed)),
        "needs_more_information_false": rate(sum(r["needs_more_information"] is False for r in parsed), len(parsed)),
        "evidence_used_coerced": rate(sum(r["evidence_used_coerced"] for r in parsed), len(parsed)),
        "citation_valid": rate(sum(r["citation_valid"] is True for r in parsed), len(parsed)),
        "citing_no_evidence": rate(sum(not r["evidence_used"] for r in parsed), len(parsed)),
        "max_copy_similarity": distribution(r["max_copy_similarity"] for r in rows),
        "latency_seconds": distribution(r["latency_seconds"] for r in rows),
        "latency_missing": sum(r["latency_missing"] for r in rows),
        "attempts": dict(sorted(Counter(r["attempt"] for r in rows).items())),
    }


# Baselines

def rebuild_baselines(customer_message: str, evidence: list[dict]) -> dict:
    return {
        "baseline_b_template": generate_baseline_template(customer_message, evidence),
        "baseline_a_echo": generate_baseline_echo(customer_message, evidence),
    }


def baseline_measures(recorded: dict, rebuilt: dict) -> dict:
    same = (recorded["reply"] == rebuilt["reply"]
            and flag_codes(recorded["grounding_flags"]) == flag_codes(rebuilt["grounding_flags"])
            and recorded["grounding_passed"] == rebuilt["grounding_passed"])
    if not same:
        raise EvaluationIntegrityError(f"rebuilt {rebuilt['system']} differs from the recorded output")
    return {"reply": rebuilt["reply"],
            "grounding_flag_codes": flag_codes(rebuilt["grounding_flags"]),
            "grounding_passed": rebuilt["grounding_passed"]}


def baseline_summary(rows: list[dict], system: str) -> dict:
    summary = {
        "system": system,
        "rows": len(rows),
        "flags": _flag_counts(rows),
        "all_checks_passed": rate(sum(r["grounding_passed"] for r in rows), len(rows)),
    }
    if system == "baseline_a_echo":
        summary["caveat"] = BASELINE_A_CAVEAT
    return summary


# Escalation

def escalation_record(predicted_intent: str, confidence: float, evidence: list[dict],
                      llm: dict, thresholds=None) -> dict:
    return decide(build_signals(predicted_intent, confidence, evidence, llm),
                  thresholds or default_thresholds())


def escalation_summary(rows: list[dict]) -> dict:
    """Descriptive only. Each row: decision, reason_codes, predicted_intent, gold_intent,
    classifier_correct, gold_predictable."""
    n = len(rows)
    escalated = [r for r in rows if r["decision"] == "escalate"]
    patterns = Counter("+".join(r["reason_codes"]) or "none" for r in rows)

    def escalation_rate(subset):
        return rate(sum(r["decision"] == "escalate" for r in subset), len(subset))

    by_gold = {}
    for intent in ALL_LABELS:
        subset = [r for r in rows if r["gold_intent"] == intent]
        if subset:
            by_gold[intent] = escalation_rate(subset)

    praise = [r for r in rows if r["predicted_intent"] == "praise_and_compliment"]
    return {
        "note": "descriptive system behaviour; no escalation outcome labels exist",
        "rows": n,
        "auto_handle": rate(n - len(escalated), n),
        "escalate": rate(len(escalated), n),
        "reason_code_counts": {code: rate(c, n) for code, c in
                               sorted(Counter(c for r in rows for c in r["reason_codes"]).items())},
        "reason_code_patterns": dict(sorted(patterns.items())),
        "informational_code_counts": dict(sorted(
            Counter(c for r in rows for c in r["informational_codes"]).items())),
        "weak_grounding_rule_fired": rate(sum(WEAK_GROUNDING_FOR_SUBSTANTIVE_REPLY in r["reason_codes"]
                                              for r in rows), n),
        "predicted_praise_weak_grounding": rate(
            sum(WEAK_GROUNDING_FOR_SUBSTANTIVE_REPLY in r["reason_codes"] for r in praise), len(praise)),
        "escalation_rate_by_gold_intent": by_gold,
        "escalation_rate_when_classifier_correct": escalation_rate([r for r in rows if r["classifier_correct"]]),
        "escalation_rate_when_classifier_wrong": escalation_rate([r for r in rows if not r["classifier_correct"]]),
        "escalation_rate_gold_label_not_predictable": escalation_rate(
            [r for r in rows if not r["gold_predictable"]]),
    }


# Human evidence

def human_agreement_evidence(report: dict) -> dict:
    kind = report.get("measurement_type", "")
    if "intra-annotator" not in kind or "test-retest" not in kind:
        raise EvaluationIntegrityError(f"retest report has unexpected measurement type: {kind!r}")
    return {
        "measurement": "intra-annotator short-gap test-retest agreement (same annotator); "
                       "not inter-annotator agreement",
        "n": report["n"],
        "gap": report.get("gap"),
        "raw_agreement_pct": report["agreement"]["raw_agreement_pct"],
        "cohens_kappa": report["agreement"]["cohens_kappa"],
        "kappa_ci_95": [report["bootstrap"]["ci_lower"], report["bootstrap"]["ci_upper"]],
        "bootstrap_resamples": report["bootstrap"]["resamples"],
        "bootstrap_seed": report["bootstrap"]["seed"],
        "labels_used_as_golden": False,
    }


# End-to-end rows

def assemble_row(*, golden: dict, classifier: dict, evidence: list[dict],
                 random_evidence: list[dict], generation: dict, baselines: dict,
                 escalation: dict) -> dict:
    """Gold fields live only under `gold`; everything else is system output."""
    return {
        "annotation_id": golden["annotation_id"],
        "batch": golden["batch"],
        "conversation_id": golden["conversation_id"],
        "customer_message": golden["text"],
        "gold": {
            "primary_intent": golden["primary_intent"],
            "secondary_intent": golden.get("secondary_intent") or None,
            "ambiguous": golden.get("ambiguous"),
            "predictable_by_classifier": golden["primary_intent"] in PREDICTABLE_INTENTS,
        },
        "system": {
            "classifier": classifier,
            "retrieval": {"tfidf": evidence, "random": random_evidence},
            "generation": generation,
            "escalation": escalation,
        },
        "baselines": baselines,
    }


# --------------------------------------------------------------------------- layer 2

def check_ids(expected, observed, source: str) -> None:
    observed = list(observed)
    duplicates = sorted(i for i, c in Counter(observed).items() if c > 1)
    if duplicates:
        raise EvaluationIntegrityError(f"{source}: duplicate ids {duplicates[:5]}")
    missing = sorted(set(expected) - set(observed))
    unexpected = sorted(set(observed) - set(expected))
    if missing or unexpected:
        raise EvaluationIntegrityError(
            f"{source}: missing {missing[:5]} ({len(missing)}), unexpected {unexpected[:5]} ({len(unexpected)})")


def check_gold_labels(golden: pd.DataFrame) -> None:
    check_ids(golden["annotation_id"], golden["annotation_id"], "golden set")
    labels = golden["primary_intent"].fillna("")
    invalid = sorted(set(labels) - set(ALL_LABELS))
    if invalid:
        raise EvaluationIntegrityError(f"missing or unknown gold labels: {invalid}")


def check_frozen_configuration() -> dict:
    thresholds = default_thresholds()
    actual = {
        "generation_model": config.GENERATION_MODEL,
        "prompt_version": PROMPT_VERSION,
        "retrieval_k": config.RETRIEVAL_TOP_K,
        "temperature": FROZEN["temperature"],
        "seed": config.RANDOM_SEED,
        "escalation_min_intent_confidence": thresholds.intent_confidence,
        "escalation_min_retrieval_similarity": thresholds.top1_similarity,
        "escalation_min_evidence_count": thresholds.min_evidence_count,
        "escalation_policy_version": POLICY_VERSION,
        "taxonomy_version": TAXONOMY_VERSION,
    }
    drift = {k: (FROZEN[k], v) for k, v in actual.items() if v != FROZEN[k]}
    if drift:
        raise EvaluationIntegrityError(f"configuration differs from the frozen run: {drift}")
    return actual


def check_record_configuration(records: list[dict]) -> None:
    for record in records:
        if record.get("row_error") is not None or "llm" not in record:
            raise EvaluationIntegrityError(f"{record.get('annotation_id')}: required generation record missing")
        observed = {
            "generation_model": record["model"],
            "prompt_version": record["prompt_version"],
            "retrieval_k": record["retrieval_k"],
            "temperature": record["temperature"],
            "seed": record["seed"],
        }
        drift = {k: v for k, v in observed.items() if v != FROZEN[k]}
        if drift:
            raise EvaluationIntegrityError(f"{record['annotation_id']}: recorded under {drift}")


def check_merge(first: list[dict], retry: list[dict], final: list[dict]) -> None:
    """The final file must be first-pass successes plus retries, and nothing else."""
    retried = {r["annotation_id"]: r for r in retry}
    ignore = {"attempt", "first_attempt_error"}
    failed_first = {r["annotation_id"] for r in first if r.get("row_error")}
    if failed_first != set(retried):
        raise EvaluationIntegrityError("retry file does not match the first-pass failures")
    by_first = {r["annotation_id"]: r for r in first}

    def strip(record: dict) -> dict:
        return {k: v for k, v in record.items() if k not in ignore}

    for record in final:
        source = retried.get(record["annotation_id"], by_first[record["annotation_id"]])
        if strip(record) != strip(source):
            raise EvaluationIntegrityError(f"{record['annotation_id']}: final record differs from its source")


def check_predictions(ids, predicted, recorded) -> None:
    invalid = sorted({p for p in predicted if p not in PREDICTABLE_INTENTS})
    if invalid:
        raise EvaluationIntegrityError(f"invalid predicted intents: {invalid}")
    differ = [i for i, p, r in zip(ids, predicted, recorded) if p != r]
    if differ:
        raise EvaluationIntegrityError(f"refitted classifier differs from the recorded run on {differ[:5]}")


def check_retrieval_matches(annotation_id: str, recomputed: list[dict], recorded: list[dict]) -> None:
    now = [(e["conversation_id"], e["similarity"]) for e in recomputed]
    then = [(e["conversation_id"], e["similarity"]) for e in recorded]
    if now != then:
        raise EvaluationIntegrityError(f"{annotation_id}: recomputed evidence differs from the recorded run")


def check_isolation(corpus: pd.DataFrame, training: pd.DataFrame, calibration: dict) -> dict:
    from src.leakage import assert_isolated, exclude_golden

    try:
        assert_isolated(corpus)
        assert_isolated(training)
    except AssertionError as error:
        raise EvaluationIntegrityError(str(error)) from error

    _, audit = exclude_golden(corpus, text_column="customer_message", check_near_duplicates=True)
    if audit["rows_removed"]:
        raise EvaluationIntegrityError(f"golden leakage in the retrieval corpus: {audit['removed']}")
    if not set(training["conversation_id"]) <= set(corpus["conversation_id"]):
        raise EvaluationIntegrityError("weak training set is not a subset of the retrieval corpus")

    pool = calibration["development_pool"]
    hits = [v for scope in pool["golden_identifier_hits"].values() for v in scope.values()]
    if any(hits) or pool["golden_labels_read"] is not False:
        raise EvaluationIntegrityError("threshold calibration touched golden data")
    return {"corpus_leakage_audit": audit["removed"], "calibration_golden_hits": sum(hits)}


# --------------------------------------------------------------------------- layer 3

def sha256_file(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def provenance() -> dict:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=config.PROJECT_ROOT, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    files = {
        "golden_set": config.GOLDEN_SET,
        "golden_conversation_exclusions": config.GOLDEN_CONVERSATION_EXCLUSIONS,
        "golden_customer_exclusions": config.GOLDEN_CUSTOMER_EXCLUSIONS,
        "retrieval_corpus": config.CORPUS_DIR / "retrieval_corpus.parquet",
        "weak_training_set": config.CORPUS_DIR / "weak_training_set.parquet",
        "generation_records_final": FINAL_RECORDS,
        "escalation_calibration": CALIBRATION_REPORT,
        "retest_report": RETEST_REPORT,
    }
    return {
        "frozen_configuration": FROZEN,
        "prompt_template_sha256": hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest(),
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": SEED},
        "random_retrieval_seed": "42 + query index (frozen Phase 6B definition)",
        "file_sha256": {name: sha256_file(path) for name, path in files.items()},
        "git_head": head,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
    }


def load_golden(*, confirm: bool) -> pd.DataFrame:
    if confirm is not True:
        raise GoldenAccessError("gold labels are only read by an explicitly confirmed golden run")
    return pd.read_csv(config.GOLDEN_SET, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def development_pool_k_check(corpus: pd.DataFrame, retriever) -> dict:
    """Sensitivity check only: the Phase 6B rule on the golden-free Phase 6D pool. k stays 5."""
    from src.calibrate_escalation import sample_pool

    pool = sample_pool(corpus)
    queries = pool[pool["weak_label"].notna()]
    results = {}
    for k in (3, 5):
        evidence = [
            summarise_evidence(retriever.retrieve(
                row.customer_message, k,
                exclude_conversation_id=row.conversation_id,
                exclude_customer_id=row.customer_author_id))
            for row in queries.itertuples(index=False)
        ]
        results[k] = _intent_vs_weak_label(evidence, queries["weak_label"].tolist())
    selected = k_selection_rule(results[3]["agreement_pct_of_weakly_labelled_items"],
                                results[5]["agreement_pct_of_weakly_labelled_items"])
    return {
        "purpose": "sensitivity check of the Phase 6B k rule without golden data",
        "pool": "Phase 6D development pool (2,000 rows, seed 42); queries restricted to rows with a weak label",
        "queries": len(queries),
        "reference": "query weak label against evidence weak labels (weak vs weak)",
        "k3": results[3],
        "k5": results[5],
        "rule": f"k=5 unless k=3 is at least {K_RULE_MARGIN_POINTS} points more consistent",
        "rule_selects": selected,
        "frozen_k": FROZEN["retrieval_k"],
        "frozen_k_changed": False,
    }


def _subsets(rows: list[dict]) -> dict[str, list[dict]]:
    return {"all_248": rows, "independent_b02_100": [r for r in rows if r["batch"] == "b02"]}


def run_golden_evaluation(*, confirm: bool = False) -> dict:
    if confirm is not True:
        raise GoldenAccessError("run_golden_evaluation requires confirm=True")

    from src.classify_intent import build_pipeline, load_training
    from src.retrieve import Retriever, load_corpus

    configuration = check_frozen_configuration()
    corpus = load_corpus()
    training = load_training()
    calibration = json.loads(CALIBRATION_REPORT.read_text(encoding="utf-8"))
    isolation = check_isolation(corpus, training, calibration)

    golden = load_golden(confirm=True)
    if len(golden) != 248 or int((golden["batch"] == "b02").sum()) != 100:
        raise EvaluationIntegrityError("golden set must have 248 rows including 100 from b02")
    check_gold_labels(golden)
    ids = golden["annotation_id"].tolist()

    first, retry, final = (read_jsonl(p) for p in (FIRST_PASS_RECORDS, RETRY_RECORDS, FINAL_RECORDS))
    check_ids(ids, [r["annotation_id"] for r in final], "final generation records")
    check_merge(first, retry, final)
    check_record_configuration(final)
    records = {r["annotation_id"]: r for r in final}
    for row in golden.itertuples(index=False):
        if records[row.annotation_id]["eval_only_gold_intent"] != row.primary_intent:
            raise EvaluationIntegrityError(f"{row.annotation_id}: recorded gold field differs from the CSV")

    trainable = training[training["weak_label"].isin(ALL_LABELS)]
    pipeline = build_pipeline(SEED).fit(trainable["customer_message"].map(normalise),
                                        trainable["weak_label"])
    texts = golden["text"].map(normalise)
    predicted = pipeline.predict(texts)
    confidence = pipeline.predict_proba(texts).max(axis=1)
    check_predictions(ids, predicted, [records[i]["predicted_intent"] for i in ids])
    majority = majority_label(trainable["weak_label"])

    retriever = Retriever(corpus)
    queries = golden[["conversation_id", "customer_author_id"]].to_dict("records")
    random_lists = random_retrieval(retriever, queries, FROZEN["retrieval_k"])

    rows, escalation_rows = [], []
    for position, row in enumerate(golden.to_dict("records")):
        record = records[row["annotation_id"]]
        evidence = retriever.retrieve(row["text"], FROZEN["retrieval_k"],
                                      exclude_conversation_id=row["conversation_id"],
                                      exclude_customer_id=row["customer_author_id"])
        check_retrieval_matches(row["annotation_id"], evidence, record["evidence"])

        generation = generation_measures(
            record["llm"], row["text"], evidence,
            attempt=record["attempt"], first_attempt_error=record.get("first_attempt_error"))
        rebuilt = rebuild_baselines(row["text"], evidence)
        baselines = {name: baseline_measures(record[name], rebuilt[name]) for name in rebuilt}
        escalation = escalation_record(predicted[position], float(confidence[position]),
                                       evidence, record["llm"])

        rows.append(assemble_row(
            golden=row,
            classifier={"predicted_intent": predicted[position],
                        "confidence": round(float(confidence[position]), 6),
                        "majority_baseline": majority},
            evidence=summarise_evidence(evidence),
            random_evidence=summarise_evidence(random_lists[position]),
            generation=generation,
            baselines=baselines,
            escalation=escalation,
        ))
        escalation_rows.append({
            "batch": row["batch"],
            "decision": escalation["decision"],
            "reason_codes": escalation["reason_codes"],
            "informational_codes": escalation["informational_codes"],
            "predicted_intent": predicted[position],
            "gold_intent": row["primary_intent"],
            "classifier_correct": predicted[position] == row["primary_intent"],
            "gold_predictable": row["primary_intent"] in PREDICTABLE_INTENTS,
        })

    if len(rows) != 248:
        raise EvaluationIntegrityError(f"expected 248 evaluated rows, produced {len(rows)}")

    stats = {
        "provenance": {**provenance(), "checked_configuration": configuration, "isolation": isolation},
        "classifier": {},
        "retrieval": {},
        "generation": {},
        "baselines": {},
        "escalation": {},
    }
    for subset, subset_rows in _subsets(rows).items():
        truth = [r["gold"]["primary_intent"] for r in subset_rows]
        system = [r["system"]["classifier"]["predicted_intent"] for r in subset_rows]
        majority_predictions = [majority] * len(subset_rows)
        stats["classifier"][subset] = {
            name: {"metrics": classification_metrics(truth, pred, ALL_LABELS),
                   "bootstrap": bootstrap_ci(truth, pred, ALL_LABELS),
                   "secondary_predictable_intents": {
                       **predictable_intent_view(truth, pred),
                       "bootstrap": bootstrap_ci(
                           [t for t in truth if t in PREDICTABLE_INTENTS],
                           [p for t, p in zip(truth, pred) if t in PREDICTABLE_INTENTS],
                           PREDICTABLE_LABELS)}}
            for name, pred in (("baseline_0_majority_class", majority_predictions),
                               ("baseline_1_tfidf_logreg", system))
        }

        stats["retrieval"][subset] = {}
        for name in ("tfidf", "random"):
            lists = [r["system"]["retrieval"][name] for r in subset_rows]
            stats["retrieval"][subset][name] = {
                "similarity": similarity_summary(lists) if name == "tfidf" else
                              {"note": "random selection; similarity is not computed"},
                "predicted_intent_vs_weak_label_consistency":
                    predicted_intent_vs_weak_label_consistency(lists, system),
                "gold_intent_vs_weak_label_agreement":
                    gold_intent_vs_weak_label_agreement(lists, truth),
            }

        stats["generation"][subset] = generation_summary([r["system"]["generation"] for r in subset_rows])
        stats["baselines"][subset] = {
            name: baseline_summary([r["baselines"][name] for r in subset_rows], name)
            for name in ("baseline_b_template", "baseline_a_echo")
        }
        stats["escalation"][subset] = escalation_summary(_subsets(escalation_rows)[subset])

    stats["retrieval"]["development_pool_k_check"] = development_pool_k_check(corpus, retriever)
    stats["human_agreement"] = human_agreement_evidence(
        json.loads(RETEST_REPORT.read_text(encoding="utf-8")))
    stats["coverage"] = {
        "golden_rows": 248,
        "evaluated_rows": len(rows),
        "rows_dropped": 0,
        "generation_parse_failures_kept": sum(not r["system"]["generation"]["parsed"] for r in rows),
        "rows_without_evidence_kept": sum(not r["system"]["retrieval"]["tfidf"] for r in rows),
        "rows_retried_in_6c": sum(r["system"]["generation"]["attempt"] == 2 for r in rows),
    }
    return {"rows": rows, "stats": stats}


def render_report(stats: dict) -> str:
    def pct(entry):
        return "—" if entry["pct"] is None else f"{entry['pct']}% ({entry['count']}/{entry['total']})"

    lines = [
        "# Phase 6E — Deterministic evaluation",
        "",
        "Generated by `src/evaluate.py`. No LLM was called; generation is scored from the recorded "
        "Phase 6C outputs. LLM-judge results are reported separately.",
        "",
        "The 248 rows are 148 pilot rows relabelled under taxonomy v2 plus 100 fresh b02 rows. "
        "**b02 is the independent subset; the 248 rows are not an independent 248-example sample.**",
        "",
        "## Classifier",
        "",
        "| System | Subset | Accuracy [95% CI] | Macro-F1 [95% CI] | Weighted-F1 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for subset, systems in stats["classifier"].items():
        for name, result in systems.items():
            m, b = result["metrics"], result["bootstrap"]
            lines.append(f"| {name} | {subset} | {m['accuracy']} {b['accuracy']} | "
                         f"{m['macro_f1']} {b['macro_f1']} | {m['weighted_f1']} |")
    lines += ["", "**Secondary view** — rows whose gold label is one of the 10 predictable intents. "
              "Headline results above keep every row.", "",
              "| System | Subset | Rows | Accuracy [95% CI] | Macro-F1 [95% CI] |",
              "| --- | --- | --- | --- | --- |"]
    for subset, systems in stats["classifier"].items():
        for name, result in systems.items():
            v = result["secondary_predictable_intents"]
            lines.append(f"| {name} | {subset} | {v['rows_included']} | "
                         f"{v['metrics']['accuracy']} {v['bootstrap']['accuracy']} | "
                         f"{v['metrics']['macro_f1']} {v['bootstrap']['macro_f1']} |")

    lines += ["", "## Retrieval (k=5)", "",
              "Neither measure is retrieval accuracy: evidence carries only rule-based weak labels.", "",
              "| System | Subset | Predicted-intent vs weak label | Gold-intent vs weak label | "
              "Queries with any gold match | Weak-label coverage |",
              "| --- | --- | --- | --- | --- | --- |"]
    for subset, systems in stats["retrieval"].items():
        if subset == "development_pool_k_check":
            continue
        for name, result in systems.items():
            a = result["predicted_intent_vs_weak_label_consistency"]
            g = result["gold_intent_vs_weak_label_agreement"]
            lines.append(f"| {name} | {subset} | {a['agreement_pct_of_weakly_labelled_items']}% | "
                         f"{g['agreement_pct_of_weakly_labelled_items']}% | "
                         f"{pct(g['queries_with_any_match'])} | {g['weak_label_coverage_pct']}% |")
    k = stats["retrieval"]["development_pool_k_check"]
    lines += ["", f"**k sensitivity check.** k=5 was selected in Phase 6B by a pre-declared rule "
              f"applied to golden results. Re-applied on the golden-free development pool "
              f"({k['queries']} weakly-labelled queries): k=3 "
              f"{k['k3']['agreement_pct_of_weakly_labelled_items']}%, k=5 "
              f"{k['k5']['agreement_pct_of_weakly_labelled_items']}% → the rule selects "
              f"k={k['rule_selects']}. **k remains frozen at 5.**"]

    lines += ["", "## Generation and baselines", "",
              "Grounding checks detect patterns; passing them is not correctness.", "",
              "| System | Subset | All checks passed | G1 | G2 | G3 | G4 | G5 | G6 | G7 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for subset in stats["generation"]:
        entries = [("llm_grounded", stats["generation"][subset])]
        entries += list(stats["baselines"][subset].items())
        for name, result in entries:
            flags = [pct(v) for v in result["flags"].values()]
            lines.append(f"| {name} | {subset} | {pct(result['all_checks_passed'])} | " + " | ".join(flags) + " |")
    lines += ["", f"_{BASELINE_A_CAVEAT}_", ""]
    for subset, g in stats["generation"].items():
        lines.append(f"- **{subset}:** parse failures {pct(g['parse_failures'])}; "
                     f"needs_more_information true {pct(g['needs_more_information_true'])}; "
                     f"citations valid {pct(g['citation_valid'])}; "
                     f"coerced ranks {pct(g['evidence_used_coerced'])}; attempts {g['attempts']}.")

    lines += ["", "## Escalation (descriptive)", "",
              "No escalation outcome labels exist; a rule firing does not make a decision correct.", ""]
    for subset, e in stats["escalation"].items():
        lines += [f"**{subset}:** escalate {pct(e['escalate'])}, auto-handle {pct(e['auto_handle'])}.", "",
                  "| Reason code | Rows |", "| --- | --- |"]
        lines += [f"| {code} | {pct(v)} |" for code, v in e["reason_code_counts"].items()]
        lines.append("")

    h = stats["human_agreement"]
    lines += ["## Human annotation evidence", "",
              f"{h['measurement']}: n={h['n']}, raw agreement {h['raw_agreement_pct']}%, "
              f"Cohen's κ = {h['cohens_kappa']} (95% CI {h['kappa_ci_95']}). "
              "These labels are not used as golden labels.", "",
              "## Coverage", ""]
    lines += [f"- {k}: {v}" for k, v in stats["coverage"].items()]
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    if "--confirm-golden-run" not in argv:
        print("Refusing to read gold labels. Re-run with --confirm-golden-run once approved.")
        return 2
    result = run_golden_evaluation(confirm=True)
    config.CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVALUATION_ROWS, "w", encoding="utf-8") as handle:
        for row in result["rows"]:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_STATS.write_text(json.dumps(result["stats"], indent=2, default=str), encoding="utf-8")
    EVALUATION_REPORT.write_text(render_report(result["stats"]), encoding="utf-8")
    print(f"rows: {EVALUATION_ROWS}\nstats: {EVALUATION_STATS}\nreport: {EVALUATION_REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
