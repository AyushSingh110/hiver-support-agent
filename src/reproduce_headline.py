"""Reproduce the headline deterministic evaluation from the raw dataset, without an LLM.

Needs only `data/raw/twcs/twcs.csv` (see README). Generation is not rerun: the recorded
Phase 6C outputs in `artifacts/phase6/` are hash-checked and copied into place, and
`src.evaluate` then re-verifies that retrieval, classification, G1-G7 and both baselines
reproduce them exactly.

    python -m src.reproduce_headline
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from src import config

ARTIFACTS = config.PROJECT_ROOT / "artifacts" / "phase6"
EXPECTED_EVALUATION_ROWS_SHA256 = "a28dc0881327959ddf320b05d437d8a0376387ed07a8ea186e17bc583deec081"
PHASE5_RETEST_KEY = config.REPORTS_DIR / "phase5_retest_r01_key.json"
CONVERSATIONS = config.PROCESSED_DIR / "conversations.parquet"
EVALUATION_STATS = config.REPORTS_DIR / "phase6_evaluation_stats.json"
CORPUS_STATS = config.REPORTS_DIR / "phase6_corpus_stats.json"
PHASE5_AGREEMENT = config.REPORTS_DIR / "phase5_retest_agreement.json"
ROUND1_RATINGS = config.REPORTS_DIR / "phase6f_reply_ratings_round1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restore_generation_artifacts() -> None:
    """Copy the frozen 6C records into data/processed/phase6 after checking SHA256SUMS."""
    config.CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for line in (ARTIFACTS / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        source, target = ARTIFACTS / name, config.CORPUS_DIR / name
        if sha256(source) != digest:
            raise RuntimeError(f"{source} does not match SHA256SUMS")
        if target.exists() and sha256(target) != digest:
            raise RuntimeError(f"{target} exists with different content; refusing to overwrite")
        shutil.copyfile(source, target)


def count_clean_conversations(path: Path = CONVERSATIONS, brand: str = config.BRAND) -> int:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["brand", "status"])
    return int(pc.sum(pc.and_(pc.equal(table["brand"], brand), pc.equal(table["status"], "clean"))).as_py())


def headline_summary(evaluation: dict, corpus: dict, clean_conversations: int, phase5: dict,
                     round1: dict, runtime_seconds: float, rows_identical: bool) -> str:
    """Reviewer convenience only; REPORT.md and the reports are authoritative. Every value is read."""
    subsets = (("all 248", "all_248"), ("b02 100", "independent_b02_100"))

    def classifier(name, subset):
        m = evaluation["classifier"][subset][name]["metrics"]
        return f"accuracy {m['accuracy']:.4f}, macro-F1 {m['macro_f1']:.4f}"

    def retrieval(system, subset):
        g = evaluation["retrieval"][subset][system]["gold_intent_vs_weak_label_agreement"]
        return (f"gold-vs-weak consistency {g['agreement_pct_of_weakly_labelled_items']}%, "
                f"queries with any gold match {g['queries_with_any_match']['pct']}%")

    gen, esc = evaluation["generation"]["all_248"], evaluation["escalation"]["all_248"]
    rt = round1["test_retest"]
    lines = [
        "", "=" * 78, "HEADLINE SUMMARY (convenience only; REPORT.md is authoritative)", "=" * 78,
        "1. DATA",
        f"   AmericanAir clean conversations      {clean_conversations:,}",
        f"   customer-rooted openings             {corpus['eligibility']['start']:,}",
        f"   retrieval corpus (after exclusions)  {corpus['retrieval_corpus_rows']:,}",
        "2. CLASSIFICATION (golden intent labels, one annotator)",
    ]
    for label, subset in subsets:
        lines.append(f"   {label}: majority      {classifier('baseline_0_majority_class', subset)}")
        lines.append(f"   {label}: TF-IDF + LR   {classifier('baseline_1_tfidf_logreg', subset)}")
    lines.append("3. RETRIEVAL k=5 (gold query intent vs weak evidence labels; not retrieval accuracy)")
    for label, subset in subsets:
        lines.append(f"   {label}: TF-IDF  {retrieval('tfidf', subset)}")
        lines.append(f"   {label}: random  {retrieval('random', subset)}")
    lines += [
        "4. GENERATION (recorded 6C run; check pass rate is not correctness)",
        f"   output parsed                        {gen['parsed']['pct']}% ({gen['parsed']['count']}/{gen['parsed']['total']})",
        f"   passed all G1-G7 pattern checks      {gen['all_checks_passed']['pct']}% "
        f"({gen['all_checks_passed']['count']}/{gen['all_checks_passed']['total']})",
        f"   G2 unsupported URL / G7 copying      {gen['flags']['G2_unsupported_url']['count']} / "
        f"{gen['flags']['G7_excessive_copying']['count']}",
        f"   needs_more_information = true        {gen['needs_more_information_true']['pct']}% of parsed",
        "5. ESCALATION (descriptive; no outcome labels)",
        f"   escalated / auto-handled             {esc['escalate']['pct']}% / {esc['auto_handle']['pct']}%",
        f"   escalated when classifier correct    {esc['escalation_rate_when_classifier_correct']['pct']}%",
        f"   escalated when classifier wrong      {esc['escalation_rate_when_classifier_wrong']['pct']}%",
        "6. ANNOTATION AND REPLY-RATING EVIDENCE",
        f"   Phase 5 intent labels: intra-annotator test-retest, n={phase5['n']}, "
        f"raw {phase5['agreement']['raw_agreement_pct']}%, kappa {phase5['agreement']['cohens_kappa']} "
        f"CI [{phase5['bootstrap']['ci_lower']}, {phase5['bootstrap']['ci_upper']}] "
        "(not inter-annotator)",
        f"   Phase 6F round-1 reply ratings: {round1['provenance']['counts']['rated_responses']} ratable "
        "responses, descriptive only; single-annotator, developer-made, AI-assisted "
        "(not independent human ratings)",
        f"   Phase 6F retest: {rt['status'].replace('_', ' ')}; no reply-quality agreement statistic",
        "   LLM judge: failed its pre-declared validation; not run on golden rows, not used (D46)",
        "7. REPRODUCTION",
        f"   total runtime                        {runtime_seconds:.1f} s",
        f"   evaluation_rows.jsonl byte-identical to the recorded run: {rows_identical}",
        "=" * 78,
    ]
    return "\n".join(lines)


def steps(skip_phase5_key: bool) -> list[tuple[str, list[str] | None]]:
    python = [sys.executable, "-m"]
    plan = [
        ("profile raw CSV", python + ["src.profile_raw", "--source", "full"]),
        ("reconstruct conversations", python + ["src.reconstruct", "--source", "full"]),
        ("build corpus and weak labels", python + ["src.build_corpus"]),
        ("restore recorded 6C generation outputs", None),
        ("calibrate escalation thresholds", python + ["src.calibrate_escalation"]),
    ]
    if not skip_phase5_key:
        plan.append(("rebuild Phase 5 retest key", python + ["src.build_retest_batch"]))
    plan += [
        ("Phase 5 intent-label retest agreement", python + ["src.analyse_retest"]),
        ("deterministic golden evaluation", python + ["src.evaluate", "--confirm-golden-run"]),
        ("Phase 6F round-1 descriptive ratings", python + [
            "src.analyse_reply_ratings", "--round1", "human_eval/reply_rating_r01_rated.csv",
            "--round1-only", "--confirm-real-ratings"]),
    ]
    return plan


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(argv)
    if not config.RAW_TWCS_CSV.exists():
        print(f"Missing {config.RAW_TWCS_CSV}. Download the dataset first (see README).")
        return 2

    timings, started = [], time.perf_counter()
    for label, command in steps(skip_phase5_key=PHASE5_RETEST_KEY.exists()):
        print(f"\n== {label}", flush=True)
        begin = time.perf_counter()
        if command is None:
            restore_generation_artifacts()
        else:
            subprocess.run(command, cwd=config.PROJECT_ROOT, check=True)
        timings.append((label, round(time.perf_counter() - begin, 1)))

    total = round(time.perf_counter() - started, 1)
    rows_identical = sha256(config.CORPUS_DIR / "evaluation_rows.jsonl") == EXPECTED_EVALUATION_ROWS_SHA256
    print("\n== timings (seconds)")
    for label, seconds in timings:
        print(f"  {seconds:>7}  {label}")
    print(f"  {total:>7}  total")

    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    print(headline_summary(read(EVALUATION_STATS), read(CORPUS_STATS), count_clean_conversations(),
                          read(PHASE5_AGREEMENT), read(ROUND1_RATINGS), total, rows_identical))
    return 0 if rows_identical else 1


if __name__ == "__main__":
    sys.exit(main())
