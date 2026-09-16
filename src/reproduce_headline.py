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

    rows_digest = sha256(config.CORPUS_DIR / "evaluation_rows.jsonl")
    stats = json.loads((config.REPORTS_DIR / "phase6_evaluation_stats.json").read_text(encoding="utf-8"))
    classifier = stats["classifier"]["all_248"]["baseline_1_tfidf_logreg"]["metrics"]
    print("\n== timings (seconds)")
    for label, seconds in timings:
        print(f"  {seconds:>7}  {label}")
    print(f"  {round(time.perf_counter() - started, 1):>7}  total")
    print(f"\nevaluation_rows.jsonl identical to the recorded run: {rows_digest == EXPECTED_EVALUATION_ROWS_SHA256}")
    print(f"TF-IDF + LR on all 248 rows: accuracy {classifier['accuracy']}, macro-F1 {classifier['macro_f1']}")
    print(f"Escalation rate (all 248): {stats['escalation']['all_248']['escalate']['pct']}%")
    return 0 if rows_digest == EXPECTED_EVALUATION_ROWS_SHA256 else 1


if __name__ == "__main__":
    sys.exit(main())
