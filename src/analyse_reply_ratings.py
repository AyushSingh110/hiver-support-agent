"""Phase 6F: analyse the human reply-quality ratings (D47).

All analysis functions work on in-memory inputs, so they are tested on synthetic data
only. The command line refuses to read the real rating files unless it is given explicit
paths and --confirm-real-ratings, and it requires the retest file as well, so round-1
results cannot be computed before the retest exists.

    python -m src.analyse_reply_ratings --round1 PATH --retest PATH --confirm-real-ratings

Every comparison between systems is descriptive: nothing here ranks systems or picks a winner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import cohen_kappa_score

from src import config
from src import build_reply_rating_batch as batch_builder
from src.build_reply_rating_batch import (
    ANNOTATION_ID_PATTERN,
    EVIDENCE_COLUMNS,
    FORBIDDEN_TOKENS,
    NOT_RATED_EMPTY_REPLY,
    RETEST_COLUMNS,
    SCORE_COLUMNS,
    SYSTEMS,
    TO_BE_RATED,
    VALID_SCORES,
    parse_csv,
    validate_round1,
)

ANALYSIS_VERSION = "6f-a1"
BOOTSTRAP_RESAMPLES = 2_000
MEAN_SEED = 42
RETEST_KAPPA_SEED = 44
SCALE = [1, 2, 3, 4, 5]
LLM = "llm_grounded"
PAIRS = [(LLM, "baseline_b_template"), (LLM, "baseline_a_echo")]
CLAIM_FLAG_CODES = frozenset({"G2_unsupported_url", "G3_unsupported_number", "G4_action_claim",
                              "G5_resolution_claim", "G6_leaked_identifier", "G7_excessive_copying"})

RATED_RETEST = batch_builder.HUMAN_EVAL_DIR / "reply_rating_retest_rated.csv"
REAL_RATING_FILES = frozenset({batch_builder.RATED_R01.resolve(), RATED_RETEST.resolve()})
REPORT_JSON = config.REPORTS_DIR / "phase6f_reply_ratings.json"
REPORT_MD = config.REPORTS_DIR / "phase6f_reply_ratings.md"


class RatingValidationError(ValueError):
    """A rating file does not match the protocol."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_ratings(path: Path, *, allow_real: bool = False) -> bytes:
    """The only way this module reads a rating file. Real files need explicit permission."""
    if Path(path).resolve() in REAL_RATING_FILES and not allow_real:
        raise PermissionError(f"{path} is a real rating file; pass allow_real=True deliberately")
    return Path(path).read_bytes()


# --------------------------------------------------------------------------- validation

def require_valid_round1(rated_raw: bytes, blank_raw: bytes, key: list[dict]) -> dict:
    systems_per_item = {}
    for entry in key:
        if entry.get("system") not in SYSTEMS:
            raise RatingValidationError(f"{entry.get('response_id')}: missing or unknown system mapping")
        systems_per_item.setdefault(entry["item_id"], []).append(entry["system"])
    if any(sorted(v) != sorted(SYSTEMS) for v in systems_per_item.values()):
        raise RatingValidationError("system mapping does not give one response per system per item")
    facts = validate_round1(rated_raw, blank_raw, key)
    if not facts["passed"]:
        raise RatingValidationError(f"round-1 ratings failed validation: "
                                    f"{ {k: v for k, v in facts.items() if v not in (True, [], 0)} }")
    return facts


def validate_retest(retest_raw: bytes, blank_raw: bytes) -> dict:
    """Every retest response must be fully rated; text and ids must match the blank retest."""
    header, rated = parse_csv(retest_raw)
    _, blank = parse_csv(blank_raw)
    ids = [r.get("retest_id") for r in rated]
    blank_by_id = {r["retest_id"]: r for r in blank}
    text_columns = ["customer_message"] + EVIDENCE_COLUMNS + ["candidate_reply"]
    facts = {
        "columns_exact": header == RETEST_COLUMNS,
        "row_count": len(rated),
        "expected_row_count": len(blank),
        "retest_ids_match_blank_in_order": ids == [r["retest_id"] for r in blank],
        "retest_ids_unique": len(set(ids)) == len(ids),
        "text_mismatches": sorted((r.get("retest_id"), c) for r in rated if r.get("retest_id") in blank_by_id
                                  for c in text_columns if r.get(c) != blank_by_id[r["retest_id"]][c]),
        "incomplete_or_invalid": sorted((r.get("retest_id"), c) for r in rated for c in SCORE_COLUMNS
                                        if r.get(c) not in VALID_SCORES),
        "prohibited_tokens": sorted((r.get("retest_id"), c) for r in rated for c, v in r.items()
                                    if isinstance(v, str) and (any(t in v for t in FORBIDDEN_TOKENS)
                                                               or ANNOTATION_ID_PATTERN.search(v))),
    }
    facts["passed"] = (facts["columns_exact"] and facts["row_count"] == facts["expected_row_count"]
                       and facts["retest_ids_match_blank_in_order"] and facts["retest_ids_unique"]
                       and not facts["text_mismatches"] and not facts["incomplete_or_invalid"]
                       and not facts["prohibited_tokens"])
    if not facts["passed"]:
        raise RatingValidationError(f"retest ratings failed validation: {facts}")
    return facts


def ratings_table(rated_raw: bytes, key: list[dict]) -> list[dict]:
    """One record per response: hidden identity joined to integer scores (None when unrated)."""
    _, rows = parse_csv(rated_raw)
    by_id = {r["response_id"]: r for r in rows}
    table = []
    for entry in key:
        row = by_id[entry["response_id"]]
        rated = entry["status"] == TO_BE_RATED
        table.append({**entry, "scores": {c: int(row[c]) if rated else None for c in SCORE_COLUMNS}})
    return table


# --------------------------------------------------------------------------- descriptive statistics

def describe(values: list[int]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "distribution": {str(s): 0 for s in SCALE},
                "share_le_2": None, "share_ge_4": None}
    array = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "mean": round(float(array.mean()), 4),
        "median": float(np.median(array)),
        "distribution": {str(s): int((array == s).sum()) for s in SCALE},
        "share_le_2": round(float((array <= 2).mean()), 4),
        "share_ge_4": round(float((array >= 4).mean()), 4),
    }


def item_matrix(table: list[dict]) -> tuple[list[str], dict]:
    """scores[system][dimension] is an array over items (sorted item ids), NaN when unrated."""
    items = sorted({r["item_id"] for r in table})
    position = {item: i for i, item in enumerate(items)}
    scores = {s: {d: np.full(len(items), np.nan) for d in SCORE_COLUMNS} for s in SYSTEMS}
    for record in table:
        for d, value in record["scores"].items():
            if value is not None:
                scores[record["system"]][d][position[record["item_id"]]] = value
    return items, scores


def item_resamples(n_items: int, seed: int = MEAN_SEED, resamples: int = BOOTSTRAP_RESAMPLES) -> np.ndarray:
    """Whole items are resampled, so one draw keeps an item's three responses together."""
    return np.random.default_rng(seed).integers(0, n_items, size=(resamples, n_items))


def bootstrap_mean_interval(values: np.ndarray, draws: np.ndarray) -> list[float] | None:
    with np.errstate(all="ignore"):
        sampled = values[draws]
        counts = (~np.isnan(sampled)).sum(axis=1)
        means = np.where(counts > 0, np.nansum(sampled, axis=1) / np.maximum(counts, 1), np.nan)
    means = means[~np.isnan(means)]
    if means.size == 0:
        return None
    return [round(float(np.percentile(means, q)), 4) for q in (2.5, 97.5)]


def system_statistics(table: list[dict], draws: np.ndarray) -> dict:
    _, scores = item_matrix(table)
    result = {}
    for system in SYSTEMS:
        own = [r for r in table if r["system"] == system]
        result[system] = {
            "responses": len(own),
            "not_rated_empty_reply": sum(r["status"] == NOT_RATED_EMPTY_REPLY for r in own),
            "dimensions": {
                d: {**describe([r["scores"][d] for r in own if r["scores"][d] is not None]),
                    "mean_interval_95": bootstrap_mean_interval(scores[system][d], draws)}
                for d in SCORE_COLUMNS
            },
        }
    return result


def paired_comparisons(table: list[dict], draws: np.ndarray) -> dict:
    _, scores = item_matrix(table)
    result = {}
    for first, second in PAIRS:
        dims = {}
        for d in SCORE_COLUMNS:
            diff = scores[first][d] - scores[second][d]
            paired = diff[~np.isnan(diff)]
            dims[d] = {
                "pairs": int(paired.size),
                "unpaired_items": int(np.isnan(diff).sum()),
                "first_higher": int((paired > 0).sum()),
                "tie": int((paired == 0).sum()),
                "second_higher": int((paired < 0).sum()),
                "mean_difference": round(float(paired.mean()), 4) if paired.size else None,
                "mean_difference_interval_95": bootstrap_mean_interval(diff, draws),
            }
        result[f"{first}_minus_{second}"] = {"kind": "descriptive paired comparison", "dimensions": dims}
    return result


# --------------------------------------------------------------------------- automated vs human

def share(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def claim_flag_crosstab(table: list[dict], automated: dict) -> dict:
    """Deterministic G2-G7 flags against human claim_safety <= 2, per system, rated responses only."""
    result = {}
    for system in SYSTEMS:
        cells = {"flagged_low": 0, "flagged_not_low": 0, "unflagged_low": 0, "unflagged_not_low": 0}
        for record in table:
            if record["system"] != system or record["scores"]["claim_safety"] is None:
                continue
            flagged = bool(set(automated[record["annotation_id"]][system]["flag_codes"]) & CLAIM_FLAG_CODES)
            low = record["scores"]["claim_safety"] <= 2
            cells[f"{'flagged' if flagged else 'unflagged'}_{'low' if low else 'not_low'}"] += 1
        flagged_total = cells["flagged_low"] + cells["flagged_not_low"]
        low_total = cells["flagged_low"] + cells["unflagged_low"]
        unflagged_total = cells["unflagged_low"] + cells["unflagged_not_low"]
        result[system] = {
            **cells,
            "rated": sum(cells.values()),
            "share_of_flagged_rated_low": share(cells["flagged_low"], flagged_total),
            "share_of_low_that_were_flagged": share(cells["flagged_low"], low_total),
            "share_of_unflagged_rated_low": share(cells["unflagged_low"], unflagged_total),
        }
    return result


def llm_behaviour_breakdowns(table: list[dict], automated: dict) -> dict:
    llm = [r for r in table if r["system"] == LLM and r["status"] == TO_BE_RATED]

    by_request = {}
    for value in (True, False):
        subset = [r for r in llm if automated[r["annotation_id"]][LLM]["needs_more_information"] is value]
        by_request[str(value).lower()] = describe([r["scores"]["information_request"] for r in subset])

    by_decision = {}
    for decision in ("auto_handle", "escalate"):
        subset = [r for r in llm if automated[r["annotation_id"]][LLM]["escalation_decision"] == decision]
        low = sum(r["scores"]["claim_safety"] <= 2 for r in subset)
        by_decision[decision] = {
            "rated": len(subset),
            "claim_safety_le_2": low,
            "share_claim_safety_le_2": share(low, len(subset)),
            "dimensions": {d: describe([r["scores"][d] for r in subset]) for d in SCORE_COLUMNS},
        }
    return {"information_request_by_needs_more_information": by_request,
            "ratings_by_escalation_decision": by_decision}


# --------------------------------------------------------------------------- test-retest

def weighted_kappa(first: np.ndarray, second: np.ndarray) -> float:
    """Quadratic-weighted kappa on the fixed 1-5 scale; NaN when undefined (e.g. one shared value)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        return float(cohen_kappa_score(first, second, labels=SCALE, weights="quadratic"))


def agreement(first: np.ndarray, second: np.ndarray) -> dict:
    kappa = weighted_kappa(first, second)
    return {
        "pairs": int(first.size),
        "exact_agreement": round(float((first == second).mean()), 4),
        "within_one_agreement": round(float((np.abs(first - second) <= 1).mean()), 4),
        "mean_absolute_difference": round(float(np.abs(first - second).mean()), 4),
        "weighted_kappa_quadratic": None if np.isnan(kappa) else round(float(kappa), 4),
    }


def bootstrap_kappa(first: np.ndarray, second: np.ndarray, draws: np.ndarray) -> dict:
    """`first`/`second` are units x dimensions; whole units are resampled."""
    values = []
    for row in draws:
        kappa = weighted_kappa(first[row].ravel(), second[row].ravel())
        if not np.isnan(kappa):
            values.append(kappa)
    return {
        "resamples": int(draws.shape[0]),
        "usable_resamples": len(values),
        "undefined_resamples": int(draws.shape[0]) - len(values),
        "interval_95": ([round(float(np.percentile(values, q)), 4) for q in (2.5, 97.5)]
                        if values else None),
    }


def retest_agreement(retest_raw: bytes, retest_key: list[dict], table: list[dict]) -> dict:
    _, rows = parse_csv(retest_raw)
    by_retest = {r["retest_id"]: r for r in rows}
    by_response = {r["response_id"]: r for r in table}
    first, second = [], []
    for entry in retest_key:
        original = by_response[entry["response_id"]]
        if original["status"] != TO_BE_RATED:
            raise RatingValidationError(f"{entry['retest_id']} maps to an unrated response")
        first.append([original["scores"][d] for d in SCORE_COLUMNS])
        second.append([int(by_retest[entry["retest_id"]][d]) for d in SCORE_COLUMNS])
    first, second = np.asarray(first), np.asarray(second)
    draws = np.random.default_rng(RETEST_KAPPA_SEED).integers(
        0, len(first), size=(BOOTSTRAP_RESAMPLES, len(first)))

    per_dimension = {}
    for j, d in enumerate(SCORE_COLUMNS):
        per_dimension[d] = {
            **agreement(first[:, j], second[:, j]),
            "kappa_bootstrap": bootstrap_kappa(first[:, [j]], second[:, [j]], draws),
            "round1_distribution": describe(first[:, j].tolist())["distribution"],
            "retest_distribution": describe(second[:, j].tolist())["distribution"],
        }
    return {
        "measurement": "intra-annotator short-gap test-retest agreement (same annotator); "
                       "not inter-annotator agreement and not independent corroboration",
        "responses": int(len(first)),
        "responses_per_system": {s: sum(e["system"] == s for e in retest_key) for s in SYSTEMS},
        "per_dimension": per_dimension,
        "pooled": {
            **agreement(first.ravel(), second.ravel()),
            "kappa_bootstrap": bootstrap_kappa(first, second, draws),
            "note": "pooled over five dimensions of the same responses; these pairs are not "
                    "independent observations",
        },
        "note": "score distributions are reported because kappa can be low or undefined when "
                "ratings are concentrated on a few values",
    }


# --------------------------------------------------------------------------- orchestration

def analyse(*, round1_raw: bytes, round1_blank_raw: bytes, key: list[dict], automated: dict,
            retest_raw: bytes, retest_blank_raw: bytes, retest_key: list[dict],
            input_kind: str, extra_hashes: dict | None = None) -> dict:
    if input_kind not in ("synthetic", "real"):
        raise ValueError("input_kind must be 'synthetic' or 'real'")
    round1_facts = require_valid_round1(round1_raw, round1_blank_raw, key)
    retest_facts = validate_retest(retest_raw, retest_blank_raw)
    missing = {r["annotation_id"] for r in key} - set(automated)
    if missing:
        raise RatingValidationError(f"automated metadata missing for {len(missing)} items")

    table = ratings_table(round1_raw, key)
    items, _ = item_matrix(table)
    draws = item_resamples(len(items))
    return {
        "provenance": {
            "analysis_version": ANALYSIS_VERSION,
            "input_kind": input_kind,
            "input_sha256": {"round1_rated": sha256_bytes(round1_raw),
                             "round1_blank": sha256_bytes(round1_blank_raw),
                             "retest_rated": sha256_bytes(retest_raw),
                             "retest_blank": sha256_bytes(retest_blank_raw),
                             **(extra_hashes or {})},
            "counts": {"items": len(items), "responses": len(table),
                       "rated_responses": sum(r["status"] == TO_BE_RATED for r in table),
                       "retest_responses": len(retest_key)},
            "seeds": {"item_bootstrap": MEAN_SEED, "retest_kappa_bootstrap": RETEST_KAPPA_SEED},
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "methodology": "D47. Item-level percentile bootstrap: whole items are resampled, so an "
                           "item's three responses stay together. Paired comparisons are "
                           "descriptive. Gold intent labels are not read.",
            "round1_validation": round1_facts,
            "retest_validation": retest_facts,
        },
        "systems": system_statistics(table, draws),
        "paired_comparisons": paired_comparisons(table, draws),
        "claim_flags_vs_human_claim_safety": claim_flag_crosstab(table, automated),
        "llm_behaviour": llm_behaviour_breakdowns(table, automated),
        "test_retest": retest_agreement(retest_raw, retest_key, table),
    }


def load_automated_metadata(path: Path = batch_builder.EVALUATION_ROWS) -> dict:
    """Frozen G-flags, needs_more_information and escalation decision per row. `gold` is never read."""
    automated = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            generation = row["system"]["generation"]
            automated[row["annotation_id"]] = {
                LLM: {"flag_codes": list(generation["grounding_flag_codes"]),
                      "needs_more_information": generation["needs_more_information"],
                      "escalation_decision": row["system"]["escalation"]["decision"]},
                "baseline_a_echo": {"flag_codes": list(row["baselines"]["baseline_a_echo"]["grounding_flag_codes"])},
                "baseline_b_template": {"flag_codes": list(row["baselines"]["baseline_b_template"]["grounding_flag_codes"])},
            }
    return automated


def fmt(value) -> str:
    return "—" if value is None else str(value)


def render_report(stats: dict) -> str:
    p = stats["provenance"]
    lines = [
        "# Phase 6F — Human reply-quality ratings",
        "",
        f"Input: **{p['input_kind']}**. Analysis version {p['analysis_version']}. One annotator, who "
        "also built the system. Blinding was weak (Baseline B is a fixed sentence and Baseline A "
        "repeats evidence item 1). Comparisons are descriptive: no ranking and no winner.",
        "",
        "## Ratings by system",
        "",
        "| System | Dimension | n | Mean [95% item bootstrap] | Median | Share ≤2 | Share ≥4 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for system, entry in stats["systems"].items():
        for d, v in entry["dimensions"].items():
            lines.append(f"| {system} | {d} | {v['n']} | {fmt(v['mean'])} {fmt(v['mean_interval_95'])} | "
                         f"{fmt(v['median'])} | {fmt(v['share_le_2'])} | {fmt(v['share_ge_4'])} |")
    lines += ["", "## Paired descriptive comparisons", "",
              "| Comparison | Dimension | Pairs | First higher / tie / second higher | "
              "Mean difference [95% item bootstrap] |", "| --- | --- | --- | --- | --- |"]
    for name, entry in stats["paired_comparisons"].items():
        for d, v in entry["dimensions"].items():
            lines.append(f"| {name} | {d} | {v['pairs']} | {v['first_higher']} / {v['tie']} / "
                         f"{v['second_higher']} | {fmt(v['mean_difference'])} "
                         f"{fmt(v['mean_difference_interval_95'])} |")
    lines += ["", "## Deterministic G2–G7 flags vs human claim safety ≤2", "",
              "| System | Flagged & ≤2 | Flagged & >2 | Unflagged & ≤2 | Unflagged & >2 |",
              "| --- | --- | --- | --- | --- |"]
    for system, v in stats["claim_flags_vs_human_claim_safety"].items():
        lines.append(f"| {system} | {v['flagged_low']} | {v['flagged_not_low']} | "
                     f"{v['unflagged_low']} | {v['unflagged_not_low']} |")
    decisions = stats["llm_behaviour"]["ratings_by_escalation_decision"]
    lines += ["", "## LLM replies by escalation decision", ""]
    lines += [f"- {k}: {v['rated']} rated, claim safety ≤2 in {v['claim_safety_le_2']} "
              f"(share {fmt(v['share_claim_safety_le_2'])})" for k, v in decisions.items()]
    rt = stats["test_retest"]
    lines += ["", "## Test-retest", "", f"{rt['measurement']}. {rt['responses']} responses.", "",
              "| Dimension | Exact | Within 1 | Mean abs. diff | Weighted κ [95% CI] |",
              "| --- | --- | --- | --- | --- |"]
    for d, v in list(rt["per_dimension"].items()) + [("pooled", rt["pooled"])]:
        lines.append(f"| {d} | {v['exact_agreement']} | {v['within_one_agreement']} | "
                     f"{v['mean_absolute_difference']} | {fmt(v['weighted_kappa_quadratic'])} "
                     f"{fmt(v['kappa_bootstrap']['interval_95'])} |")
    lines += ["", f"_{rt['pooled']['note']}. {rt['note']}._", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--round1", type=Path, required=True)
    parser.add_argument("--retest", type=Path, required=True)
    parser.add_argument("--confirm-real-ratings", action="store_true")
    args = parser.parse_args(argv)
    if not args.confirm_real_ratings:
        print("Refusing to read rating files without --confirm-real-ratings.")
        return 2

    round1_raw = read_ratings(args.round1, allow_real=True)
    retest_raw = read_ratings(args.retest, allow_real=True)
    r01 = batch_builder.rebuild_batch()
    round1_blank = batch_builder.BLANK_R01.read_bytes()
    retest_blank = batch_builder.BLANK_RETEST.read_bytes()
    if round1_blank != r01["blank"]:
        raise RatingValidationError("round-1 blank file differs from the frozen rebuild")
    retest = batch_builder.build_retest(r01["batch"]["csv_rows"], r01["batch"]["key"])
    if retest_blank != batch_builder.render_retest_csv(retest["rows"]):
        raise RatingValidationError("retest blank file differs from the frozen rebuild")

    stats = analyse(round1_raw=round1_raw, round1_blank_raw=round1_blank, key=r01["batch"]["key"],
                    automated=load_automated_metadata(), retest_raw=retest_raw,
                    retest_blank_raw=retest_blank, retest_key=retest["key"], input_kind="real",
                    extra_hashes={"evaluation_rows": sha256_bytes(batch_builder.EVALUATION_ROWS.read_bytes())})
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    REPORT_MD.write_text(render_report(stats), encoding="utf-8")
    print(f"wrote {REPORT_JSON} and {REPORT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
