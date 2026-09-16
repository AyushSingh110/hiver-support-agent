"""Phase 6F: build the blinded human reply-quality rating batch.

Reads the frozen Phase 6E evaluation rows and the frozen retrieval corpus; never reads
gold labels, never calls a model, never reruns retrieval. The system mapping is not
written anywhere: it is rebuilt from the frozen inputs and the declared seed, and only
its SHA-256 is recorded in the manifest.

    python -m src.build_reply_rating_batch --round r01    # write blank batch + manifest
    python -m src.build_reply_rating_batch --retest       # validate round 1, build the retest
    python -m src.build_reply_rating_batch --verify       # rebuild and compare with disk

Round-1 validation and the retest never print, store or copy a score value or a note.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.generate_reply import build_prompt, format_evidence, sanitise_evidence_text
from src.llm import _cache_key
from src.taxonomy import ALL_LABELS

HUMAN_EVAL_DIR = config.PROJECT_ROOT / "human_eval"
EVALUATION_ROWS = config.CORPUS_DIR / "evaluation_rows.jsonl"
RETRIEVAL_CORPUS = config.CORPUS_DIR / "retrieval_corpus.parquet"
BLANK_R01 = HUMAN_EVAL_DIR / "reply_rating_r01_blank.csv"
MANIFEST = HUMAN_EVAL_DIR / "reply_rating_manifest.json"

SAMPLE_SEED = 46
RETEST_SEED = 47
SAMPLE_SIZE = 40
EXPECTED_POPULATION = 243

# Rows whose message or reply text was shown during development (fixed before sampling).
PRIOR_EXPOSURE_EXCLUSIONS = ("b01_0001", "b01_0019", "b01_0057", "b01_0074", "b02_0012")

# Strata use only the batch and the frozen escalation decision; never labels or quality.
DECLARED_ALLOCATION = {
    ("b01", "auto_handle"): 16,
    ("b01", "escalate"): 7,
    ("b02", "auto_handle"): 11,
    ("b02", "escalate"): 6,
}

SYSTEMS = ("llm_grounded", "baseline_a_echo", "baseline_b_template")
SCORE_COLUMNS = ["relevance", "helpfulness", "groundedness", "information_request", "claim_safety"]
EVIDENCE_COLUMNS = [f"evidence_{n}" for n in range(1, 6)]
CSV_COLUMNS = (["response_id", "item_id", "customer_message"] + EVIDENCE_COLUMNS
               + ["candidate_reply"] + SCORE_COLUMNS + ["notes"])

NOT_RATED_EMPTY_REPLY = "not_rated_empty_reply"
TO_BE_RATED = "to_be_rated"
EMPTY_REPLY_PLACEHOLDER = "[NOT RATED - there is no reply text. Leave every score field blank.]"
MAX_SHUFFLE_ATTEMPTS = 10_000

RETEST_PLAN = {
    "responses": 36,
    "per_system": 12,
    "seed": RETEST_SEED,
    "minimum_gap_hours": 72,
    "preferred_gap_days": 7,
    "status": "planned for Stage 3; not built",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8"))


# --------------------------------------------------------------------------- inputs

def restrict_row(row: dict) -> dict:
    """Keep only what sampling, display and the evidence check need. `gold` is never read."""
    tfidf = row["system"]["retrieval"]["tfidf"]
    return {
        "annotation_id": row["annotation_id"],
        "batch": row["batch"],
        "customer_message": row["customer_message"],
        "decision": row["system"]["escalation"]["decision"],
        "evidence": [{"rank": e["rank"], "conversation_id": e["conversation_id"]} for e in tfidf],
        "replies": {
            "llm_grounded": row["system"]["generation"]["reply"],
            "baseline_a_echo": row["baselines"]["baseline_a_echo"]["reply"],
            "baseline_b_template": row["baselines"]["baseline_b_template"]["reply"],
        },
        # Used only to confirm the rebuilt evidence equals the generator's prompt.
        "prompt_intent": row["system"]["classifier"]["predicted_intent"],
    }


def load_frozen_rows(path: Path = EVALUATION_ROWS) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [restrict_row(json.loads(line)) for line in handle if line.strip()]


def corpus_lookup(corpus: pd.DataFrame) -> dict[str, dict]:
    frame = corpus.set_index("conversation_id")
    return {cid: {"customer_message": frame.at[cid, "customer_message"],
                  "brand_reply": frame.at[cid, "brand_reply"]}
            for cid in frame.index}


def load_corpus_lookup(path: Path = RETRIEVAL_CORPUS) -> dict[str, dict]:
    return corpus_lookup(pd.read_parquet(path, columns=["conversation_id", "customer_message",
                                                        "brand_reply"]))


# --------------------------------------------------------------------------- sampling

def build_population(rows: list[dict], exclusions=PRIOR_EXPOSURE_EXCLUSIONS) -> list[dict]:
    ids = [r["annotation_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate annotation ids in the frozen rows")
    missing = set(exclusions) - set(ids)
    if missing:
        raise ValueError(f"exclusions not present in the frozen rows: {sorted(missing)}")
    return [r for r in rows if r["annotation_id"] not in set(exclusions)]


def stratum(row: dict) -> tuple[str, str]:
    return row["batch"], row["decision"]


def largest_remainder_allocation(population: list[dict], total: int) -> dict[tuple, int]:
    counts = Counter(stratum(r) for r in population)
    quotas = {k: total * v / len(population) for k, v in counts.items()}
    allocation = {k: math.floor(q) for k, q in quotas.items()}
    remaining = total - sum(allocation.values())
    for key in sorted(quotas, key=lambda k: (-(quotas[k] - allocation[k]), k))[:remaining]:
        allocation[key] += 1
    return dict(sorted(allocation.items()))


def sample_items(population: list[dict], allocation: dict[tuple, int],
                 rng: np.random.Generator) -> list[dict]:
    chosen = []
    for key in sorted(allocation):
        pool = sorted((r for r in population if stratum(r) == key), key=lambda r: r["annotation_id"])
        if len(pool) < allocation[key]:
            raise ValueError(f"stratum {key} has {len(pool)} rows, needs {allocation[key]}")
        positions = rng.choice(len(pool), size=allocation[key], replace=False)
        chosen.extend(pool[int(p)] for p in sorted(positions))
    return chosen


# --------------------------------------------------------------------------- blinding

def rebuild_evidence(row: dict, lookup: dict[str, dict]) -> list[dict]:
    items = []
    for e in row["evidence"]:
        if e["conversation_id"] not in lookup:
            raise ValueError(f"{row['annotation_id']}: evidence {e['conversation_id']} not in corpus")
        source = lookup[e["conversation_id"]]
        items.append({"rank": e["rank"],
                      "historical_customer_message": source["customer_message"],
                      "historical_brand_reply": source["brand_reply"]})
    return items


def shuffle_without_adjacent_items(entries: list[dict], rng: np.random.Generator) -> tuple[list[dict], int]:
    """Draw permutations from the seeded generator until no two neighbours share an item."""
    for attempt in range(1, MAX_SHUFFLE_ATTEMPTS + 1):
        order = [entries[int(i)] for i in rng.permutation(len(entries))]
        if all(a["item_id"] != b["item_id"] for a, b in zip(order, order[1:])):
            return order, attempt
    raise RuntimeError("could not separate responses of the same item")


def build_batch(rows: list[dict], lookup: dict[str, dict], *,
                exclusions=PRIOR_EXPOSURE_EXCLUSIONS, sample_size: int = SAMPLE_SIZE,
                seed: int = SAMPLE_SEED, declared_allocation: dict | None = None) -> dict:
    """Pure: returns the annotation rows, the hidden key and the manifest core."""
    population = build_population(rows, exclusions)
    allocation = largest_remainder_allocation(population, sample_size)
    if declared_allocation is not None and allocation != dict(sorted(declared_allocation.items())):
        raise ValueError(f"allocation {allocation} differs from the declared {declared_allocation}")

    rng = np.random.default_rng(seed)
    sampled = sample_items(population, allocation, rng)

    # Item ids are assigned in a shuffled order so they carry no batch or stratum order.
    item_order = rng.permutation(len(sampled))
    items = [sampled[int(i)] for i in item_order]

    entries = []
    for number, row in enumerate(items, start=1):
        item_id = f"I{number:02d}"
        evidence = [format_evidence([item]) for item in rebuild_evidence(row, lookup)]
        message = sanitise_evidence_text(row["customer_message"])
        for system in SYSTEMS:
            reply = row["replies"][system]
            empty = not reply.strip()
            entries.append({
                "item_id": item_id,
                "annotation_id": row["annotation_id"],
                "system": system,
                "status": NOT_RATED_EMPTY_REPLY if empty else TO_BE_RATED,
                "customer_message": message,
                "evidence": evidence,
                "candidate_reply": EMPTY_REPLY_PLACEHOLDER if empty else reply,
            })

    ordered, attempts = shuffle_without_adjacent_items(entries, rng)
    csv_rows, key = [], []
    for number, entry in enumerate(ordered, start=1):
        response_id = f"R{number:03d}"
        csv_rows.append({
            "response_id": response_id,
            "item_id": entry["item_id"],
            "customer_message": entry["customer_message"],
            **dict(zip(EVIDENCE_COLUMNS, entry["evidence"])),
            "candidate_reply": entry["candidate_reply"],
            **{column: "" for column in SCORE_COLUMNS},
            "notes": "",
        })
        key.append({"response_id": response_id, "item_id": entry["item_id"],
                    "annotation_id": entry["annotation_id"], "system": entry["system"],
                    "status": entry["status"]})

    population_counts = Counter(stratum(r) for r in population)
    sampled_ids = sorted(r["annotation_id"] for r in sampled)
    core = {
        "population": {
            "source_rows": len(rows),
            "excluded_prior_exposure": sorted(exclusions),
            "population_rows": len(population),
            "strata_population": {f"{b}/{d}": n for (b, d), n in sorted(population_counts.items())},
        },
        "sampling": {
            "method": "proportional stratified random sampling by batch x escalation decision, "
                      "largest-remainder allocation; numpy default_rng, choice without replacement "
                      "over each stratum sorted by annotation_id",
            "seed": seed,
            "sample_size": len(sampled),
            "allocation": {f"{b}/{d}": n for (b, d), n in allocation.items()},
            "selection_signals": ["batch", "escalation decision"],
            "sampled_annotation_ids": sampled_ids,
            "selection_sha256": sha256_json(sampled_ids),
        },
        "blinding": {
            "items": len(items),
            "responses": len(csv_rows),
            "systems_per_item": len(SYSTEMS),
            "item_ids": "I01-I40, assigned after shuffling the sampled rows",
            "response_ids": "R001-R120, assigned after shuffling all responses",
            "no_adjacent_same_item": True,
            "shuffle_attempts": attempts,
            "mapping_written_to_disk": False,
            "mapping_rebuild": "rebuild_batch() from the frozen inputs and the declared seed",
            "mapping_sha256": sha256_json(key),
            "responses_not_rated_empty_reply": sum(k["status"] == NOT_RATED_EMPTY_REPLY for k in key),
        },
    }
    return {"csv_rows": csv_rows, "key": key, "manifest_core": core, "sampled": sampled,
            "items": items}


# --------------------------------------------------------------------------- evidence check

def verify_evidence_against_generation(items: list[dict], lookup: dict[str, dict],
                                       cache_dir: Path | None = None) -> dict:
    """Rebuild each generation prompt from the rebuilt evidence and confirm a production
    cache entry exists for it. Read-only: the cache is never written."""
    cache_dir = cache_dir or config.CACHE_DIR
    matched, missing = 0, []
    for row in items:
        prompt = build_prompt(row["customer_message"], row["prompt_intent"],
                              rebuild_evidence(row, lookup))
        key = _cache_key(prompt, config.GENERATION_MODEL, 0.0, config.RANDOM_SEED, True)
        if (cache_dir / f"{key}.json").exists():
            matched += 1
        else:
            missing.append(row["annotation_id"])
    return {"items_checked": len(items), "generation_prompts_found_in_cache": matched,
            "missing": missing}


# --------------------------------------------------------------------------- outputs

def render_csv(csv_rows: list[dict]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_ALL, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(csv_rows)
    return buffer.getvalue().encode("utf-8-sig")


def build_manifest(batch: dict, blank_csv: bytes, input_hashes: dict, evidence_check: dict) -> dict:
    return {
        "phase": "6F human reply-quality evaluation",
        "round": "r01",
        "status": "blank batch built; no ratings exist",
        "gold_labels_read": False,
        "inputs_sha256": input_hashes,
        **batch["manifest_core"],
        "evidence": {
            "source": "frozen retrieved conversation ids in evaluation_rows.jsonl, text from the "
                      "frozen retrieval corpus, formatted with generate_reply.format_evidence",
            "retrieval_rerun": False,
            "check": evidence_check,
        },
        "outputs": {
            "blank_csv": BLANK_R01.name,
            "blank_csv_sha256": sha256_bytes(blank_csv),
            "columns": CSV_COLUMNS,
            "score_columns": SCORE_COLUMNS,
        },
        "retest_plan": RETEST_PLAN,
    }


def rebuild_batch() -> dict:
    rows = load_frozen_rows()
    lookup = load_corpus_lookup()
    batch = build_batch(rows, lookup, declared_allocation=DECLARED_ALLOCATION)
    if batch["manifest_core"]["population"]["population_rows"] != EXPECTED_POPULATION:
        raise ValueError("population is not the approved 243 rows")
    blank = render_csv(batch["csv_rows"])
    input_hashes = {"evaluation_rows.jsonl": sha256_bytes(EVALUATION_ROWS.read_bytes()),
                    "retrieval_corpus.parquet": sha256_bytes(RETRIEVAL_CORPUS.read_bytes())}
    evidence_check = verify_evidence_against_generation(batch["items"], lookup)
    manifest = build_manifest(batch, blank, input_hashes, evidence_check)
    return {"batch": batch, "blank": blank, "manifest": manifest}


def manifest_bytes(manifest: dict) -> bytes:
    return (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def write_outputs(result: dict, directory: Path = HUMAN_EVAL_DIR) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    blank_path = directory / BLANK_R01.name
    if blank_path.exists() and blank_path.read_bytes() != result["blank"]:
        raise FileExistsError(f"{blank_path} exists with different content; refusing to overwrite")
    blank_path.write_bytes(result["blank"])
    (directory / MANIFEST.name).write_bytes(manifest_bytes(result["manifest"]))


def verify_on_disk(result: dict, directory: Path = HUMAN_EVAL_DIR) -> dict:
    blank = (directory / BLANK_R01.name).read_bytes()
    manifest = (directory / MANIFEST.name).read_bytes()
    return {"blank_csv_identical": blank == result["blank"],
            "manifest_identical": manifest == manifest_bytes(result["manifest"]),
            "mapping_sha256": result["manifest"]["blinding"]["mapping_sha256"]}


# --------------------------------------------------------------------------- round 1 and retest
# Validation reports only structural facts and positions, never score values or notes.

RATED_R01 = HUMAN_EVAL_DIR / "reply_rating_r01_rated.csv"
BLANK_RETEST = HUMAN_EVAL_DIR / "reply_rating_retest_blank.csv"
RETEST_MANIFEST = HUMAN_EVAL_DIR / "reply_rating_retest_manifest.json"

RETEST_SIZE = 36
RETEST_PER_SYSTEM = 12
MINIMUM_GAP_HOURS = 72
PREFERRED_GAP_DAYS = 7
RETEST_COLUMNS = (["retest_id", "customer_message"] + EVIDENCE_COLUMNS + ["candidate_reply"]
                  + SCORE_COLUMNS + ["notes"])
TEXT_COLUMNS = ["item_id", "customer_message"] + EVIDENCE_COLUMNS + ["candidate_reply"]
VALID_SCORES = {"1", "2", "3", "4", "5"}
FORBIDDEN_TOKENS = ("llm_grounded", "baseline_a_echo", "baseline_b_template", "auto_handle",
                    NOT_RATED_EMPTY_REPLY, TO_BE_RATED) + tuple(
    label for label in ALL_LABELS if "_" in label)
ANNOTATION_ID_PATTERN = re.compile(r"\bb0[12]_\d{4}\b|\bconv_\d+")


def parse_csv(raw: bytes) -> tuple[list[str], list[dict]]:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("file is not UTF-8") from error
    reader = csv.DictReader(io.StringIO(text, newline=""))
    rows = list(reader)
    return list(reader.fieldnames or []), rows


def validate_round1(rated_raw: bytes, blank_raw: bytes, key: list[dict]) -> dict:
    """Integrity of a returned round-1 file against the frozen blank batch and hidden key."""
    header, rated = parse_csv(rated_raw)
    _, blank = parse_csv(blank_raw)
    blank_by_id = {r["response_id"]: r for r in blank}
    rated_ids = [r.get("response_id") for r in rated]
    rated_by_id = {r.get("response_id"): r for r in rated}
    unrated_expected = sorted(k["response_id"] for k in key if k["status"] == NOT_RATED_EMPTY_REPLY)
    ratable_expected = sorted(k["response_id"] for k in key if k["status"] == TO_BE_RATED)

    text_mismatches = sorted(
        (rid, column) for rid, row in rated_by_id.items() if rid in blank_by_id
        for column in TEXT_COLUMNS if row.get(column) != blank_by_id[rid][column])
    invalid_score_cells = sorted(
        (rid, column) for rid, row in rated_by_id.items()
        for column in SCORE_COLUMNS if row.get(column, "") not in VALID_SCORES | {""})
    incomplete_ratable = sorted(
        rid for rid in ratable_expected
        if rid not in rated_by_id or any(rated_by_id[rid].get(c) not in VALID_SCORES for c in SCORE_COLUMNS))
    rated_empty = sorted(
        rid for rid in unrated_expected
        if rid not in rated_by_id or any(rated_by_id[rid].get(c, "") != "" for c in SCORE_COLUMNS))
    forbidden = sorted(
        (row.get("response_id"), column, token)
        for row in rated for column, value in row.items() if isinstance(value, str)
        for token in FORBIDDEN_TOKENS if token in value)
    id_leaks = sorted(
        (row.get("response_id"), column)
        for row in rated for column, value in row.items()
        if isinstance(value, str) and ANNOTATION_ID_PATTERN.search(value))

    facts = {
        "columns_exact": header == CSV_COLUMNS,
        "row_count": len(rated),
        "expected_row_count": len(blank),
        "response_ids_unique": len(set(rated_ids)) == len(rated_ids),
        "response_ids_match_blank_in_order": rated_ids == [r["response_id"] for r in blank],
        "item_ids_expected": len({r["item_id"] for r in blank}),
        "item_ids_found": len({r.get("item_id") for r in rated}),
        "text_unchanged": not text_mismatches,
        "text_mismatches": text_mismatches,
        "scores_are_blank_or_integers_1_to_5": not invalid_score_cells,
        "invalid_score_cells": invalid_score_cells,
        "ratable_expected": len(ratable_expected),
        "ratable_complete": len(ratable_expected) - len(incomplete_ratable),
        "ratable_incomplete": incomplete_ratable,
        "unrated_expected": unrated_expected,
        "unrated_rows_left_blank": not rated_empty,
        "unrated_notes_blank": all(rated_by_id.get(rid, {}).get("notes", "") == "" for rid in unrated_expected),
        "prohibited_tokens": forbidden,
        "identifier_patterns": id_leaks,
    }
    facts["passed"] = all((
        facts["columns_exact"], facts["row_count"] == facts["expected_row_count"],
        facts["response_ids_unique"], facts["response_ids_match_blank_in_order"],
        facts["item_ids_found"] == facts["item_ids_expected"], facts["text_unchanged"],
        facts["scores_are_blank_or_integers_1_to_5"], not incomplete_ratable,
        facts["unrated_rows_left_blank"], not forbidden, not id_leaks,
    ))
    return facts


def retest_timing(rated_mtime: float, blank_mtime: float, now: float) -> dict:
    """Round-1 completion is taken as the rated file's last-modified time."""
    if rated_mtime < blank_mtime:
        raise ValueError("rated file is older than the blank batch; completion time cannot be established")
    if rated_mtime > now:
        raise ValueError("rated file is modified in the future; completion time cannot be established")

    def stamp(seconds: float) -> str:
        return datetime.fromtimestamp(seconds, tz=timezone.utc).astimezone().isoformat(timespec="seconds")

    return {
        "round1_completed_at": stamp(rated_mtime),
        "round1_completed_at_basis": "last-modified time of reply_rating_r01_rated.csv",
        "minimum_gap_hours": MINIMUM_GAP_HOURS,
        "preferred_gap_days": PREFERRED_GAP_DAYS,
        "earliest_retest_start": stamp(rated_mtime + MINIMUM_GAP_HOURS * 3600),
        "preferred_retest_start": stamp(rated_mtime + PREFERRED_GAP_DAYS * 86400),
    }


def select_retest(key: list[dict], rng: np.random.Generator) -> list[str]:
    """12 rated responses per system. Uses only the hidden key, never score values."""
    chosen = []
    for system in SYSTEMS:
        pool = sorted(k["response_id"] for k in key
                      if k["system"] == system and k["status"] == TO_BE_RATED)
        if len(pool) < RETEST_PER_SYSTEM:
            raise ValueError(f"only {len(pool)} rated responses for one system")
        picks = rng.choice(len(pool), size=RETEST_PER_SYSTEM, replace=False)
        chosen.extend(pool[int(p)] for p in sorted(picks))
    return chosen


def build_retest(blank_rows: list[dict], key: list[dict], seed: int = RETEST_SEED) -> dict:
    """Retest rows come from the frozen blank batch, so no round-1 value can reach them."""
    rng = np.random.default_rng(seed)
    selected = select_retest(key, rng)
    by_id = {r["response_id"]: r for r in blank_rows}
    ordered, attempts = shuffle_without_adjacent_items([dict(by_id[rid]) for rid in selected], rng)

    rows, retest_key = [], []
    system_of = {k["response_id"]: k["system"] for k in key}
    for number, source in enumerate(ordered, start=1):
        retest_id = f"T{number:02d}"
        rows.append({
            "retest_id": retest_id,
            "customer_message": source["customer_message"],
            **{c: source[c] for c in EVIDENCE_COLUMNS},
            "candidate_reply": source["candidate_reply"],
            **{c: "" for c in SCORE_COLUMNS},
            "notes": "",
        })
        retest_key.append({"retest_id": retest_id, "response_id": source["response_id"],
                           "item_id": source["item_id"], "system": system_of[source["response_id"]]})
    return {"rows": rows, "key": retest_key, "selected": selected, "shuffle_attempts": attempts}


def render_retest_csv(rows: list[dict]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=RETEST_COLUMNS, quoting=csv.QUOTE_ALL, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")


def build_retest_manifest(rated_raw: bytes, validation: dict, timing: dict, retest: dict,
                          blank_retest: bytes, blank_r01_sha256: str) -> dict:
    items = [k["item_id"] for k in retest["key"]]
    return {
        "phase": "6F human reply-quality evaluation",
        "round": "retest",
        "status": "round 1 received and validated; blank retest batch built; no retest ratings exist",
        "gold_labels_read": False,
        "round1": {
            "file": RATED_R01.name,
            "sha256": sha256_bytes(rated_raw),
            "blank_sha256": blank_r01_sha256,
            "validation": validation,
            "ratings_included_here": False,
        },
        "timing": timing,
        "retest": {
            "seed": RETEST_SEED,
            "responses": len(retest["rows"]),
            "systems": len(SYSTEMS),
            "responses_per_system": RETEST_PER_SYSTEM,
            "drawn_from": "round-1 responses marked to_be_rated in the hidden key",
            "retest_ids": "T01-T36, assigned after shuffling",
            "no_adjacent_same_item": all(a != b for a, b in zip(items, items[1:])),
            "shuffle_attempts": retest["shuffle_attempts"],
            "selection_sha256": sha256_json(retest["selected"]),
            "mapping_sha256": sha256_json(retest["key"]),
            "mapping_written_to_disk": False,
            "round1_ratings_included": False,
        },
        "outputs": {
            "blank_csv": BLANK_RETEST.name,
            "blank_csv_sha256": sha256_bytes(blank_retest),
            "columns": RETEST_COLUMNS,
        },
    }


def rebuild_retest(r01: dict) -> dict:
    """Validate round 1, fix the timing, and build the retest. Reads the rated file read-only."""
    blank_raw = BLANK_R01.read_bytes()
    if blank_raw != r01["blank"]:
        raise ValueError("round-1 blank file on disk differs from the frozen rebuild")
    rated_raw = RATED_R01.read_bytes()
    validation = validate_round1(rated_raw, blank_raw, r01["batch"]["key"])
    timing = retest_timing(RATED_R01.stat().st_mtime, BLANK_R01.stat().st_mtime, time.time())
    retest = build_retest(r01["batch"]["csv_rows"], r01["batch"]["key"])
    blank_retest = render_retest_csv(retest["rows"])
    manifest = build_retest_manifest(rated_raw, validation, timing, retest, blank_retest,
                                     sha256_bytes(blank_raw))
    if RATED_R01.read_bytes() != rated_raw:
        raise RuntimeError("round-1 rated file changed during validation")
    return {"validation": validation, "timing": timing, "retest": retest,
            "blank": blank_retest, "manifest": manifest}


def write_retest_outputs(result: dict, directory: Path = HUMAN_EVAL_DIR) -> None:
    if not result["validation"]["passed"]:
        raise ValueError("round 1 did not pass validation; retest batch not built")
    blank_path = directory / BLANK_RETEST.name
    if blank_path.exists() and blank_path.read_bytes() != result["blank"]:
        raise FileExistsError(f"{blank_path} exists with different content; refusing to overwrite")
    blank_path.write_bytes(result["blank"])
    (directory / RETEST_MANIFEST.name).write_bytes(manifest_bytes(result["manifest"]))


def verify_retest_on_disk(result: dict, directory: Path = HUMAN_EVAL_DIR) -> dict:
    return {"retest_blank_identical": (directory / BLANK_RETEST.name).read_bytes() == result["blank"],
            "retest_manifest_identical": (directory / RETEST_MANIFEST.name).read_bytes()
                                         == manifest_bytes(result["manifest"]),
            "retest_mapping_sha256": result["manifest"]["retest"]["mapping_sha256"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--round", choices=["r01"])
    group.add_argument("--retest", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)

    result = rebuild_batch()
    check = result["manifest"]["evidence"]["check"]
    if check["generation_prompts_found_in_cache"] != check["items_checked"]:
        raise RuntimeError(f"rebuilt evidence does not match the generation prompts: {check['missing']}")

    if args.verify:
        report = verify_on_disk(result)
        ok = report["blank_csv_identical"] and report["manifest_identical"]
        if (HUMAN_EVAL_DIR / BLANK_RETEST.name).exists():
            report.update(verify_retest_on_disk(rebuild_retest(result)))
            ok = ok and report["retest_blank_identical"] and report["retest_manifest_identical"]
        print(json.dumps(report, indent=2))
        return 0 if ok else 1

    if args.retest:
        retest = rebuild_retest(result)
        # Validation facts are structural only: no score value or note text is in them.
        print(json.dumps({"round1_validation": retest["validation"],
                          "timing": retest["timing"]}, indent=2))
        if not retest["validation"]["passed"]:
            print("round 1 failed validation; retest batch NOT built")
            return 1
        write_retest_outputs(retest)
        print(f"retest rows {len(retest['retest']['rows'])}; wrote {BLANK_RETEST} and {RETEST_MANIFEST}")
        return 0

    write_outputs(result)
    core = result["manifest"]
    print(f"population {core['population']['population_rows']}, sampled {core['sampling']['sample_size']}, "
          f"responses {core['blinding']['responses']}, "
          f"not rated (empty) {core['blinding']['responses_not_rated_empty_reply']}")
    print(f"allocation {core['sampling']['allocation']}")
    print(f"evidence check {check}")
    print(f"wrote {BLANK_R01} and {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
