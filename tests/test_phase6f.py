"""Phase 6F rating-batch tests. Synthetic fixtures only: no gold labels, no model calls."""
from __future__ import annotations

import ast
import csv
import io
import json
import re
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from src import build_reply_rating_batch as b
from src import config, llm
from src.generate_reply import format_evidence

SECRETS = ["SECRET_GOLD", "PREDICTED_SECRET", "WEAK_SECRET", "G7_SECRET", "REASON_SECRET",
           "0.7777", "ev_b0", "conv_b0", "llm_grounded", "baseline_a_echo", "baseline_b_template",
           "auto_handle", "escalate", "not_rated_empty_reply", "to_be_rated"]

# Mirrors the frozen strata: b01_0057 and b02_0012 escalated; b01_0001/0019/0074 auto-handled.
B01_ESCALATED = {"b01_0057"} | {f"b01_{n:04d}" for n in range(104, 149)}
B02_ESCALATED = {f"b02_{n:04d}" for n in range(1, 35)}         # includes b02_0012


def frozen_row(annotation_id: str, number: int, decision: str, reply: str | None = None) -> dict:
    return {
        "annotation_id": annotation_id,
        "batch": annotation_id[:3],
        "conversation_id": f"conv_{annotation_id}",
        "customer_message": f"@12345 synthetic customer message number {number}",
        "gold": {"primary_intent": f"SECRET_GOLD_{number}", "secondary_intent": None,
                 "ambiguous": "no", "predictable_by_classifier": True},
        "system": {
            "classifier": {"predicted_intent": "PREDICTED_SECRET", "confidence": 0.7777},
            "retrieval": {"tfidf": [{"rank": i + 1, "conversation_id": f"ev_{annotation_id}_{i}",
                                     "similarity": 0.5, "weak_intent": "WEAK_SECRET"} for i in range(5)],
                          "random": []},
            "generation": {"reply": f"Synthetic system reply {number}." if reply is None else reply,
                           "grounding_flag_codes": ["G7_SECRET"], "parsed": True},
            "escalation": {"decision": decision, "reason_codes": ["REASON_SECRET"]},
        },
        "baselines": {
            "baseline_a_echo": {"reply": f"Historical echo {number}.", "grounding_flag_codes": ["G7_SECRET"]},
            "baseline_b_template": {"reply": "One fixed generic reply.", "grounding_flag_codes": []},
        },
    }


def synthetic_rows(empty: set[str] = frozenset()) -> list[dict]:
    rows, number = [], 0
    for batch, size, escalated in (("b01", 148, B01_ESCALATED), ("b02", 100, B02_ESCALATED)):
        for n in range(1, size + 1):
            aid = f"{batch}_{n:04d}"
            number += 1
            rows.append(frozen_row(aid, number, "escalate" if aid in escalated else "auto_handle",
                                   reply="" if aid in empty else None))
    return rows


def synthetic_lookup(rows: list[dict]) -> dict:
    lookup = {}
    for row in rows:
        for e in row["system"]["retrieval"]["tfidf"]:
            lookup[e["conversation_id"]] = {
                "customer_message": f"@999 historical message {e['conversation_id'][-6:]}",
                "brand_reply": f"@888 historical team reply {e['rank']} DM us please",
            }
    return lookup


def build(rows=None, **kwargs):
    rows = rows or synthetic_rows()
    restricted = [b.restrict_row(r) for r in rows]
    return b.build_batch(restricted, synthetic_lookup(rows),
                         declared_allocation=b.DECLARED_ALLOCATION, **kwargs)


def rendered_text(batch) -> str:
    return b.render_csv(batch["csv_rows"]).decode("utf-8-sig")


class TestPopulationAndSampling(unittest.TestCase):
    def test_synthetic_population_mirrors_the_frozen_strata(self):
        population = b.build_population([b.restrict_row(r) for r in synthetic_rows()])
        self.assertEqual(len(population), b.EXPECTED_POPULATION)
        counts = Counter(b.stratum(r) for r in population)
        self.assertEqual(counts, {("b01", "auto_handle"): 99, ("b01", "escalate"): 45,
                                  ("b02", "auto_handle"): 66, ("b02", "escalate"): 33})

    def test_largest_remainder_gives_the_declared_allocation(self):
        population = b.build_population([b.restrict_row(r) for r in synthetic_rows()])
        self.assertEqual(b.largest_remainder_allocation(population, 40),
                         dict(sorted(b.DECLARED_ALLOCATION.items())))
        self.assertEqual(sum(b.DECLARED_ALLOCATION.values()), 40)

    def test_exactly_forty_items_with_correct_strata(self):
        batch = build()
        self.assertEqual(len(batch["sampled"]), 40)
        self.assertEqual(Counter(b.stratum(r) for r in batch["sampled"]), b.DECLARED_ALLOCATION)
        self.assertEqual(batch["manifest_core"]["sampling"]["allocation"],
                         {"b01/auto_handle": 16, "b01/escalate": 7, "b02/auto_handle": 11, "b02/escalate": 6})

    def test_prior_exposure_rows_are_excluded(self):
        self.assertEqual(b.PRIOR_EXPOSURE_EXCLUSIONS,
                         ("b01_0001", "b01_0019", "b01_0057", "b01_0074", "b02_0012"))
        batch = build()
        sampled = {r["annotation_id"] for r in batch["sampled"]}
        self.assertFalse(sampled & set(b.PRIOR_EXPOSURE_EXCLUSIONS))
        self.assertEqual(batch["manifest_core"]["population"]["excluded_prior_exposure"],
                         sorted(b.PRIOR_EXPOSURE_EXCLUSIONS))
        self.assertFalse({k["annotation_id"] for k in batch["key"]} & set(b.PRIOR_EXPOSURE_EXCLUSIONS))

    def test_no_unexpected_population_rows(self):
        rows = synthetic_rows()
        batch = build(rows)
        allowed = {r["annotation_id"] for r in rows} - set(b.PRIOR_EXPOSURE_EXCLUSIONS)
        self.assertTrue({r["annotation_id"] for r in batch["sampled"]} <= allowed)
        self.assertEqual(batch["manifest_core"]["population"]["population_rows"], 243)
        self.assertEqual(batch["manifest_core"]["population"]["source_rows"], 248)

    def test_missing_exclusion_or_duplicate_rows_raise(self):
        rows = [b.restrict_row(r) for r in synthetic_rows()]
        with self.assertRaises(ValueError):
            b.build_population(rows, exclusions=("b09_9999",))
        with self.assertRaises(ValueError):
            b.build_population(rows + rows[:1])

    def test_declared_allocation_mismatch_raises(self):
        rows = synthetic_rows()
        with self.assertRaises(ValueError):
            b.build_batch([b.restrict_row(r) for r in rows], synthetic_lookup(rows),
                          declared_allocation={("b01", "auto_handle"): 40})

    def test_selection_uses_only_batch_and_decision(self):
        rows = synthetic_rows()
        altered = json.loads(json.dumps(rows))
        for row in altered:
            row["gold"]["primary_intent"] = "OTHER_GOLD"
            row["system"]["classifier"] = {"predicted_intent": "ELSE", "confidence": 0.01}
            row["system"]["generation"]["grounding_flag_codes"] = []
            row["system"]["retrieval"]["tfidf"][0]["similarity"] = 0.99
        self.assertEqual([r["annotation_id"] for r in build(rows)["sampled"]],
                         [r["annotation_id"] for r in build(altered)["sampled"]])


class TestDeterminism(unittest.TestCase):
    def test_rerun_is_identical(self):
        first, second = build(), build()
        self.assertEqual(first["key"], second["key"])
        self.assertEqual(b.render_csv(first["csv_rows"]), b.render_csv(second["csv_rows"]))
        self.assertEqual(first["manifest_core"], second["manifest_core"])

    def test_seed_changes_the_sample(self):
        self.assertNotEqual(build()["manifest_core"]["sampling"]["selection_sha256"],
                            build(seed=45)["manifest_core"]["sampling"]["selection_sha256"])

    def test_declared_seeds(self):
        self.assertEqual((b.SAMPLE_SEED, b.RETEST_SEED, b.SAMPLE_SIZE), (46, 47, 40))


class TestBlindedStructure(unittest.TestCase):
    batch = build()

    def test_exactly_120_rows_with_unique_ordered_ids(self):
        rows = self.batch["csv_rows"]
        self.assertEqual(len(rows), 120)
        self.assertEqual([r["response_id"] for r in rows], [f"R{n:03d}" for n in range(1, 121)])

    def test_forty_items_three_responses_each(self):
        counts = Counter(r["item_id"] for r in self.batch["csv_rows"])
        self.assertEqual(set(counts), {f"I{n:02d}" for n in range(1, 41)})
        self.assertEqual(set(counts.values()), {3})

    def test_each_item_has_one_response_per_system(self):
        per_item = {}
        for k in self.batch["key"]:
            per_item.setdefault(k["item_id"], []).append(k["system"])
        self.assertTrue(all(sorted(v) == sorted(b.SYSTEMS) for v in per_item.values()))
        by_item = {k["item_id"]: k["annotation_id"] for k in self.batch["key"]}
        self.assertEqual(len(set(by_item.values())), 40)

    def test_no_adjacent_responses_from_the_same_item(self):
        items = [r["item_id"] for r in self.batch["csv_rows"]]
        self.assertTrue(all(a != c for a, c in zip(items, items[1:])))
        self.assertTrue(self.batch["manifest_core"]["blinding"]["no_adjacent_same_item"])

    def test_item_ids_do_not_follow_batch_order(self):
        order = [k["annotation_id"] for k in sorted(self.batch["key"], key=lambda k: k["item_id"])][::3]
        self.assertNotEqual(order, sorted(order))

    def test_columns_and_blank_scores(self):
        self.assertEqual(b.CSV_COLUMNS, ["response_id", "item_id", "customer_message",
                                         "evidence_1", "evidence_2", "evidence_3", "evidence_4",
                                         "evidence_5", "candidate_reply", "relevance", "helpfulness",
                                         "groundedness", "information_request", "claim_safety", "notes"])
        for row in self.batch["csv_rows"]:
            self.assertEqual(list(row), b.CSV_COLUMNS)
            self.assertTrue(all(row[c] == "" for c in b.SCORE_COLUMNS + ["notes"]))

    def test_no_gold_system_or_metadata_in_the_csv(self):
        text = rendered_text(self.batch)
        for secret in SECRETS:
            self.assertNotIn(secret, text, secret)
        self.assertIsNone(re.search(r"b0[12]_\d{4}", text))
        self.assertNotRegex(text, r"@\d+")

    def test_rendered_csv_round_trips(self):
        raw = b.render_csv(self.batch["csv_rows"])
        self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))
        parsed = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        self.assertEqual(parsed, self.batch["csv_rows"])

    def test_three_candidate_replies_per_item_are_the_frozen_ones(self):
        rows = {r["annotation_id"]: r for r in synthetic_rows()}
        by_response = {r["response_id"]: r for r in self.batch["csv_rows"]}
        for k in self.batch["key"]:
            frozen = rows[k["annotation_id"]]
            expected = (frozen["system"]["generation"]["reply"] if k["system"] == "llm_grounded"
                        else frozen["baselines"][k["system"]]["reply"])
            self.assertEqual(by_response[k["response_id"]]["candidate_reply"], expected)


class TestEvidence(unittest.TestCase):
    def test_evidence_is_rebuilt_from_frozen_ids_and_formatted_as_for_the_generator(self):
        rows = synthetic_rows()
        batch = build(rows)
        lookup = synthetic_lookup(rows)
        frozen = {r["annotation_id"]: r for r in rows}
        by_response = {r["response_id"]: r for r in batch["csv_rows"]}
        for k in batch["key"]:
            shown = by_response[k["response_id"]]
            for e in frozen[k["annotation_id"]]["system"]["retrieval"]["tfidf"]:
                source = lookup[e["conversation_id"]]
                expected = format_evidence([{"rank": e["rank"],
                                             "historical_customer_message": source["customer_message"],
                                             "historical_brand_reply": source["brand_reply"]}])
                self.assertEqual(shown[f"evidence_{e['rank']}"], expected)
            self.assertTrue(shown["evidence_1"].startswith("[1] customer: historical message"))
            self.assertEqual(shown["customer_message"],
                             frozen[k["annotation_id"]]["customer_message"].replace("@12345 ", ""))

    def test_missing_corpus_evidence_raises(self):
        rows = synthetic_rows()
        lookup = synthetic_lookup(rows)
        sampled = build(rows)["sampled"][0]
        lookup.pop(sampled["evidence"][2]["conversation_id"])
        with self.assertRaises(ValueError):
            b.build_batch([b.restrict_row(r) for r in rows], lookup,
                          declared_allocation=b.DECLARED_ALLOCATION)

    def test_evidence_check_against_a_cache_directory(self):
        rows = synthetic_rows()
        batch = build(rows)
        lookup = synthetic_lookup(rows)
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            first = batch["items"][0]
            prompt = b.build_prompt(first["customer_message"], first["prompt_intent"],
                                    b.rebuild_evidence(first, lookup))
            key = llm._cache_key(prompt, config.GENERATION_MODEL, 0.0, config.RANDOM_SEED, True)
            (cache / f"{key}.json").write_text("{}", encoding="utf-8")
            before = sorted(p.name for p in cache.iterdir())
            check = b.verify_evidence_against_generation(batch["items"], lookup, cache)
            self.assertEqual(check["items_checked"], 40)
            self.assertEqual(check["generation_prompts_found_in_cache"], 1)
            self.assertEqual(len(check["missing"]), 39)
            self.assertEqual(sorted(p.name for p in cache.iterdir()), before)


class TestEmptyReplies(unittest.TestCase):
    def test_empty_reply_is_marked_and_not_substituted(self):
        normal = build()
        target = next(k["annotation_id"] for k in normal["key"] if k["system"] == "llm_grounded")
        with_empty = build(synthetic_rows(empty={target}))
        self.assertEqual([r["annotation_id"] for r in normal["sampled"]],
                         [r["annotation_id"] for r in with_empty["sampled"]])
        marked = [k for k in with_empty["key"] if k["status"] == b.NOT_RATED_EMPTY_REPLY]
        self.assertEqual([(k["annotation_id"], k["system"]) for k in marked], [(target, "llm_grounded")])
        row = next(r for r in with_empty["csv_rows"] if r["response_id"] == marked[0]["response_id"])
        self.assertEqual(row["candidate_reply"], b.EMPTY_REPLY_PLACEHOLDER)
        self.assertTrue(all(row[c] == "" for c in b.SCORE_COLUMNS))
        self.assertEqual(with_empty["manifest_core"]["blinding"]["responses_not_rated_empty_reply"], 1)
        self.assertEqual(len(with_empty["csv_rows"]), 120)

    def test_placeholder_reveals_no_system_or_reason(self):
        lowered = b.EMPTY_REPLY_PLACEHOLDER.lower()
        for word in ("llm", "system", "parse", "baseline", "fail", "error", "model"):
            self.assertNotIn(word, lowered)


class TestManifest(unittest.TestCase):
    def setUp(self):
        self.batch = build()
        self.blank = b.render_csv(self.batch["csv_rows"])
        self.manifest = b.build_manifest(self.batch, self.blank, {"evaluation_rows.jsonl": "x"},
                                         {"items_checked": 40, "generation_prompts_found_in_cache": 40,
                                          "missing": []})

    def test_hashes_are_consistent(self):
        self.assertEqual(self.manifest["outputs"]["blank_csv_sha256"], b.sha256_bytes(self.blank))
        self.assertEqual(self.manifest["blinding"]["mapping_sha256"], b.sha256_json(self.batch["key"]))
        ids = sorted(r["annotation_id"] for r in self.batch["sampled"])
        self.assertEqual(self.manifest["sampling"]["sampled_annotation_ids"], ids)
        self.assertEqual(self.manifest["sampling"]["selection_sha256"], b.sha256_json(ids))

    def test_manifest_does_not_expose_the_mapping(self):
        text = json.dumps(self.manifest)
        for leak in ("llm_grounded", "baseline_a_echo", "baseline_b_template", "SECRET_GOLD",
                     "PREDICTED_SECRET", "\"R001\"", "\"I01\""):
            self.assertNotIn(leak, text, leak)
        self.assertFalse(self.manifest["blinding"]["mapping_written_to_disk"])
        self.assertFalse(self.manifest["gold_labels_read"])
        self.assertEqual(self.manifest["sampling"]["selection_signals"], ["batch", "escalation decision"])

    def test_retest_plan_is_declared_but_not_built(self):
        plan = self.manifest["retest_plan"]
        self.assertEqual((plan["responses"], plan["per_system"], plan["seed"], plan["minimum_gap_hours"]),
                         (36, 12, 47, 72))
        self.assertIn("not built", plan["status"])


class TestOutputsAndSafety(unittest.TestCase):
    def test_writes_only_the_two_files_and_verifies(self):
        batch = build()
        blank = b.render_csv(batch["csv_rows"])
        result = {"batch": batch, "blank": blank,
                  "manifest": b.build_manifest(batch, blank, {}, {"items_checked": 40,
                                                                  "generation_prompts_found_in_cache": 40,
                                                                  "missing": []})}
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            b.write_outputs(result, directory)
            self.assertEqual(sorted(p.name for p in directory.iterdir()),
                             ["reply_rating_manifest.json", "reply_rating_r01_blank.csv"])
            report = b.verify_on_disk(result, directory)
            self.assertTrue(report["blank_csv_identical"] and report["manifest_identical"])
            (directory / b.BLANK_R01.name).write_bytes(b"edited")
            with self.assertRaises(FileExistsError):
                b.write_outputs(result, directory)

    def test_output_paths_stay_in_human_eval(self):
        for path in (b.BLANK_R01, b.MANIFEST):
            self.assertEqual(path.parent, config.PROJECT_ROOT / "human_eval")
        self.assertNotIn("golden", str(b.BLANK_R01.parent))

    def test_no_model_or_network_call_during_a_build(self):
        with mock.patch.object(llm, "complete", side_effect=AssertionError("LLM called")), \
             mock.patch.object(llm, "_post", side_effect=AssertionError("network called")):
            build()

    def test_imports_and_gold_access(self):
        source = Path(b.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported |= {f"{node.module}.{alias.name}" for alias in node.names}
        for forbidden in ("urllib", "requests", "socket", "http", "src.judge", "src.evaluate",
                          "src.retrieve", "src.run_golden_generation", "src.retry_golden_generation"):
            self.assertFalse(any(name == forbidden or name.startswith(forbidden + ".") for name in imported),
                             forbidden)
        self.assertEqual({n for n in imported if n.startswith("src.llm")}, {"src.llm._cache_key"})
        subscripts = [n.slice.value for n in ast.walk(tree)
                      if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant)]
        self.assertNotIn("gold", subscripts)
        self.assertNotIn("primary_intent", source)
        self.assertNotIn("GOLDEN_SET", source)

    def test_restrict_row_drops_gold_and_metadata(self):
        restricted = b.restrict_row(synthetic_rows()[0])
        self.assertEqual(set(restricted), {"annotation_id", "batch", "customer_message", "decision",
                                           "evidence", "replies", "prompt_intent"})
        self.assertNotIn("SECRET_GOLD", json.dumps(restricted))

    def test_load_frozen_rows_reads_jsonl(self):
        rows = synthetic_rows()[:3]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows) + "\n", encoding="utf-8")
            loaded = b.load_frozen_rows(path)
        self.assertEqual([r["annotation_id"] for r in loaded], [r["annotation_id"] for r in rows])
        self.assertNotIn("gold", loaded[0])

    def test_shuffle_gives_up_loudly_when_impossible(self):
        entries = [{"item_id": "I01"}, {"item_id": "I01"}]
        with mock.patch.object(b, "MAX_SHUFFLE_ATTEMPTS", 5):
            with self.assertRaises(RuntimeError):
                b.shuffle_without_adjacent_items(entries, b.np.random.default_rng(1))


SENTINEL_NOTE = "SENTINEL_NOTE_ZX9"


def synthetic_round1(empty_target: bool = True):
    """A synthetic batch with one empty LLM reply, its blank bytes, and a completed rated file."""
    normal = build()
    target = next(k["annotation_id"] for k in normal["key"] if k["system"] == "llm_grounded")
    batch = build(synthetic_rows(empty={target} if empty_target else set()))
    blank = b.render_csv(batch["csv_rows"])
    status = {k["response_id"]: k["status"] for k in batch["key"]}
    rng = b.np.random.default_rng(123)
    rated_rows = []
    for row in batch["csv_rows"]:
        filled = dict(row)
        if status[row["response_id"]] == b.TO_BE_RATED:
            for column in b.SCORE_COLUMNS:
                filled[column] = str(int(rng.integers(1, 6)))
            filled["notes"] = SENTINEL_NOTE
        rated_rows.append(filled)
    return batch, blank, rated_rows


def render_rows(rows, columns=None) -> bytes:
    columns = columns or b.CSV_COLUMNS
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, quoting=csv.QUOTE_ALL, lineterminator="\r\n",
                            extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")


class TestRound1Validation(unittest.TestCase):
    def setUp(self):
        self.batch, self.blank, self.rated = synthetic_round1()

    def validate(self, rows=None, columns=None, raw=None):
        raw = raw if raw is not None else render_rows(rows if rows is not None else self.rated, columns)
        return b.validate_round1(raw, self.blank, self.batch["key"])

    def test_complete_file_passes(self):
        facts = self.validate()
        self.assertTrue(facts["passed"], facts)
        self.assertEqual((facts["row_count"], facts["ratable_expected"], facts["ratable_complete"]), (120, 119, 119))
        self.assertEqual(len(facts["unrated_expected"]), 1)
        self.assertTrue(facts["unrated_rows_left_blank"])
        self.assertEqual(facts["item_ids_found"], 40)

    def test_facts_never_contain_scores_or_notes(self):
        facts = self.validate()
        text = json.dumps(facts)
        self.assertNotIn(SENTINEL_NOTE, text)
        allowed = {"columns_exact", "row_count", "expected_row_count", "response_ids_unique",
                   "response_ids_match_blank_in_order", "item_ids_expected", "item_ids_found",
                   "text_unchanged", "text_mismatches", "scores_are_blank_or_integers_1_to_5",
                   "invalid_score_cells", "ratable_expected", "ratable_complete", "ratable_incomplete",
                   "unrated_expected", "unrated_rows_left_blank", "unrated_notes_blank",
                   "prohibited_tokens", "identifier_patterns", "passed"}
        self.assertEqual(set(facts), allowed)
        broken = [dict(r) for r in self.rated]
        broken[5]["relevance"] = "7"
        facts = self.validate(broken)
        self.assertEqual(facts["invalid_score_cells"], [(broken[5]["response_id"], "relevance")])
        self.assertNotIn("7", json.dumps(facts["invalid_score_cells"]).replace(broken[5]["response_id"], ""))

    def test_bad_score_formats_fail(self):
        for value in ("6", "0", "4.0", "x", " 3", "3 "):
            rows = [dict(r) for r in self.rated]
            ratable = next(r for r in rows if r["notes"] == SENTINEL_NOTE)
            ratable["claim_safety"] = value
            facts = self.validate(rows)
            self.assertFalse(facts["passed"], value)
            self.assertFalse(facts["scores_are_blank_or_integers_1_to_5"], value)

    def test_incomplete_ratable_row_fails(self):
        rows = [dict(r) for r in self.rated]
        ratable = next(r for r in rows if r["notes"] == SENTINEL_NOTE)
        ratable["helpfulness"] = ""
        facts = self.validate(rows)
        self.assertFalse(facts["passed"])
        self.assertEqual(facts["ratable_incomplete"], [ratable["response_id"]])
        self.assertEqual(facts["ratable_complete"], 118)

    def test_rating_the_empty_reply_fails(self):
        rows = [dict(r) for r in self.rated]
        empty_id = self.validate()["unrated_expected"][0]
        next(r for r in rows if r["response_id"] == empty_id)["relevance"] = "3"
        facts = self.validate(rows)
        self.assertFalse(facts["passed"])
        self.assertFalse(facts["unrated_rows_left_blank"])

    def test_structural_changes_fail(self):
        cases = {
            "missing row": (self.rated[:-1], None),
            "extra row": (self.rated + [dict(self.rated[0])], None),
            "reordered": (list(reversed(self.rated)), None),
            "extra column": ([{**r, "system": ""} for r in self.rated], b.CSV_COLUMNS + ["system"]),
            "dropped column": (self.rated, [c for c in b.CSV_COLUMNS if c != "notes"]),
        }
        for name, (rows, columns) in cases.items():
            self.assertFalse(self.validate(rows, columns)["passed"], name)

    def test_text_change_is_reported_by_position_only(self):
        rows = [dict(r) for r in self.rated]
        rows[3]["candidate_reply"] += " edited"
        rows[4]["evidence_2"] = rows[4]["evidence_2"].replace("\n", "\r\n")
        facts = self.validate(rows)
        self.assertFalse(facts["text_unchanged"])
        self.assertEqual(facts["text_mismatches"],
                         sorted([(rows[3]["response_id"], "candidate_reply"),
                                 (rows[4]["response_id"], "evidence_2")]))

    def test_prohibited_tokens_and_identifiers_fail(self):
        for text in ("looks like baseline_b_template", "same as b01_0042", "flight_delay again", "conv_123456"):
            rows = [dict(r) for r in self.rated]
            rows[7]["notes"] = text
            self.assertFalse(self.validate(rows)["passed"], text)

    def test_non_utf8_file_raises(self):
        with self.assertRaises(ValueError):
            self.validate(raw="response_id\n\xe9".encode("latin-1"))


class TestRetestTiming(unittest.TestCase):
    def test_gap_is_computed_from_the_completion_time(self):
        timing = b.retest_timing(1_000_000.0, 999_000.0, 2_000_000.0)
        completed = b.datetime.fromisoformat(timing["round1_completed_at"])
        earliest = b.datetime.fromisoformat(timing["earliest_retest_start"])
        preferred = b.datetime.fromisoformat(timing["preferred_retest_start"])
        self.assertEqual((earliest - completed).total_seconds(), 72 * 3600)
        self.assertEqual((preferred - completed).total_seconds(), 7 * 86400)
        self.assertEqual((timing["minimum_gap_hours"], timing["preferred_gap_days"]), (72, 7))

    def test_implausible_times_stop(self):
        with self.assertRaises(ValueError):
            b.retest_timing(100.0, 200.0, 300.0)
        with self.assertRaises(ValueError):
            b.retest_timing(500.0, 100.0, 300.0)


class TestRetestBatch(unittest.TestCase):
    def setUp(self):
        self.batch, self.blank, self.rated = synthetic_round1()
        self.retest = b.build_retest(self.batch["csv_rows"], self.batch["key"])
        self.status = {k["response_id"]: k for k in self.batch["key"]}

    def test_36_rows_12_per_hidden_system_from_rated_responses_only(self):
        self.assertEqual(len(self.retest["rows"]), 36)
        self.assertEqual(Counter(k["system"] for k in self.retest["key"]),
                         {s: 12 for s in b.SYSTEMS})
        self.assertTrue(all(self.status[k["response_id"]]["status"] == b.TO_BE_RATED
                            for k in self.retest["key"]))
        self.assertEqual(len({k["response_id"] for k in self.retest["key"]}), 36)

    def test_ids_columns_blank_scores_and_order(self):
        rows = self.retest["rows"]
        self.assertEqual([r["retest_id"] for r in rows], [f"T{n:02d}" for n in range(1, 37)])
        self.assertTrue(all(list(r) == b.RETEST_COLUMNS for r in rows))
        self.assertTrue(all(r[c] == "" for r in rows for c in b.SCORE_COLUMNS + ["notes"]))
        items = [k["item_id"] for k in self.retest["key"]]
        self.assertTrue(all(x != y for x, y in zip(items, items[1:])))
        self.assertNotEqual([k["response_id"] for k in self.retest["key"]],
                            sorted(k["response_id"] for k in self.retest["key"]))

    def test_text_matches_the_frozen_round1_rows(self):
        blank_by_id = {r["response_id"]: r for r in self.batch["csv_rows"]}
        for row, k in zip(self.retest["rows"], self.retest["key"]):
            source = blank_by_id[k["response_id"]]
            for column in ["customer_message", "candidate_reply"] + b.EVIDENCE_COLUMNS:
                self.assertEqual(row[column], source[column])

    def test_no_round1_values_can_reach_the_retest(self):
        from_rated = b.build_retest(self.rated, self.batch["key"])
        self.assertEqual(from_rated["rows"], self.retest["rows"])
        text = b.render_retest_csv(from_rated["rows"]).decode("utf-8-sig")
        self.assertNotIn(SENTINEL_NOTE, text)

    def test_no_identifying_metadata_in_the_csv(self):
        text = b.render_retest_csv(self.retest["rows"]).decode("utf-8-sig")
        for secret in SECRETS + ["response_id", "item_id"]:
            self.assertNotIn(secret, text, secret)
        self.assertIsNone(re.search(r"\bR\d{3}\b|\bI\d{2}\b|b0[12]_\d{4}", text))
        header = text.splitlines()[0].replace('"', "").split(",")
        self.assertEqual(header, b.RETEST_COLUMNS)

    def test_deterministic_and_seeded(self):
        again = b.build_retest(self.batch["csv_rows"], self.batch["key"])
        self.assertEqual(again["key"], self.retest["key"])
        self.assertEqual(b.render_retest_csv(again["rows"]), b.render_retest_csv(self.retest["rows"]))
        other = b.build_retest(self.batch["csv_rows"], self.batch["key"], seed=48)
        self.assertNotEqual(other["selected"], self.retest["selected"])
        self.assertEqual(b.RETEST_SEED, 47)

    def test_too_few_rated_responses_raise(self):
        key = [dict(k, status=b.NOT_RATED_EMPTY_REPLY) if k["system"] == "baseline_a_echo" else k
               for k in self.batch["key"]]
        with self.assertRaises(ValueError):
            b.build_retest(self.batch["csv_rows"], key)


class TestRetestManifestAndFiles(unittest.TestCase):
    def setUp(self):
        self.batch, self.blank, self.rated = synthetic_round1()
        self.rated_raw = render_rows(self.rated)
        self.validation = b.validate_round1(self.rated_raw, self.blank, self.batch["key"])
        self.timing = b.retest_timing(1_000_000.0, 999_000.0, 2_000_000.0)
        self.retest = b.build_retest(self.batch["csv_rows"], self.batch["key"])
        self.blank_retest = b.render_retest_csv(self.retest["rows"])
        self.manifest = b.build_retest_manifest(self.rated_raw, self.validation, self.timing,
                                                self.retest, self.blank_retest, b.sha256_bytes(self.blank))

    def test_manifest_hashes_and_no_mapping_or_ratings(self):
        m = self.manifest
        self.assertEqual(m["retest"]["mapping_sha256"], b.sha256_json(self.retest["key"]))
        self.assertEqual(m["retest"]["selection_sha256"], b.sha256_json(self.retest["selected"]))
        self.assertEqual(m["outputs"]["blank_csv_sha256"], b.sha256_bytes(self.blank_retest))
        self.assertEqual(m["round1"]["sha256"], b.sha256_bytes(self.rated_raw))
        self.assertFalse(m["retest"]["mapping_written_to_disk"])
        self.assertFalse(m["retest"]["round1_ratings_included"])
        self.assertFalse(m["round1"]["ratings_included_here"])
        self.assertEqual((m["retest"]["responses"], m["retest"]["responses_per_system"]), (36, 12))
        text = json.dumps(m)
        for leak in ("llm_grounded", "baseline_a_echo", "baseline_b_template", SENTINEL_NOTE, '"T01"'):
            self.assertNotIn(leak, text, leak)
        for k in self.retest["key"]:
            self.assertNotIn(f'"{k["response_id"]}"', json.dumps(m["retest"]))

    def test_writes_two_files_refuses_failed_validation_and_overwrites(self):
        result = {"validation": self.validation, "blank": self.blank_retest, "manifest": self.manifest}
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            b.write_retest_outputs(result, directory)
            self.assertEqual(sorted(p.name for p in directory.iterdir()),
                             ["reply_rating_retest_blank.csv", "reply_rating_retest_manifest.json"])
            report = b.verify_retest_on_disk(result, directory)
            self.assertTrue(report["retest_blank_identical"] and report["retest_manifest_identical"])
            (directory / b.BLANK_RETEST.name).write_bytes(b"edited")
            with self.assertRaises(FileExistsError):
                b.write_retest_outputs(result, directory)
        failed = dict(result, validation=dict(self.validation, passed=False))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                b.write_retest_outputs(failed, Path(tmp))
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_rebuild_reads_the_rated_file_without_changing_it(self):
        import os
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            blank_path, rated_path = directory / "blank.csv", directory / "rated.csv"
            blank_path.write_bytes(self.blank)
            rated_path.write_bytes(self.rated_raw)
            os.utime(blank_path, (1_000_000, 1_000_000))
            os.utime(rated_path, (1_003_600, 1_003_600))
            before = (rated_path.read_bytes(), rated_path.stat().st_mtime)
            r01 = {"blank": self.blank, "batch": self.batch}
            with mock.patch.object(b, "BLANK_R01", blank_path), mock.patch.object(b, "RATED_R01", rated_path):
                result = b.rebuild_retest(r01)
            self.assertEqual((rated_path.read_bytes(), rated_path.stat().st_mtime), before)
            self.assertTrue(result["validation"]["passed"])
            self.assertEqual(len(result["retest"]["rows"]), 36)
            self.assertEqual(result["timing"]["round1_completed_at"],
                             b.retest_timing(1_003_600, 1_000_000, 2e9)["round1_completed_at"])

    def test_rebuild_refuses_a_changed_round1_blank(self):
        with tempfile.TemporaryDirectory() as tmp:
            blank_path = Path(tmp) / "blank.csv"
            blank_path.write_bytes(self.blank + b"x")
            with mock.patch.object(b, "BLANK_R01", blank_path):
                with self.assertRaises(ValueError):
                    b.rebuild_retest({"blank": self.blank, "batch": self.batch})

    def test_cli_prints_no_score_values(self):
        source = Path(b.__file__).read_text(encoding="utf-8")
        main_body = source[source.index("def main("):]
        self.assertNotIn("rated_rows", main_body)
        self.assertNotIn("csv_rows\"]", main_body.split("if args.retest")[1].split("write_outputs(result)")[0])


# --------------------------------------------------------------------------- Stage 2: analysis
# Small hand-checkable fixture: 4 items x 3 systems; the LLM reply for I03 is empty.
# Scores are (relevance, helpfulness, groundedness, information_request, claim_safety).

from src import analyse_reply_ratings as ar

L, A, B = "llm_grounded", "baseline_a_echo", "baseline_b_template"
FIXTURE_SCORES = {
    ("I01", L): (5, 4, 4, 5, 5), ("I01", A): (3, 2, 5, 5, 5), ("I01", B): (2, 3, 3, 4, 5),
    ("I02", L): (4, 4, 3, 5, 1), ("I02", A): (4, 2, 5, 5, 2), ("I02", B): (2, 3, 3, 4, 5),
    ("I03", L): None,            ("I03", A): (1, 1, 5, 5, 5), ("I03", B): (2, 3, 3, 4, 5),
    ("I04", L): (2, 2, 2, 1, 4), ("I04", A): (3, 1, 5, 5, 5), ("I04", B): (2, 3, 3, 4, 5),
}
FIXTURE_ORDER = [("I02", B), ("I01", L), ("I03", A), ("I04", B), ("I02", L), ("I01", A),
                 ("I03", L), ("I04", A), ("I01", B), ("I02", A), ("I04", L), ("I03", B)]
AUTOMATED = {
    "a1": {L: {"flag_codes": [], "needs_more_information": True, "escalation_decision": "auto_handle"},
           A: {"flag_codes": ["G7_excessive_copying"]}, B: {"flag_codes": []}},
    "a2": {L: {"flag_codes": ["G4_action_claim"], "needs_more_information": False,
               "escalation_decision": "auto_handle"},
           A: {"flag_codes": ["G7_excessive_copying"]}, B: {"flag_codes": []}},
    "a3": {L: {"flag_codes": ["G1_empty_reply"], "needs_more_information": None,
               "escalation_decision": "escalate"},
           A: {"flag_codes": ["G7_excessive_copying"]}, B: {"flag_codes": []}},
    "a4": {L: {"flag_codes": ["G2_unsupported_url"], "needs_more_information": True,
               "escalation_decision": "escalate"},
           A: {"flag_codes": []}, B: {"flag_codes": []}},
}


def fixture_round1(scores=FIXTURE_SCORES):
    key, blank_rows, rated_rows = [], [], []
    for number, (item, system) in enumerate(FIXTURE_ORDER, start=1):
        rid = f"R{number:03d}"
        empty = scores[(item, system)] is None
        key.append({"response_id": rid, "item_id": item, "annotation_id": f"a{item[-1]}",
                    "system": system, "status": b.NOT_RATED_EMPTY_REPLY if empty else b.TO_BE_RATED})
        blank = {"response_id": rid, "item_id": item,
                 "customer_message": f"Customer message for {item}.",
                 **{c: f"[{n}] customer: example {item} {n}" for n, c in enumerate(b.EVIDENCE_COLUMNS, 1)},
                 "candidate_reply": b.EMPTY_REPLY_PLACEHOLDER if empty else f"Candidate reply {number}.",
                 **{c: "" for c in b.SCORE_COLUMNS}, "notes": ""}
        blank_rows.append(blank)
        rated = dict(blank)
        if not empty:
            rated.update({c: str(v) for c, v in zip(b.SCORE_COLUMNS, scores[(item, system)])})
        rated_rows.append(rated)
    return key, blank_rows, rated_rows


RETEST_UNITS = [("I01", L), ("I01", A), ("I01", B), ("I02", L), ("I02", A), ("I04", B)]


def fixture_retest(key, rated_rows, changes=None):
    """Retest rows copy round-1 text; scores equal round 1 unless `changes` overrides them."""
    by_unit = {(k["item_id"], k["system"]): (k, r) for k, r in zip(key, rated_rows)}
    retest_key, blank_rows, rated = [], [], []
    for number, unit in enumerate(RETEST_UNITS, start=1):
        k, source = by_unit[unit]
        tid = f"T{number:02d}"
        retest_key.append({"retest_id": tid, "response_id": k["response_id"],
                           "item_id": k["item_id"], "system": k["system"]})
        blank = {"retest_id": tid, "customer_message": source["customer_message"],
                 **{c: source[c] for c in b.EVIDENCE_COLUMNS}, "candidate_reply": source["candidate_reply"],
                 **{c: "" for c in b.SCORE_COLUMNS}, "notes": ""}
        blank_rows.append(blank)
        row = dict(blank)
        row.update({c: source[c] for c in b.SCORE_COLUMNS})
        row.update((changes or {}).get(tid, {}))
        rated.append(row)
    return retest_key, blank_rows, rated


def run_analysis(scores=FIXTURE_SCORES, changes=None, automated=AUTOMATED, **overrides):
    key, blank_rows, rated_rows = fixture_round1(scores)
    retest_key, retest_blank, retest_rated = fixture_retest(key, rated_rows, changes)
    arguments = dict(round1_raw=render_rows(rated_rows), round1_blank_raw=render_rows(blank_rows), key=key,
                     automated=automated, retest_raw=render_rows(retest_rated, b.RETEST_COLUMNS),
                     retest_blank_raw=render_rows(retest_blank, b.RETEST_COLUMNS), retest_key=retest_key,
                     input_kind="synthetic")
    arguments.update(overrides)
    return ar.analyse(**arguments)


_ANALYSIS_CACHE = {}


def cached_analysis(changes=None) -> dict:
    """Analysis of the default fixture, computed once per distinct `changes`."""
    marker = json.dumps(changes, sort_keys=True)
    if marker not in _ANALYSIS_CACHE:
        _ANALYSIS_CACHE[marker] = run_analysis(changes=changes)
    return _ANALYSIS_CACHE[marker]


DISAGREEMENT = {"T01": {"relevance": "4"}, "T02": {"relevance": "5"}, "T04": {"relevance": "4"}}


class TestAnalysisDescriptive(unittest.TestCase):
    stats = cached_analysis()

    def test_counts_and_provenance(self):
        p = self.stats["provenance"]
        self.assertEqual(p["input_kind"], "synthetic")
        self.assertEqual(p["analysis_version"], "6f-a1")
        self.assertEqual(p["counts"], {"items": 4, "responses": 12, "rated_responses": 11, "retest_responses": 6})
        self.assertEqual(p["seeds"], {"item_bootstrap": 42, "retest_kappa_bootstrap": 44})
        self.assertEqual(p["bootstrap_resamples"], 2000)
        self.assertEqual(set(p["input_sha256"]), {"round1_rated", "round1_blank", "retest_rated", "retest_blank"})
        self.assertTrue(p["round1_validation"]["passed"] and p["retest_validation"]["passed"])

    def test_llm_relevance_statistics(self):
        rel = self.stats["systems"][L]["dimensions"]["relevance"]
        self.assertEqual((rel["n"], rel["mean"], rel["median"]), (3, 3.6667, 4.0))
        self.assertEqual(rel["distribution"], {"1": 0, "2": 1, "3": 0, "4": 1, "5": 1})
        self.assertEqual((rel["share_le_2"], rel["share_ge_4"]), (0.3333, 0.6667))
        self.assertEqual(self.stats["systems"][L]["not_rated_empty_reply"], 1)
        self.assertEqual(self.stats["systems"][L]["responses"], 4)

    def test_other_systems(self):
        a_help = self.stats["systems"][A]["dimensions"]["helpfulness"]
        self.assertEqual((a_help["n"], a_help["mean"], a_help["median"]), (4, 1.5, 1.5))
        self.assertEqual((a_help["share_le_2"], a_help["share_ge_4"]), (1.0, 0.0))
        b_rel = self.stats["systems"][B]["dimensions"]["relevance"]
        self.assertEqual((b_rel["mean"], b_rel["mean_interval_95"]), (2.0, [2.0, 2.0]))

    def test_bootstrap_interval_brackets_the_mean(self):
        for system in (L, A):
            for d in b.SCORE_COLUMNS:
                entry = self.stats["systems"][system]["dimensions"][d]
                low, high = entry["mean_interval_95"]
                self.assertLessEqual(low, entry["mean"])
                self.assertGreaterEqual(high, entry["mean"])

    def test_rerun_is_identical(self):
        self.assertEqual(run_analysis(), self.stats)

    def test_item_resamples_keep_items_whole(self):
        draws = ar.item_resamples(4)
        self.assertEqual(draws.shape, (2000, 4))
        self.assertTrue(((draws >= 0) & (draws < 4)).all())
        self.assertTrue((draws == ar.item_resamples(4)).all())
        values = b.np.array([1.0, 2.0, b.np.nan, 4.0])
        manual = []
        for row in draws:
            sample = values[row]
            if not b.np.isnan(sample).all():
                manual.append(b.np.nanmean(sample))
        expected = [round(float(b.np.percentile(manual, q)), 4) for q in (2.5, 97.5)]
        self.assertEqual(ar.bootstrap_mean_interval(values, draws), expected)
        self.assertIsNone(ar.bootstrap_mean_interval(b.np.full(4, b.np.nan), draws))

    def test_describe_empty(self):
        self.assertEqual(ar.describe([])["n"], 0)
        self.assertIsNone(ar.describe([])["mean"])


class TestAnalysisPairs(unittest.TestCase):
    stats = cached_analysis()

    def test_llm_minus_baseline_b(self):
        dims = self.stats["paired_comparisons"][f"{L}_minus_{B}"]["dimensions"]
        rel, claim = dims["relevance"], dims["claim_safety"]
        self.assertEqual((rel["pairs"], rel["unpaired_items"], rel["first_higher"], rel["tie"],
                          rel["second_higher"], rel["mean_difference"]), (3, 1, 2, 1, 0, 1.6667))
        self.assertEqual((claim["first_higher"], claim["tie"], claim["second_higher"], claim["mean_difference"]),
                         (0, 1, 2, -1.6667))

    def test_llm_minus_baseline_a(self):
        rel = self.stats["paired_comparisons"][f"{L}_minus_{A}"]["dimensions"]["relevance"]
        self.assertEqual((rel["first_higher"], rel["tie"], rel["second_higher"], rel["mean_difference"]),
                         (1, 1, 1, 0.3333))
        low, high = rel["mean_difference_interval_95"]
        self.assertLessEqual(low, 0.3333)
        self.assertGreaterEqual(high, 0.3333)

    def test_constant_difference_has_a_degenerate_interval(self):
        scores = {k: v for k, v in FIXTURE_SCORES.items()}
        for item in ("I01", "I02", "I03", "I04"):
            scores[(item, L)] = (4, 4, 4, 4, 4)
            scores[(item, B)] = (3, 3, 3, 3, 3)
        dims = run_analysis(scores)["paired_comparisons"][f"{L}_minus_{B}"]["dimensions"]
        self.assertEqual(dims["groundedness"]["mean_difference_interval_95"], [1.0, 1.0])
        self.assertEqual(dims["groundedness"]["first_higher"], 4)

    def test_descriptive_only(self):
        for entry in self.stats["paired_comparisons"].values():
            self.assertEqual(entry["kind"], "descriptive paired comparison")
        text = json.dumps(self.stats).lower()
        for word in ("winner", "ranking", "superior", "better", "recommend"):
            self.assertNotIn(word, text)


class TestAnalysisAutomatedVsHuman(unittest.TestCase):
    stats = cached_analysis()

    def test_claim_flag_crosstab(self):
        cross = self.stats["claim_flags_vs_human_claim_safety"]
        self.assertEqual({k: cross[L][k] for k in ("flagged_low", "flagged_not_low", "unflagged_low",
                                                   "unflagged_not_low", "rated")},
                         {"flagged_low": 1, "flagged_not_low": 1, "unflagged_low": 0,
                          "unflagged_not_low": 1, "rated": 3})
        self.assertEqual((cross[L]["share_of_flagged_rated_low"], cross[L]["share_of_low_that_were_flagged"],
                          cross[L]["share_of_unflagged_rated_low"]), (0.5, 1.0, 0.0))
        self.assertEqual((cross[A]["flagged_low"], cross[A]["flagged_not_low"], cross[A]["unflagged_not_low"]),
                         (1, 2, 1))
        self.assertEqual(cross[A]["share_of_flagged_rated_low"], 0.3333)
        self.assertEqual(cross[B]["unflagged_not_low"], 4)
        self.assertIsNone(cross[B]["share_of_flagged_rated_low"])

    def test_g1_alone_is_not_a_claim_flag(self):
        self.assertNotIn("G1_empty_reply", ar.CLAIM_FLAG_CODES)
        self.assertEqual(len(ar.CLAIM_FLAG_CODES), 6)

    def test_needs_more_information_breakdown(self):
        info = self.stats["llm_behaviour"]["information_request_by_needs_more_information"]
        self.assertEqual((info["true"]["n"], info["true"]["mean"]), (2, 3.0))
        self.assertEqual((info["false"]["n"], info["false"]["mean"]), (1, 5.0))

    def test_escalation_breakdown(self):
        decisions = self.stats["llm_behaviour"]["ratings_by_escalation_decision"]
        self.assertEqual((decisions["auto_handle"]["rated"], decisions["auto_handle"]["claim_safety_le_2"],
                          decisions["auto_handle"]["share_claim_safety_le_2"]), (2, 1, 0.5))
        self.assertEqual((decisions["escalate"]["rated"], decisions["escalate"]["claim_safety_le_2"]), (1, 0))
        self.assertEqual(decisions["auto_handle"]["dimensions"]["relevance"]["mean"], 4.5)

    def test_missing_automated_metadata_raises(self):
        partial = {k: v for k, v in AUTOMATED.items() if k != "a4"}
        with self.assertRaises(ar.RatingValidationError):
            run_analysis(automated=partial)


class TestAnalysisRetest(unittest.TestCase):
    def test_perfect_agreement(self):
        rt = cached_analysis()["test_retest"]
        self.assertIn("intra-annotator", rt["measurement"])
        self.assertIn("not inter-annotator", rt["measurement"])
        self.assertEqual(rt["responses"], 6)
        self.assertEqual(rt["responses_per_system"], {L: 2, A: 2, B: 2})
        claim = rt["per_dimension"]["claim_safety"]
        self.assertEqual((claim["exact_agreement"], claim["within_one_agreement"],
                          claim["mean_absolute_difference"], claim["weighted_kappa_quadratic"]), (1.0, 1.0, 0.0, 1.0))
        self.assertEqual(claim["round1_distribution"], {"1": 1, "2": 1, "3": 0, "4": 0, "5": 4})
        self.assertEqual(rt["pooled"]["pairs"], 30)
        self.assertIn("not independent", rt["pooled"]["note"])
        self.assertIn("concentrated", rt["note"])

    def test_disagreement(self):
        rel = cached_analysis(DISAGREEMENT)["test_retest"]["per_dimension"]["relevance"]
        first, second = [5, 3, 2, 4, 4, 2], [4, 5, 2, 4, 4, 2]
        self.assertEqual((rel["exact_agreement"], rel["within_one_agreement"], rel["mean_absolute_difference"]),
                         (0.6667, 0.8333, 0.5))
        expected = ar.cohen_kappa_score(first, second, labels=ar.SCALE, weights="quadratic")
        self.assertEqual(rel["weighted_kappa_quadratic"], round(float(expected), 4))
        self.assertEqual(rel["retest_distribution"], {"1": 0, "2": 2, "3": 0, "4": 3, "5": 1})

    def test_weighted_kappa_hand_value(self):
        self.assertAlmostEqual(ar.weighted_kappa(b.np.array([1, 2, 3, 4]), b.np.array([1, 2, 4, 4])), 11 / 12)
        self.assertAlmostEqual(ar.weighted_kappa(b.np.array([1, 2, 3, 4, 5]), b.np.array([5, 4, 3, 2, 1])), -1.0)

    def test_undefined_and_clustered_kappa(self):
        constant = ar.agreement(b.np.array([5, 5, 5]), b.np.array([5, 5, 5]))
        self.assertEqual((constant["exact_agreement"], constant["weighted_kappa_quadratic"]), (1.0, None))
        clustered = ar.agreement(b.np.array([5, 5, 5, 5, 4]), b.np.array([5, 5, 5, 5, 5]))
        self.assertEqual((clustered["exact_agreement"], clustered["weighted_kappa_quadratic"]), (0.8, 0.0))
        boot = ar.bootstrap_kappa(b.np.array([[5], [5]]), b.np.array([[5], [5]]),
                                  b.np.zeros((10, 2), dtype=int))
        self.assertEqual((boot["usable_resamples"], boot["undefined_resamples"], boot["interval_95"]), (0, 10, None))

    def test_bootstrap_kappa_is_seeded_and_resamples_units(self):
        rt1 = cached_analysis(DISAGREEMENT)["test_retest"]
        rt2 = run_analysis(changes=DISAGREEMENT)["test_retest"]
        self.assertEqual(rt1, rt2)
        boot = rt1["pooled"]["kappa_bootstrap"]
        self.assertEqual(boot["resamples"], 2000)
        self.assertEqual(boot["usable_resamples"] + boot["undefined_resamples"], 2000)

    def test_retest_mapped_to_unrated_response_raises(self):
        key, blank_rows, rated_rows = fixture_round1()
        retest_key, retest_blank, retest_rated = fixture_retest(key, rated_rows)
        unrated = next(k["response_id"] for k in key if k["status"] == b.NOT_RATED_EMPTY_REPLY)
        retest_key[0] = dict(retest_key[0], response_id=unrated)
        table = ar.ratings_table(render_rows(rated_rows), key)
        with self.assertRaises(ar.RatingValidationError):
            ar.retest_agreement(render_rows(retest_rated, b.RETEST_COLUMNS), retest_key, table)


class TestAnalysisValidation(unittest.TestCase):
    def setUp(self):
        self.key, self.blank_rows, self.rated_rows = fixture_round1()
        self.blank = render_rows(self.blank_rows)

    def check(self, rows, columns=None, key=None):
        return ar.require_valid_round1(render_rows(rows, columns), self.blank, key or self.key)

    def test_valid_round1(self):
        self.assertTrue(self.check(self.rated_rows)["passed"])

    def test_round1_failures(self):
        def modified(index, **changes):
            rows = [dict(r) for r in self.rated_rows]
            rows[index].update(changes)
            return rows
        cases = {
            "invalid value": modified(0, relevance="6"),
            "float value": modified(0, relevance="4.0"),
            "blank rating": modified(0, claim_safety=""),
            "rated empty reply": modified(6, relevance="3"),
            "changed text": modified(1, candidate_reply="edited"),
            "missing id": self.rated_rows[:-1],
            "duplicate id": self.rated_rows[:-1] + [dict(self.rated_rows[0])],
        }
        for name, rows in cases.items():
            with self.assertRaises(ar.RatingValidationError, msg=name):
                self.check(rows)
        with self.assertRaises(ar.RatingValidationError):
            self.check([{**r, "system": ""} for r in self.rated_rows], b.CSV_COLUMNS + ["system"])

    def test_system_mapping_problems(self):
        missing = [dict(k) for k in self.key]
        missing[0].pop("system")
        with self.assertRaises(ar.RatingValidationError):
            self.check(self.rated_rows, key=missing)
        doubled = [dict(k) for k in self.key]
        doubled[0]["system"] = next(k["system"] for k in self.key
                                    if k["item_id"] == doubled[0]["item_id"] and k is not self.key[0])
        with self.assertRaises(ar.RatingValidationError):
            self.check(self.rated_rows, key=doubled)

    def test_malformed_input(self):
        with self.assertRaises(ValueError):
            ar.require_valid_round1(b"\xff\xfe\x00bad", self.blank, self.key)

    def test_retest_failures(self):
        retest_key, retest_blank, retest_rated = fixture_retest(self.key, self.rated_rows)
        blank = render_rows(retest_blank, b.RETEST_COLUMNS)
        self.assertTrue(ar.validate_retest(render_rows(retest_rated, b.RETEST_COLUMNS), blank)["passed"])
        cases = {
            "blank rating": [dict(r, relevance="") if i == 0 else r for i, r in enumerate(retest_rated)],
            "float": [dict(r, helpfulness="3.0") if i == 1 else r for i, r in enumerate(retest_rated)],
            "changed text": [dict(r, customer_message="x") if i == 2 else r for i, r in enumerate(retest_rated)],
            "reordered": list(reversed(retest_rated)),
            "missing": retest_rated[:-1],
            "token": [dict(r, notes="baseline_a_echo?") if i == 0 else r for i, r in enumerate(retest_rated)],
        }
        for name, rows in cases.items():
            with self.assertRaises(ar.RatingValidationError, msg=name):
                ar.validate_retest(render_rows(rows, b.RETEST_COLUMNS), blank)
        with self.assertRaises(ar.RatingValidationError):
            ar.validate_retest(render_rows([{**r, "response_id": "R001"} for r in retest_rated],
                                           b.RETEST_COLUMNS + ["response_id"]), blank)

    def test_input_kind_must_be_declared(self):
        with self.assertRaises(ValueError):
            run_analysis(input_kind="unknown")


class TestAnalysisSafety(unittest.TestCase):
    def test_real_rating_files_need_explicit_permission(self):
        for path in (b.RATED_R01, ar.RATED_RETEST):
            with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("file was read")):
                with self.assertRaises(PermissionError):
                    ar.read_ratings(path)

    def test_synthetic_files_are_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ratings.csv"
            path.write_bytes(b"x")
            self.assertEqual(ar.read_ratings(path), b"x")

    def test_cli_refuses_without_confirmation_and_needs_both_files(self):
        with mock.patch.object(ar, "read_ratings", side_effect=AssertionError("read")), \
             mock.patch.object(ar.batch_builder, "rebuild_batch", side_effect=AssertionError("rebuilt")):
            self.assertEqual(ar.main(["--round1", "x.csv", "--retest", "y.csv"]), 2)
            with self.assertRaises(SystemExit):
                ar.main(["--round1", "x.csv", "--confirm-real-ratings"])

    def test_no_discovery_of_rating_files(self):
        source = Path(ar.__file__).read_text(encoding="utf-8")
        for token in ("glob(", "iterdir(", "listdir(", "os.walk", "urllib", "requests", "src.judge",
                      "src.llm", "primary_intent", "GOLDEN_SET"):
            self.assertNotIn(token, source)

    def test_these_tests_never_name_the_real_files(self):
        own = Path(__file__).read_text(encoding="utf-8")
        self.assertNotIn("reply_rating_r01_" + "rated.csv", own)
        self.assertNotIn("reply_rating_retest_" + "rated.csv", own)

    def test_report_renders_without_ranking_language(self):
        report = ar.render_report(cached_analysis())
        self.assertIn("Input: **synthetic**", report)
        self.assertIn("descriptive", report)
        for phrase in ("is better", "best system", "outperform", "superior"):
            self.assertNotIn(phrase, report.lower())
        self.assertIn("intra-annotator", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
