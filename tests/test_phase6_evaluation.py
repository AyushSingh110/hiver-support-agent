"""Phase 6E harness tests. Synthetic fixtures only: no gold labels, no LLM, no network."""
from __future__ import annotations

import ast
import copy
import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from src import evaluate as ev
from src.classify_intent import score as phase6a_score
from src.escalate import Thresholds
from src.generate_reply import TEMPLATE_BASELINE_REPLY, generate_llm, parse_response
from src.taxonomy import ALL_LABELS

THRESHOLDS = Thresholds(intent_confidence=0.19128133486537377, top1_similarity=0.198143,
                        min_evidence_count=2)


def evidence(n: int = 3, top: float = 0.5, weak=("baggage", None, "flight_delay")) -> list[dict]:
    return [
        {
            "rank": i + 1,
            "conversation_id": f"conv_syn_{i}",
            "tweet_id": 1000 + i,
            "similarity": round(top - 0.05 * i, 6),
            "historical_customer_message": f"my bag is missing at the airport case {i}",
            "historical_brand_reply": f"@12345{i} Sorry to hear that. Please DM your record locator, case {i}.",
            "historical_weak_intent": weak[i % len(weak)],
            "n_turns": 2, "n_brand_turns": 1, "customer_followups": 0,
            "evidence_shape": "two_turn", "retrieved_by": "tfidf",
        }
        for i in range(n)
    ]


def llm_record(text: str, items: list[dict], message: str = "my bag never arrived") -> dict:
    def complete(prompt, **kwargs):
        return {"text": text, "model": "llama3.1:8b", "latency_seconds": 1.5, "from_cache": False}
    return generate_llm(message, "baggage", items, complete_fn=complete, use_cache=False)


PARSED = '{"reply":"Sorry about your bag. Could you DM us your record locator?",' \
         '"needs_more_information":true,"evidence_used":["1","2"]}'


class TestClassifierMetrics(unittest.TestCase):
    truth = ["baggage", "baggage", "flight_delay", "OTHER", "praise_and_compliment"]
    predicted = ["baggage", "flight_delay", "flight_delay", "baggage", "praise_and_compliment"]

    def test_basic_metrics(self):
        m = ev.classification_metrics(self.truth, self.predicted, ALL_LABELS)
        self.assertEqual(m["n"], 5)
        self.assertEqual(m["accuracy"], 0.6)
        self.assertEqual(m["per_intent"]["baggage"], {"support": 2, "precision": 0.5, "recall": 0.5, "f1": 0.5})
        self.assertEqual(m["per_intent"]["flight_delay"]["precision"], 0.5)
        self.assertEqual(m["per_intent"]["flight_delay"]["recall"], 1.0)
        self.assertEqual(m["per_intent"]["OTHER"]["support"], 1)
        self.assertEqual(m["per_intent"]["OTHER"]["recall"], 0.0)

    def test_confusion_matrix(self):
        m = ev.classification_metrics(self.truth, self.predicted, ALL_LABELS)
        matrix = m["confusion_matrix"]["rows_true_columns_predicted"]
        labels = m["confusion_matrix"]["labels"]
        self.assertEqual(len(matrix), 13)
        self.assertEqual(matrix[labels.index("OTHER")][labels.index("baggage")], 1)
        self.assertEqual(sum(map(sum, matrix)), 5)

    def test_matches_phase6a_scoring(self):
        ours = ev.classification_metrics(self.truth, self.predicted, ALL_LABELS)
        theirs = phase6a_score("x", np.array(self.truth), np.array(self.predicted))
        for key in ("n", "accuracy", "macro_f1", "weighted_f1", "per_intent"):
            self.assertEqual(ours[key], theirs[key], key)

    def test_macro_f1_counts_absent_labels_as_zero(self):
        m = ev.classification_metrics(["baggage"], ["baggage"], ALL_LABELS)
        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["macro_f1"], round(1 / 13, 4))

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            ev.classification_metrics(["a"], ["a", "b"], ALL_LABELS)
        with self.assertRaises(ValueError):
            ev.classification_metrics([], [], ALL_LABELS)

    def test_majority_label(self):
        self.assertEqual(ev.majority_label(["a", "b", "b", "c"]), "b")


class TestBootstrap(unittest.TestCase):
    truth = ["baggage", "flight_delay"] * 20
    predicted = ["baggage", "baggage"] * 20

    def test_deterministic(self):
        a = ev.bootstrap_ci(self.truth, self.predicted, ALL_LABELS, resamples=200)
        b = ev.bootstrap_ci(self.truth, self.predicted, ALL_LABELS, resamples=200)
        self.assertEqual(a, b)
        self.assertEqual(a["seed"], 42)

    def test_interval_brackets_point_estimate(self):
        ci = ev.bootstrap_ci(self.truth, self.predicted, ALL_LABELS, resamples=500)
        self.assertLessEqual(ci["accuracy"][0], 0.5)
        self.assertGreaterEqual(ci["accuracy"][1], 0.5)

    def test_perfect_predictions(self):
        ci = ev.bootstrap_ci(self.truth, self.truth, ALL_LABELS, resamples=50)
        self.assertEqual(ci["accuracy"], [1.0, 1.0])

    def test_default_is_2000_resamples(self):
        self.assertEqual(ev.BOOTSTRAP_RESAMPLES, 2000)


class TestPredictableView(unittest.TestCase):
    def test_excludes_unpredictable_gold_rows(self):
        truth = ["baggage", "OTHER", "UNCLEAR", "non_support_commentary", "flight_delay"]
        predicted = ["baggage", "baggage", "baggage", "praise_and_compliment", "baggage"]
        view = ev.predictable_intent_view(truth, predicted)
        self.assertEqual(view["rows_included"], 2)
        self.assertEqual(view["rows_outside_view"], 3)
        self.assertEqual(view["metrics"]["accuracy"], 0.5)
        self.assertEqual(list(view["metrics"]["per_intent"]), ev.PREDICTABLE_LABELS)
        self.assertTrue(view["view"].startswith("secondary"))

    def test_predictable_labels_are_the_ten_trainable_intents(self):
        self.assertEqual(len(ev.PREDICTABLE_LABELS), 10)
        self.assertFalse({"OTHER", "UNCLEAR", "non_support_commentary"} & set(ev.PREDICTABLE_LABELS))


class TestRetrievalMeasures(unittest.TestCase):
    def lists(self):
        return [
            [{"weak_intent": "baggage", "similarity": 0.5}, {"weak_intent": None, "similarity": 0.4},
             {"weak_intent": "flight_delay", "similarity": 0.3}],
            [{"weak_intent": None, "similarity": 0.2}],
            [],
        ]

    def test_predicted_consistency(self):
        m = ev.predicted_intent_vs_weak_label_consistency(self.lists(), ["baggage", "baggage", "baggage"])
        self.assertEqual(m["measure"], "predicted_intent_vs_weak_label_consistency")
        self.assertEqual(m["evidence_items"], 4)
        self.assertEqual(m["evidence_with_weak_label"], 2)
        self.assertEqual(m["matching_items"], 1)
        self.assertEqual(m["agreement_pct_of_weakly_labelled_items"], 50.0)
        self.assertEqual(m["weak_label_coverage_pct"], 50.0)
        self.assertEqual(m["queries_with_any_match"]["count"], 1)

    def test_gold_agreement_is_named_distinctly(self):
        m = ev.gold_intent_vs_weak_label_agreement(self.lists(), ["flight_delay", "OTHER", "OTHER"])
        self.assertEqual(m["measure"], "gold_intent_vs_weak_label_agreement")
        self.assertIn("not retrieval accuracy", m["definition"])
        self.assertEqual(m["matching_items"], 1)
        for measure in (m, ev.predicted_intent_vs_weak_label_consistency(self.lists(), ["x"] * 3)):
            self.assertNotIn("accuracy", measure["measure"])

    def test_no_weak_labels_gives_none(self):
        m = ev.gold_intent_vs_weak_label_agreement([[{"weak_intent": None, "similarity": 0.1}]], ["baggage"])
        self.assertIsNone(m["agreement_pct_of_weakly_labelled_items"])

    def test_requires_one_reference_per_query(self):
        with self.assertRaises(ValueError):
            ev.gold_intent_vs_weak_label_agreement(self.lists(), ["baggage"])

    def test_similarity_summary_counts_empty_queries(self):
        s = ev.similarity_summary(self.lists())
        self.assertEqual(s["queries_without_evidence"], 1)
        self.assertEqual(s["evidence_count_distribution"], {0: 1, 1: 1, 3: 1})
        self.assertEqual(s["top1_similarity"]["n"], 2)
        self.assertEqual(s["top1_similarity"]["max"], 0.5)

    def test_summarise_evidence(self):
        out = ev.summarise_evidence(evidence(2))
        self.assertEqual(out[0], {"rank": 1, "conversation_id": "conv_syn_0",
                                  "similarity": 0.5, "weak_intent": "baggage"})

    def test_k_rule(self):
        self.assertEqual(ev.k_selection_rule(45.0, 40.0), 3)
        self.assertEqual(ev.k_selection_rule(44.99, 40.0), 5)
        self.assertEqual(ev.k_selection_rule(38.52, 39.83), 5)
        self.assertEqual(ev.k_selection_rule(None, 40.0), 5)


class TestRandomRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.retrieve import Retriever
        cls.corpus = pd.DataFrame({
            "conversation_id": [f"conv_r{i}" for i in range(8)],
            "tweet_id": list(range(8)),
            "customer_author_id": [f"70{i}" for i in range(8)],
            "customer_message": ["lost bag at airport", "lost bag at gate", "delayed flight today",
                                 "delayed flight on tarmac", "seat upgrade request", "seat upgrade denied",
                                 "refund fare request", "refund fare denied"],
            "brand_reply": [f"reply {i}" for i in range(8)],
            "weak_label": ["baggage"] * 8,
            "n_turns": [2] * 8, "n_brand_turns": [1] * 8, "customer_followups": [0] * 8,
        })
        cls.retriever = Retriever(cls.corpus)
        cls.queries = [{"conversation_id": "conv_r0", "customer_author_id": "701"},
                       {"conversation_id": "conv_r2", "customer_author_id": "703"}]

    def test_uses_per_query_seed(self):
        calls = []

        class Recorder:
            def retrieve_random(self, k, *, seed, exclude_conversation_id, exclude_customer_id):
                calls.append((k, seed, exclude_conversation_id, exclude_customer_id))
                return []

        ev.random_retrieval(Recorder(), self.queries, 5)
        self.assertEqual(calls, [(5, 42, "conv_r0", "701"), (5, 43, "conv_r2", "703")])

    def test_matches_direct_calls_and_is_reproducible(self):
        first = ev.random_retrieval(self.retriever, self.queries, 3)
        second = ev.random_retrieval(self.retriever, self.queries, 3)
        self.assertEqual(first, second)
        direct = self.retriever.retrieve_random(3, seed=43, exclude_conversation_id="conv_r2",
                                                exclude_customer_id="703")
        self.assertEqual(first[1], direct)

    def test_respects_exclusions(self):
        result = ev.random_retrieval(self.retriever, self.queries, 6)
        ids = [e["conversation_id"] for e in result[0]]
        self.assertNotIn("conv_r0", ids)
        self.assertNotIn("conv_r1", ids)


class TestGenerationRecords(unittest.TestCase):
    def test_parse_error_kinds_from_real_parser_messages(self):
        cases = {"not json": "invalid_json", "[1]": "non_object", '{"reply":"x"}': "missing_fields",
                 '{"reply":1,"needs_more_information":true}': "field_type",
                 '{"reply":"x","needs_more_information":"no"}': "field_type",
                 '{"reply":"x","needs_more_information":true,"evidence_used":["flight_delay"]}':
                     "invalid_evidence_used"}
        for text, kind in cases.items():
            with self.assertRaises(ValueError) as caught:
                parse_response(text)
            self.assertEqual(ev.parse_error_kind(str(caught.exception)), kind, text)
        self.assertIsNone(ev.parse_error_kind(None))
        self.assertEqual(ev.parse_error_kind("something new"), "other")

    def test_citation_validity(self):
        self.assertTrue(ev.citation_validity([1, 5], 5, True))
        self.assertTrue(ev.citation_validity([], 0, True))
        self.assertFalse(ev.citation_validity([6], 5, True))
        self.assertFalse(ev.citation_validity([1], 0, True))
        self.assertIsNone(ev.citation_validity([], 5, False))

    def test_max_copy_similarity(self):
        items = evidence(2)
        self.assertIsNone(ev.max_copy_similarity("", items))
        self.assertIsNone(ev.max_copy_similarity("some reply text here", []))
        self.assertEqual(ev.max_copy_similarity(items[1]["historical_brand_reply"], items), 1.0)

    def test_parsed_record(self):
        items = evidence(3)
        m = ev.generation_measures(llm_record(PARSED, items), "my bag never arrived", items,
                                   attempt=1, first_attempt_error=None)
        self.assertTrue(m["parsed"])
        self.assertIsNone(m["parse_error_kind"])
        self.assertEqual(m["evidence_used"], [1, 2])
        self.assertTrue(m["evidence_used_coerced"])
        self.assertTrue(m["citation_valid"])
        self.assertTrue(m["needs_more_information"])
        self.assertEqual(m["grounding_flag_codes"], [])
        self.assertEqual(m["latency_seconds"], 1.5)
        self.assertFalse(m["latency_missing"])

    def test_parse_failure_is_kept_as_a_row(self):
        items = evidence(3)
        record = llm_record("garbage", items)
        m = ev.generation_measures(record, "my bag never arrived", items,
                                   attempt=2, first_attempt_error="MemoryError: x")
        self.assertFalse(m["parsed"])
        self.assertEqual(m["parse_error_kind"], "invalid_json")
        self.assertEqual(m["grounding_flag_codes"], ["G1_empty_reply"])
        self.assertIsNone(m["needs_more_information"])
        self.assertIsNone(m["citation_valid"])
        self.assertEqual(m["attempt"], 2)
        self.assertEqual(m["first_attempt_error"], "MemoryError: x")

    def test_missing_latency_is_not_fabricated(self):
        items = evidence(3)
        record = llm_record(PARSED, items)
        del record["latency_seconds"]
        m = ev.generation_measures(record, "my bag never arrived", items, attempt=1, first_attempt_error=None)
        self.assertIsNone(m["latency_seconds"])
        self.assertTrue(m["latency_missing"])

    def test_missing_required_field_raises(self):
        items = evidence(3)
        record = llm_record(PARSED, items)
        del record["grounding_flags"]
        with self.assertRaises(ev.EvaluationIntegrityError):
            ev.generation_measures(record, "x", items, attempt=1, first_attempt_error=None)

    def test_recorded_flags_must_match_recomputed(self):
        items = evidence(3)
        record = llm_record(PARSED, items)
        record["grounding_flags"] = [{"code": "G4_action_claim", "detail": "tampered"}]
        with self.assertRaises(ev.EvaluationIntegrityError):
            ev.generation_measures(record, "my bag never arrived", items, attempt=1, first_attempt_error=None)

    def test_summary_counts_every_row(self):
        items = evidence(3)
        rows = [ev.generation_measures(llm_record(t, items), "my bag never arrived", items,
                                       attempt=1, first_attempt_error=None)
                for t in (PARSED, "garbage", PARSED)]
        s = ev.generation_summary(rows)
        self.assertEqual(s["rows"], 3)
        self.assertEqual(s["parse_failures"]["count"], 1)
        self.assertEqual(s["parse_failure_kinds"], {"invalid_json": 1})
        self.assertEqual(s["flags"]["G1_empty_reply"]["count"], 1)
        self.assertEqual(s["all_checks_passed"]["count"], 2)
        self.assertEqual(s["needs_more_information_true"]["total"], 2)
        self.assertIn("not reply correctness", s["note"])


class TestBaselines(unittest.TestCase):
    def test_rebuild_and_compare(self):
        items = evidence(3)
        rebuilt = ev.rebuild_baselines("my bag", items)
        self.assertEqual(rebuilt["baseline_b_template"]["reply"], TEMPLATE_BASELINE_REPLY)
        self.assertNotIn("@", rebuilt["baseline_a_echo"]["reply"])
        recorded = copy.deepcopy(rebuilt)
        for name in rebuilt:
            m = ev.baseline_measures(recorded[name], rebuilt[name])
            self.assertEqual(m["reply"], rebuilt[name]["reply"])
        self.assertIn("G7_excessive_copying",
                      ev.baseline_measures(recorded["baseline_a_echo"], rebuilt["baseline_a_echo"])
                      ["grounding_flag_codes"])

    def test_tampered_recording_raises(self):
        rebuilt = ev.rebuild_baselines("my bag", evidence(3))
        recorded = copy.deepcopy(rebuilt["baseline_b_template"])
        recorded["reply"] = "different"
        with self.assertRaises(ev.EvaluationIntegrityError):
            ev.baseline_measures(recorded, rebuilt["baseline_b_template"])

    def test_caveat_only_on_baseline_a(self):
        row = {"grounding_flag_codes": ["G7_excessive_copying"], "grounding_passed": False}
        self.assertIn("by construction", ev.baseline_summary([row], "baseline_a_echo")["caveat"])
        self.assertNotIn("caveat", ev.baseline_summary([row], "baseline_b_template"))


class TestEscalation(unittest.TestCase):
    def test_record_uses_frozen_rules(self):
        items = evidence(2, top=0.10)
        result = ev.escalation_record("baggage", 0.9, items,
                                      llm_record(PARSED.replace("true", "false"), items), THRESHOLDS)
        self.assertEqual(result["decision"], "escalate")
        self.assertEqual(result["reason_codes"], ["weak_grounding_for_substantive_reply"])
        self.assertEqual(result["policy_version"], "esc-v1")

    def test_parse_failure_record(self):
        items = evidence(5)
        result = ev.escalation_record("baggage", 0.9, items, llm_record("garbage", items), THRESHOLDS)
        self.assertEqual(result["reason_codes"], ["generation_unusable"])

    def test_default_thresholds_are_the_frozen_values(self):
        items = evidence(5)
        result = ev.escalation_record("baggage", 0.9, items, llm_record(PARSED, items))
        self.assertEqual(result["thresholds"], {"intent_confidence": 0.19128133486537377,
                                                "top1_similarity": 0.198143, "min_evidence_count": 2})

    def test_summary(self):
        def row(decision, codes, predicted, gold, batch="b01", info=()):
            return {"decision": decision, "reason_codes": list(codes), "informational_codes": list(info),
                    "predicted_intent": predicted, "gold_intent": gold, "batch": batch,
                    "classifier_correct": predicted == gold,
                    "gold_predictable": gold in ev.PREDICTABLE_INTENTS}
        rows = [
            row("auto_handle", [], "baggage", "baggage", info=["awaiting_customer_details"]),
            row("escalate", ["low_intent_confidence"], "baggage", "OTHER"),
            row("escalate", ["low_intent_confidence", "weak_grounding_for_substantive_reply"],
                "praise_and_compliment", "praise_and_compliment", batch="b02"),
            row("auto_handle", [], "praise_and_compliment", "flight_delay", batch="b02"),
        ]
        s = ev.escalation_summary(rows)
        self.assertEqual(s["escalate"]["count"], 2)
        self.assertEqual(s["auto_handle"]["count"], 2)
        self.assertEqual(s["reason_code_counts"]["low_intent_confidence"]["count"], 2)
        self.assertEqual(s["reason_code_patterns"], {
            "none": 2, "low_intent_confidence": 1,
            "low_intent_confidence+weak_grounding_for_substantive_reply": 1})
        self.assertEqual(s["informational_code_counts"], {"awaiting_customer_details": 1})
        self.assertEqual(s["escalation_rate_by_gold_intent"]["OTHER"]["pct"], 100.0)
        self.assertNotIn("UNCLEAR", s["escalation_rate_by_gold_intent"])
        self.assertEqual(s["predicted_praise_weak_grounding"], {"count": 1, "total": 2, "pct": 50.0})
        self.assertEqual(s["escalation_rate_when_classifier_correct"]["total"], 2)
        self.assertEqual(s["escalation_rate_gold_label_not_predictable"]["count"], 1)
        self.assertIn("descriptive", s["note"])
        self.assertEqual(len(ev._subsets(rows)["independent_b02_100"]), 2)
        self.assertEqual(len(ev._subsets(rows)["all_248"]), 4)


class TestHumanEvidence(unittest.TestCase):
    report = {
        "measurement_type": "intra-annotator short-gap test-retest agreement",
        "n": 45, "gap": "approximately 12-14 hours",
        "agreement": {"raw_agreement_pct": 84.4, "cohens_kappa": 0.8281},
        "bootstrap": {"ci_lower": 0.7039, "ci_upper": 0.9254, "resamples": 2000, "seed": 44},
    }

    def test_reported_as_intra_annotator(self):
        h = ev.human_agreement_evidence(self.report)
        self.assertEqual((h["n"], h["raw_agreement_pct"], h["cohens_kappa"], h["kappa_ci_95"]),
                         (45, 84.4, 0.8281, [0.7039, 0.9254]))
        self.assertIn("not inter-annotator", h["measurement"])
        self.assertFalse(h["labels_used_as_golden"])

    def test_rejects_other_measurement_types(self):
        for kind in ("inter-annotator agreement", "", "test-retest"):
            with self.assertRaises(ev.EvaluationIntegrityError, msg=kind):
                ev.human_agreement_evidence({**self.report, "measurement_type": kind})


class TestRowAssembly(unittest.TestCase):
    def build(self):
        items = evidence(3)
        message = "my bag never arrived"
        record = llm_record(PARSED, items, message)
        golden = {"annotation_id": "syn_0001", "batch": "b02", "conversation_id": "conv_syn_q",
                  "text": message, "primary_intent": "SYNTHETIC_GOLD", "secondary_intent": "",
                  "ambiguous": "no"}
        return ev.assemble_row(
            golden=golden,
            classifier={"predicted_intent": "baggage", "confidence": 0.8, "majority_baseline": "baggage"},
            evidence=ev.summarise_evidence(items),
            random_evidence=[],
            generation=ev.generation_measures(record, message, items, attempt=1, first_attempt_error=None),
            baselines={k: ev.baseline_measures(v, v) for k, v in ev.rebuild_baselines(message, items).items()},
            escalation=ev.escalation_record("baggage", 0.8, items, record, THRESHOLDS),
        )

    def test_gold_is_isolated_in_its_own_section(self):
        row = self.build()
        self.assertEqual(row["gold"]["primary_intent"], "SYNTHETIC_GOLD")
        self.assertIsNone(row["gold"]["secondary_intent"])
        rest = json.dumps({k: v for k, v in row.items() if k != "gold"})
        self.assertNotIn("SYNTHETIC_GOLD", rest)
        self.assertEqual(set(row), {"annotation_id", "batch", "conversation_id",
                                    "customer_message", "gold", "system", "baselines"})
        self.assertEqual(set(row["system"]), {"classifier", "retrieval", "generation", "escalation"})

    def test_deterministic(self):
        self.assertEqual(json.dumps(self.build(), sort_keys=True), json.dumps(self.build(), sort_keys=True))

    def test_unknown_gold_marked_unpredictable(self):
        self.assertFalse(self.build()["gold"]["predictable_by_classifier"])


class TestIntegrityChecks(unittest.TestCase):
    def test_check_ids(self):
        ev.check_ids(["a", "b"], ["b", "a"], "x")
        with self.assertRaises(ev.EvaluationIntegrityError):
            ev.check_ids(["a", "b"], ["a", "a", "b"], "x")
        with self.assertRaises(ev.EvaluationIntegrityError):
            ev.check_ids(["a", "b"], ["a"], "x")
        with self.assertRaises(ev.EvaluationIntegrityError):
            ev.check_ids(["a"], ["a", "c"], "x")

    def test_check_gold_labels(self):
        ok = pd.DataFrame({"annotation_id": ["s1", "s2"], "primary_intent": ["baggage", "OTHER"]})
        ev.check_gold_labels(ok)
        for bad in ("", "not_a_label"):
            frame = ok.assign(primary_intent=["baggage", bad])
            with self.assertRaises(ev.EvaluationIntegrityError, msg=bad):
                ev.check_gold_labels(frame)
        with self.assertRaises(ev.EvaluationIntegrityError):
            ev.check_gold_labels(ok.assign(annotation_id=["s1", "s1"]))

    def test_frozen_configuration_holds(self):
        self.assertEqual(ev.check_frozen_configuration()["prompt_version"], "p2")

    def test_configuration_drift_raises(self):
        with mock.patch.object(ev.config, "RETRIEVAL_TOP_K", 3):
            with self.assertRaises(ev.EvaluationIntegrityError):
                ev.check_frozen_configuration()

    def record(self, **overrides):
        return {"annotation_id": "s1", "row_error": None, "llm": {}, "model": "llama3.1:8b",
                "prompt_version": "p2", "retrieval_k": 5, "temperature": 0.0, "seed": 42, **overrides}

    def test_record_configuration(self):
        ev.check_record_configuration([self.record()])
        for overrides in ({"prompt_version": "p1"}, {"model": "other"}, {"row_error": "boom"}):
            with self.assertRaises(ev.EvaluationIntegrityError, msg=overrides):
                ev.check_record_configuration([self.record(**overrides)])
        missing = self.record()
        del missing["llm"]
        with self.assertRaises(ev.EvaluationIntegrityError):
            ev.check_record_configuration([missing])

    def test_check_merge(self):
        first = [{"annotation_id": "a", "row_error": None, "v": 1},
                 {"annotation_id": "b", "row_error": "MemoryError", "v": 0}]
        retry = [{"annotation_id": "b", "row_error": None, "v": 2, "attempt": 2, "first_attempt_error": "MemoryError"}]
        final = [{**first[0], "attempt": 1, "first_attempt_error": None}, dict(retry[0])]
        ev.check_merge(first, retry, final)
        tampered = copy.deepcopy(final)
        tampered[1]["v"] = 99
        with self.assertRaises(ev.EvaluationIntegrityError):
            ev.check_merge(first, retry, tampered)
        with self.assertRaises(ev.EvaluationIntegrityError):
            ev.check_merge(first, [], final)

    def test_check_predictions(self):
        ev.check_predictions(["a"], ["baggage"], ["baggage"])
        with self.assertRaises(ev.EvaluationIntegrityError):
            ev.check_predictions(["a"], ["OTHER"], ["OTHER"])
        with self.assertRaises(ev.EvaluationIntegrityError):
            ev.check_predictions(["a"], ["baggage"], ["flight_delay"])

    def test_check_retrieval_matches(self):
        now = [{"conversation_id": "c1", "similarity": 0.5}]
        ev.check_retrieval_matches("s1", now, [dict(now[0])])
        with self.assertRaises(ev.EvaluationIntegrityError):
            ev.check_retrieval_matches("s1", now, [{"conversation_id": "c2", "similarity": 0.5}])
        with self.assertRaises(ev.EvaluationIntegrityError):
            ev.check_retrieval_matches("s1", now, [])


class TestIsolationCheck(unittest.TestCase):
    calibration = {"development_pool": {
        "golden_identifier_hits": {"corpus": {"a": 0}, "pool": {"b": 0}},
        "golden_labels_read": False}}

    def corpus(self, conversation="conv_iso_1", customer="990001",
               message="a completely synthetic message about zebras and telescopes"):
        return pd.DataFrame({"conversation_id": [conversation, "conv_iso_2"],
                             "customer_author_id": [customer, "990002"],
                             "customer_message": [message, "another synthetic message about kites"]})

    def test_clean_frames_pass(self):
        corpus = self.corpus()
        audit = ev.check_isolation(corpus, corpus.iloc[:1], self.calibration)
        self.assertEqual(audit["calibration_golden_hits"], 0)

    def test_golden_conversation_raises(self):
        from src.leakage import golden_conversation_ids
        corpus = self.corpus(conversation=sorted(golden_conversation_ids())[0])
        with self.assertRaises(ev.EvaluationIntegrityError):
            ev.check_isolation(corpus, corpus.iloc[1:], self.calibration)

    def test_golden_customer_raises(self):
        from src.leakage import golden_customer_ids
        corpus = self.corpus(customer=sorted(golden_customer_ids())[0])
        with self.assertRaises(ev.EvaluationIntegrityError):
            ev.check_isolation(corpus, corpus.iloc[1:], self.calibration)

    def test_training_outside_corpus_raises(self):
        corpus = self.corpus()
        training = pd.DataFrame({"conversation_id": ["conv_iso_elsewhere"]})
        with self.assertRaises(ev.EvaluationIntegrityError):
            ev.check_isolation(corpus, training, self.calibration)

    def test_contaminated_calibration_raises(self):
        corpus = self.corpus()
        for pool in ({"golden_identifier_hits": {"pool": {"b": 1}}, "golden_labels_read": False},
                     {"golden_identifier_hits": {"pool": {"b": 0}}, "golden_labels_read": True}):
            with self.assertRaises(ev.EvaluationIntegrityError):
                ev.check_isolation(corpus, corpus.iloc[:1], {"development_pool": pool})


class TestGoldenAccessGuard(unittest.TestCase):
    def test_run_refuses_without_confirmation_before_any_io(self):
        with mock.patch.object(pd, "read_csv", side_effect=AssertionError("golden read")), \
             mock.patch.object(ev, "check_frozen_configuration", side_effect=AssertionError("ran")):
            with self.assertRaises(ev.GoldenAccessError):
                ev.run_golden_evaluation()
            with self.assertRaises(ev.GoldenAccessError):
                ev.run_golden_evaluation(confirm="yes")

    def test_loader_refuses_without_confirmation(self):
        with mock.patch.object(pd, "read_csv", side_effect=AssertionError("golden read")):
            for value in (False, None, 1, "true"):
                with self.assertRaises(ev.GoldenAccessError, msg=repr(value)):
                    ev.load_golden(confirm=value)

    def test_cli_refuses_without_flag(self):
        with mock.patch.object(ev, "run_golden_evaluation", side_effect=AssertionError("ran")):
            self.assertEqual(ev.main([]), 2)
            self.assertEqual(ev.main(["--confirm"]), 2)

    def test_golden_file_is_only_touched_by_loader_and_hashing(self):
        tree = ast.parse(Path(ev.__file__).read_text(encoding="utf-8"))
        users = set()
        for function in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
            for node in ast.walk(function):
                if isinstance(node, ast.Attribute) and node.attr == "GOLDEN_SET":
                    users.add(function.name)
        self.assertEqual(users, {"load_golden", "provenance"})

    def test_pure_layer_never_reads_files(self):
        source = Path(ev.__file__).read_text(encoding="utf-8")
        pure = source.split("# --------------------------------------------------------------------------- layer 2")[0]
        for token in ("read_csv", "read_parquet", "read_text", "open(", "read_jsonl("):
            self.assertNotIn(token, pure)

    def test_this_suite_never_confirms_a_golden_run(self):
        own = Path(__file__).read_text(encoding="utf-8")
        self.assertNotIn("confirm" + "=True", own)
        self.assertNotIn("--confirm-golden" + "-run", own)


if __name__ == "__main__":
    unittest.main(verbosity=2)
