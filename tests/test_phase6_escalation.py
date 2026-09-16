from __future__ import annotations

import ast
import dataclasses
import unittest
from pathlib import Path

from src import escalate
from src.escalate import (
    AWAITING_CUSTOMER_DETAILS,
    EXPLICIT_HUMAN_REVIEW_POLICY,
    GENERATION_UNUSABLE,
    GROUNDING_CHECK_FAILED,
    INSUFFICIENT_EVIDENCE,
    INVALID_EVIDENCE_CITATION,
    LOW_INTENT_CONFIDENCE,
    RULE_ORDER,
    WEAK_GROUNDING_FOR_SUBSTANTIVE_REPLY,
    EscalationSignals,
    Thresholds,
    build_signals,
    decide,
)

THRESHOLDS = Thresholds(intent_confidence=0.40, top1_similarity=0.20, min_evidence_count=2)

CLEAN = dict(
    predicted_intent="baggage",
    intent_confidence=0.90,
    evidence_count=5,
    top1_similarity=0.60,
    reply="Sorry about your bag. Please DM us your record locator.",
    parse_error=None,
    grounding_flag_codes=(),
    needs_more_information=False,
    evidence_used=(1, 2),
)


def signals(**overrides) -> EscalationSignals:
    return EscalationSignals(**{**CLEAN, **overrides})


def unparsed(**overrides) -> EscalationSignals:
    return signals(**{"reply": "", "parse_error": "model did not return valid JSON",
                      "grounding_flag_codes": ("G1_empty_reply",),
                      "needs_more_information": None, "evidence_used": (), **overrides})


def codes(s: EscalationSignals) -> list[str]:
    return decide(s, THRESHOLDS)["reason_codes"]


class TestCleanInput(unittest.TestCase):
    def test_clean_input_auto_handles(self):
        result = decide(signals(), THRESHOLDS)
        self.assertEqual(result["decision"], "auto_handle")
        self.assertEqual(result["reason_codes"], [])
        self.assertEqual(result["informational_codes"], [])
        self.assertEqual(result["reason"], "No escalation rule fired.")

    def test_output_schema(self):
        result = decide(signals(), THRESHOLDS)
        self.assertEqual(set(result), {"decision", "reason_codes", "informational_codes",
                                       "reason", "signals", "thresholds", "policy_version"})
        self.assertEqual(result["policy_version"], "esc-v1")
        self.assertEqual(result["thresholds"], {"intent_confidence": 0.40,
                                                "top1_similarity": 0.20,
                                                "min_evidence_count": 2})
        self.assertEqual(result["signals"]["evidence_used"], [1, 2])


class TestEachRuleInIsolation(unittest.TestCase):
    def test_generation_unusable_on_parse_error(self):
        self.assertEqual(codes(unparsed()), [GENERATION_UNUSABLE])

    def test_generation_unusable_on_empty_reply(self):
        self.assertEqual(codes(signals(reply="   ")), [GENERATION_UNUSABLE])

    def test_generation_unusable_on_g1_flag(self):
        self.assertEqual(codes(signals(grounding_flag_codes=("G1_empty_reply",))),
                         [GENERATION_UNUSABLE])

    def test_each_claim_flag_escalates_alone(self):
        for flag in ("G2_unsupported_url", "G3_unsupported_number", "G4_action_claim",
                     "G5_resolution_claim", "G6_leaked_identifier", "G7_excessive_copying"):
            self.assertEqual(codes(signals(grounding_flag_codes=(flag,))),
                             [GROUNDING_CHECK_FAILED], flag)

    def test_invalid_citation_above_range(self):
        self.assertEqual(codes(signals(evidence_used=(1, 6))), [INVALID_EVIDENCE_CITATION])

    def test_invalid_citation_rank_zero(self):
        self.assertEqual(codes(signals(evidence_used=(0,))), [INVALID_EVIDENCE_CITATION])

    def test_insufficient_evidence(self):
        self.assertEqual(codes(signals(evidence_count=1, evidence_used=(1,))),
                         [INSUFFICIENT_EVIDENCE])

    def test_explicit_human_review_policy_intents(self):
        for intent in ("staff_and_service_complaint", "general_dissatisfaction"):
            self.assertEqual(codes(signals(predicted_intent=intent, intent_confidence=0.99)),
                             [EXPLICIT_HUMAN_REVIEW_POLICY], intent)

    def test_policy_covers_exactly_two_intents(self):
        self.assertEqual(escalate.HUMAN_REVIEW_POLICY_INTENTS,
                         {"staff_and_service_complaint", "general_dissatisfaction"})

    def test_other_predictable_intents_are_not_policy_routed(self):
        for intent in escalate.PREDICTABLE_INTENTS - escalate.HUMAN_REVIEW_POLICY_INTENTS:
            self.assertEqual(codes(signals(predicted_intent=intent)), [], intent)

    def test_low_intent_confidence(self):
        self.assertEqual(codes(signals(intent_confidence=0.10)), [LOW_INTENT_CONFIDENCE])

    def test_weak_grounding_for_substantive_reply(self):
        self.assertEqual(codes(signals(top1_similarity=0.05)),
                         [WEAK_GROUNDING_FOR_SUBSTANTIVE_REPLY])


class TestBoundaries(unittest.TestCase):
    def test_confidence_at_threshold_does_not_fire(self):
        self.assertNotIn(LOW_INTENT_CONFIDENCE, codes(signals(intent_confidence=0.40)))

    def test_confidence_just_below_threshold_fires(self):
        self.assertIn(LOW_INTENT_CONFIDENCE, codes(signals(intent_confidence=0.3999)))

    def test_similarity_at_threshold_does_not_fire(self):
        self.assertNotIn(WEAK_GROUNDING_FOR_SUBSTANTIVE_REPLY, codes(signals(top1_similarity=0.20)))

    def test_similarity_just_below_threshold_fires(self):
        self.assertIn(WEAK_GROUNDING_FOR_SUBSTANTIVE_REPLY, codes(signals(top1_similarity=0.1999)))

    def test_evidence_count_at_minimum_does_not_fire(self):
        self.assertNotIn(INSUFFICIENT_EVIDENCE, codes(signals(evidence_count=2)))

    def test_evidence_count_below_minimum_fires(self):
        self.assertIn(INSUFFICIENT_EVIDENCE, codes(signals(evidence_count=1, evidence_used=(1,))))

    def test_citation_at_top_of_range_is_valid(self):
        self.assertNotIn(INVALID_EVIDENCE_CITATION, codes(signals(evidence_count=5, evidence_used=(5,))))

    def test_citation_one_past_range_is_invalid(self):
        self.assertIn(INVALID_EVIDENCE_CITATION, codes(signals(evidence_count=5, evidence_used=(6,))))

    def test_zero_evidence(self):
        result = codes(signals(evidence_count=0, top1_similarity=None, evidence_used=()))
        self.assertEqual(result, [INSUFFICIENT_EVIDENCE, WEAK_GROUNDING_FOR_SUBSTANTIVE_REPLY])

    def test_any_citation_with_zero_evidence_is_invalid(self):
        self.assertIn(INVALID_EVIDENCE_CITATION,
                      codes(signals(evidence_count=0, top1_similarity=None, evidence_used=(1,))))


class TestNeedsMoreInformation(unittest.TestCase):
    def test_alone_never_escalates(self):
        result = decide(signals(needs_more_information=True), THRESHOLDS)
        self.assertEqual(result["decision"], "auto_handle")
        self.assertEqual(result["reason_codes"], [])
        self.assertEqual(result["informational_codes"], [AWAITING_CUSTOMER_DETAILS])
        self.assertEqual(result["reason"],
                         "No escalation rule fired. The reply asks the customer for more details.")

    def test_weak_retrieval_with_clarifying_question_auto_handles(self):
        result = decide(signals(top1_similarity=0.05, needs_more_information=True), THRESHOLDS)
        self.assertEqual(result["decision"], "auto_handle")
        self.assertEqual(result["informational_codes"], [AWAITING_CUSTOMER_DETAILS])

    def test_weak_retrieval_with_substantive_reply_escalates(self):
        result = decide(signals(top1_similarity=0.05, needs_more_information=False), THRESHOLDS)
        self.assertEqual(result["decision"], "escalate")
        self.assertEqual(result["informational_codes"], [])

    def test_informational_code_kept_alongside_escalation(self):
        result = decide(signals(needs_more_information=True, intent_confidence=0.10), THRESHOLDS)
        self.assertEqual(result["decision"], "escalate")
        self.assertEqual(result["informational_codes"], [AWAITING_CUSTOMER_DETAILS])

    def test_praise_is_not_exempt(self):
        result = decide(signals(predicted_intent="praise_and_compliment",
                                top1_similarity=0.05, needs_more_information=False), THRESHOLDS)
        self.assertEqual(result["reason_codes"], [WEAK_GROUNDING_FOR_SUBSTANTIVE_REPLY])


class TestParseFailureGating(unittest.TestCase):
    def test_parse_failure_suppresses_parsed_field_rules(self):
        # Low similarity would fire rule 7 if needs_more_information were read.
        result = decide(unparsed(top1_similarity=0.01), THRESHOLDS)
        self.assertEqual(result["reason_codes"], [GENERATION_UNUSABLE])
        self.assertEqual(result["informational_codes"], [])

    def test_parse_failure_still_allows_non_generation_rules(self):
        result = codes(unparsed(intent_confidence=0.05, predicted_intent="general_dissatisfaction"))
        self.assertEqual(result, [GENERATION_UNUSABLE, EXPLICIT_HUMAN_REVIEW_POLICY,
                                  LOW_INTENT_CONFIDENCE])

    def test_unparsed_reply_cannot_carry_model_fields(self):
        with self.assertRaises(ValueError):
            unparsed(needs_more_information=False)
        with self.assertRaises(ValueError):
            unparsed(evidence_used=(1,))


class TestMultipleReasons(unittest.TestCase):
    def test_all_rules_fire_in_fixed_order(self):
        s = signals(
            predicted_intent="staff_and_service_complaint",
            intent_confidence=0.05,
            evidence_count=1,
            top1_similarity=0.05,
            grounding_flag_codes=("G7_excessive_copying", "G2_unsupported_url"),
            evidence_used=(3,),
            reply="",
        )
        self.assertEqual(codes(s), list(RULE_ORDER))

    def test_order_does_not_depend_on_flag_order(self):
        a = decide(signals(grounding_flag_codes=("G7_excessive_copying", "G2_unsupported_url"),
                           intent_confidence=0.1), THRESHOLDS)
        b = decide(signals(grounding_flag_codes=("G2_unsupported_url", "G7_excessive_copying"),
                           intent_confidence=0.1), THRESHOLDS)
        self.assertEqual(a["reason_codes"], b["reason_codes"])
        self.assertEqual(a["reason"], b["reason"])

    def test_reason_text_follows_code_order(self):
        result = decide(signals(intent_confidence=0.1, top1_similarity=0.05,
                                grounding_flag_codes=("G4_action_claim",)), THRESHOLDS)
        self.assertEqual(result["reason_codes"],
                         [GROUNDING_CHECK_FAILED, LOW_INTENT_CONFIDENCE,
                          WEAK_GROUNDING_FOR_SUBSTANTIVE_REPLY])
        text = result["reason"]
        self.assertLess(text.index("G4_action_claim"), text.index("Intent confidence"))
        self.assertLess(text.index("Intent confidence"), text.index("Top retrieval similarity"))

    def test_rule_order_is_the_agreed_order(self):
        self.assertEqual(RULE_ORDER, (
            "generation_unusable", "grounding_check_failed", "invalid_evidence_citation",
            "insufficient_evidence", "explicit_human_review_policy", "low_intent_confidence",
            "weak_grounding_for_substantive_reply",
        ))


class TestValidation(unittest.TestCase):
    def test_rejects_unpredictable_intents(self):
        for intent in ("OTHER", "UNCLEAR", "non_support_commentary", "not_a_label", ""):
            with self.assertRaises(ValueError, msg=intent):
                signals(predicted_intent=intent)

    def test_rejects_unknown_grounding_flag(self):
        with self.assertRaises(ValueError):
            signals(grounding_flag_codes=("G8_new_check",))

    def test_rejects_out_of_range_confidence(self):
        for value in (-0.01, 1.01, True, "0.5", None):
            with self.assertRaises(ValueError, msg=repr(value)):
                signals(intent_confidence=value)

    def test_rejects_out_of_range_similarity(self):
        for value in (-0.01, 1.01, None, True):
            with self.assertRaises(ValueError, msg=repr(value)):
                signals(top1_similarity=value)

    def test_similarity_must_be_none_without_evidence(self):
        with self.assertRaises(ValueError):
            signals(evidence_count=0, top1_similarity=0.3, evidence_used=())

    def test_rejects_bad_evidence_count(self):
        for value in (-1, 2.0, True):
            with self.assertRaises(ValueError, msg=repr(value)):
                signals(evidence_count=value)

    def test_parsed_reply_needs_boolean_flag(self):
        with self.assertRaises(ValueError):
            signals(needs_more_information=None)

    def test_rejects_non_integer_citations(self):
        for value in (("1",), (True,), (1.0,)):
            with self.assertRaises(ValueError, msg=repr(value)):
                signals(evidence_used=value)

    def test_rejects_bad_evidence_intent_matches(self):
        for value in (-1, 6, True):
            with self.assertRaises(ValueError, msg=repr(value)):
                signals(evidence_intent_matches=value)

    def test_threshold_validation(self):
        with self.assertRaises(ValueError):
            Thresholds(intent_confidence=1.5, top1_similarity=0.2, min_evidence_count=2)
        with self.assertRaises(ValueError):
            Thresholds(intent_confidence=0.5, top1_similarity=-0.1, min_evidence_count=2)
        with self.assertRaises(ValueError):
            Thresholds(intent_confidence=0.5, top1_similarity=0.2, min_evidence_count=-1)

    def test_signals_are_immutable(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            signals().intent_confidence = 0.0


class TestRecordedButUnusedSignals(unittest.TestCase):
    def test_coercion_does_not_change_decision(self):
        self.assertEqual(decide(signals(evidence_used_coerced=True), THRESHOLDS)["decision"],
                         "auto_handle")

    def test_evidence_intent_matches_does_not_change_decision(self):
        a = decide(signals(evidence_intent_matches=0), THRESHOLDS)
        b = decide(signals(evidence_intent_matches=5), THRESHOLDS)
        self.assertEqual(a["reason_codes"], b["reason_codes"])
        self.assertEqual(a["signals"]["evidence_intent_matches"], 0)


class TestDeterminism(unittest.TestCase):
    def test_identical_input_gives_identical_output(self):
        s = signals(intent_confidence=0.123456, top1_similarity=0.0123,
                    grounding_flag_codes=("G3_unsupported_number",))
        self.assertEqual(decide(s, THRESHOLDS), decide(s, THRESHOLDS))
        rebuilt = signals(intent_confidence=0.123456, top1_similarity=0.0123,
                          grounding_flag_codes=["G3_unsupported_number"])
        self.assertEqual(decide(s, THRESHOLDS), decide(rebuilt, THRESHOLDS))

    def test_explicit_thresholds_are_used(self):
        strict = Thresholds(intent_confidence=0.95, top1_similarity=0.2, min_evidence_count=2)
        self.assertIn(LOW_INTENT_CONFIDENCE, decide(signals(), strict)["reason_codes"])
        self.assertNotIn(LOW_INTENT_CONFIDENCE, decide(signals(), THRESHOLDS)["reason_codes"])

    def test_default_thresholds_come_from_config(self):
        from src import config
        t = escalate.default_thresholds()
        self.assertEqual(t.intent_confidence, config.ESCALATION_MIN_INTENT_CONFIDENCE)
        self.assertEqual(t.top1_similarity, config.ESCALATION_MIN_RETRIEVAL_SIMILARITY)
        self.assertEqual(t.min_evidence_count, config.ESCALATION_MIN_EVIDENCE_COUNT)


class TestBuildSignals(unittest.TestCase):
    def evidence(self, n=5):
        return [{"rank": i + 1, "similarity": 0.5 - 0.05 * i,
                 "historical_weak_intent": "baggage" if i < 2 else None} for i in range(n)]

    def test_from_parsed_generation(self):
        generation = {"reply": "hello there", "parse_error": None, "needs_more_information": True,
                      "evidence_used": [1, 2], "evidence_used_coerced": True,
                      "grounding_flags": [{"code": "G2_unsupported_url", "detail": "x"}]}
        s = build_signals("baggage", 0.8, self.evidence(), generation)
        self.assertEqual(s.evidence_count, 5)
        self.assertEqual(s.top1_similarity, 0.5)
        self.assertEqual(s.grounding_flag_codes, ("G2_unsupported_url",))
        self.assertEqual(s.evidence_used, (1, 2))
        self.assertTrue(s.evidence_used_coerced)
        self.assertEqual(s.evidence_intent_matches, 2)

    def test_from_failed_generation_drops_model_fields(self):
        # generate_llm fills needs_more_information=False on parse failure; that must not leak.
        generation = {"reply": "", "parse_error": "bad json", "needs_more_information": False,
                      "evidence_used": [], "evidence_used_coerced": False,
                      "grounding_flags": [{"code": "G1_empty_reply", "detail": "reply is blank"}]}
        s = build_signals("baggage", 0.8, self.evidence(), generation)
        self.assertIsNone(s.needs_more_information)
        self.assertEqual(decide(s, THRESHOLDS)["reason_codes"], [GENERATION_UNUSABLE])

    def test_with_no_evidence(self):
        generation = {"reply": "Could you share more details?", "parse_error": None,
                      "needs_more_information": True, "evidence_used": [],
                      "grounding_flags": []}
        s = build_signals("baggage", 0.8, [], generation)
        self.assertIsNone(s.top1_similarity)
        self.assertEqual(decide(s, THRESHOLDS)["reason_codes"], [INSUFFICIENT_EVIDENCE])

    def test_flag_codes_match_generation_module(self):
        # Every code check_grounding can emit must be known here, or decide() would raise.
        from src.generate_reply import check_grounding
        evidence = [{"rank": 1, "similarity": 0.5,
                     "historical_customer_message": "my bag is lost",
                     "historical_brand_reply": "Sorry about that, please DM us your record locator today."}]
        reply = ("@123456 We have refunded you $50, see https://bad.example. This issue is now resolved. "
                 "Sorry about that, please DM us your record locator today.")
        emitted = {f["code"] for f in check_grounding(reply, "bag", evidence)}
        emitted |= {f["code"] for f in check_grounding("", "bag", evidence)}
        emitted |= {f["code"] for f in check_grounding(
            evidence[0]["historical_brand_reply"], "bag", evidence)}
        self.assertEqual(emitted, escalate.KNOWN_FLAGS)


class TestIsolation(unittest.TestCase):
    SOURCE = Path(escalate.__file__)

    def imported_modules(self) -> set[str]:
        tree = ast.parse(self.SOURCE.read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names |= {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                names.add(node.module or "")
                names |= {f"{node.module}.{alias.name}" for alias in node.names}
        return names

    def test_does_not_import_models_data_or_network(self):
        forbidden = ("llm", "generate_reply", "leakage", "retrieve", "classify_intent",
                     "build_corpus", "sklearn", "pandas", "urllib", "requests", "socket")
        for name in self.imported_modules():
            for bad in forbidden:
                self.assertNotIn(bad, name.split("."), f"escalate imports {name}")

    def test_has_no_file_io(self):
        source = self.SOURCE.read_text(encoding="utf-8")
        for token in ("open(", "read_text", "write_text", "read_csv", "read_parquet", "GOLDEN"):
            self.assertNotIn(token, source)

    def test_signals_have_no_gold_label_field(self):
        fields = {f.name for f in dataclasses.fields(EscalationSignals)}
        self.assertFalse({name for name in fields if "gold" in name or "label" in name})
        with self.assertRaises(TypeError):
            EscalationSignals(**CLEAN, eval_only_gold_intent="baggage")


if __name__ == "__main__":
    unittest.main(verbosity=2)
