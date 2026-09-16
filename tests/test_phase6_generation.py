from __future__ import annotations

import unittest

import pandas as pd

from src import config
from src.generate_reply import (
    build_prompt,
    check_grounding,
    generate_baseline_echo,
    generate_baseline_template,
    generate_llm,
    parse_response,
    sanitise_evidence_text,
)
from src.llm import _cache_key


def fake_evidence(n: int = 2) -> list[dict]:
    return [
        {
            "rank": i + 1,
            "conversation_id": f"conv_e{i}",
            "tweet_id": i,
            "similarity": 0.3 - 0.05 * i,
            "historical_customer_message": f"@AmericanAir my bag is missing at DFW case {i}",
            "historical_brand_reply": f"@12345{i} Sorry about that. Please share your record locator via DM.",
            "historical_weak_intent": "baggage",
            "n_turns": 2,
            "n_brand_turns": 1,
            "customer_followups": 0,
            "evidence_shape": "two_turn",
            "retrieved_by": "tfidf",
        }
        for i in range(n)
    ]


def fake_complete(text: str):
    def _call(prompt, **kwargs):
        return {"text": text, "model": "mock", "latency_seconds": 0.01, "from_cache": False}
    return _call


class TestResponseParsing(unittest.TestCase):
    def test_parses_valid_response(self):
        parsed = parse_response(
            '{"reply":"hello","needs_more_information":true,"evidence_used":[1,2]}')
        self.assertEqual(parsed["reply"], "hello")
        self.assertTrue(parsed["needs_more_information"])
        self.assertEqual(parsed["evidence_used"], [1, 2])

    def test_evidence_used_defaults_to_empty(self):
        parsed = parse_response('{"reply":"hi","needs_more_information":false}')
        self.assertEqual(parsed["evidence_used"], [])

    def test_rejects_malformed_json(self):
        with self.assertRaises(ValueError):
            parse_response("not json at all")

    def test_rejects_non_object_json(self):
        with self.assertRaises(ValueError):
            parse_response('["a","b"]')

    def test_rejects_missing_fields(self):
        with self.assertRaises(ValueError):
            parse_response('{"reply":"hi"}')

    def test_rejects_wrong_types(self):
        with self.assertRaises(ValueError):
            parse_response('{"reply":"hi","needs_more_information":"yes"}')
        with self.assertRaises(ValueError):
            parse_response('{"reply":123,"needs_more_information":true}')
        with self.assertRaises(ValueError):
            parse_response('{"reply":"hi","needs_more_information":true,"evidence_used":"1"}')

    def test_malformed_output_is_reported_not_raised(self):
        result = generate_llm("my bag is lost", "baggage", fake_evidence(),
                              complete_fn=fake_complete("garbage"), use_cache=False)
        self.assertIsNotNone(result["parse_error"])
        self.assertFalse(result["grounding_passed"])
        self.assertEqual(result["grounding_flags"][0]["code"], "G1_empty_reply")


class TestGroundingChecks(unittest.TestCase):
    def setUp(self):
        self.evidence = fake_evidence()
        self.customer = "@AmericanAir my bag never arrived"

    def codes(self, reply: str) -> set[str]:
        return {f["code"] for f in check_grounding(reply, self.customer, self.evidence)}

    def test_clean_reply_passes(self):
        self.assertEqual(self.codes("Sorry about your bag. Could you share more details?"), set())

    def test_g1_empty_reply(self):
        self.assertIn("G1_empty_reply", self.codes("   "))

    def test_g2_unsupported_url(self):
        self.assertIn("G2_unsupported_url", self.codes("See https://aa.com/invented for help"))

    def test_g2_allows_url_present_in_evidence(self):
        evidence = fake_evidence(1)
        evidence[0]["historical_brand_reply"] += " https://t.co/REAL"
        codes = {f["code"] for f in check_grounding(
            "Please see https://t.co/REAL", self.customer, evidence)}
        self.assertNotIn("G2_unsupported_url", codes)

    def test_g3_currency_and_percentage(self):
        self.assertIn("G3_unsupported_number", self.codes("We will send you $200 today"))
        self.assertIn("G3_unsupported_number", self.codes("You get a 50% voucher"))

    def test_g3_phone_number(self):
        self.assertIn("G3_unsupported_number", self.codes("Call 800-555-1234 now"))

    def test_g3_allows_number_from_customer_message(self):
        codes = {f["code"] for f in check_grounding(
            "We looked at flight 5465 for you",
            "@AmericanAir flight 5465 was late", self.evidence)}
        self.assertNotIn("G3_unsupported_number", codes)

    def test_g4_action_claims(self):
        for reply in ("We have refunded your ticket.",
                      "I've rebooked you on the next flight.",
                      "Your reservation has been changed.",
                      "You will receive a voucher."):
            self.assertIn("G4_action_claim", self.codes(reply), reply)

    def test_g5_resolution_claim(self):
        self.assertIn("G5_resolution_claim", self.codes("This issue is now resolved."))

    def test_g6_leaked_identifier(self):
        self.assertIn("G6_leaked_identifier", self.codes("@123456 sorry about that"))

    def test_flags_carry_inspectable_detail(self):
        flags = check_grounding("See https://bad.example/x", self.customer, self.evidence)
        self.assertTrue(flags)
        self.assertTrue(all(f.get("detail") for f in flags))


class TestSanitisationAndPrompt(unittest.TestCase):
    def test_sanitise_strips_mentions(self):
        self.assertNotIn("@", sanitise_evidence_text("@123456 hello @AmericanAir there"))

    def test_prompt_contains_no_anonymised_identifier(self):
        prompt = build_prompt("@AmericanAir help me", "baggage", fake_evidence())
        self.assertNotRegex(prompt, r"@\d+")

    def test_prompt_handles_no_evidence(self):
        self.assertIn("no similar historical cases", build_prompt("help", "OTHER", []))

    def test_prompt_never_mentions_gold_labels(self):
        prompt = build_prompt("help me", "baggage", fake_evidence())
        self.assertNotIn("gold", prompt.lower())


class TestBaselines(unittest.TestCase):
    def test_template_baseline_is_fixed_and_grounded(self):
        first = generate_baseline_template("my bag is lost", fake_evidence())
        second = generate_baseline_template("a totally different message", fake_evidence())
        self.assertEqual(first["reply"], second["reply"])
        self.assertTrue(first["grounding_passed"])
        self.assertTrue(first["needs_more_information"])

    def test_echo_baseline_returns_sanitised_top1(self):
        result = generate_baseline_echo("my bag is lost", fake_evidence())
        self.assertIn("record locator", result["reply"])
        self.assertNotRegex(result["reply"], r"@\d+")
        self.assertEqual(result["evidence_used"], [1])

    def test_echo_baseline_handles_no_evidence(self):
        result = generate_baseline_echo("hello", [])
        self.assertEqual(result["reply"], "")
        self.assertFalse(result["grounding_passed"])


class TestCacheAndDeterminism(unittest.TestCase):
    def test_cache_key_is_stable_and_input_sensitive(self):
        self.assertEqual(_cache_key("p", "m", 0.0, 42, True), _cache_key("p", "m", 0.0, 42, True))
        self.assertNotEqual(_cache_key("p", "m", 0.0, 42, True), _cache_key("p", "m", 0.0, 43, True))
        self.assertNotEqual(_cache_key("p", "m", 0.0, 42, True), _cache_key("q", "m", 0.0, 42, True))

    def test_repeated_generation_is_reproducible(self):
        payload = ('{"reply":"Sorry about that, please share more detail.",'
                   '"needs_more_information":true,"evidence_used":[1]}')
        call = fake_complete(payload)
        first = generate_llm("bag lost", "baggage", fake_evidence(), complete_fn=call, use_cache=False)
        second = generate_llm("bag lost", "baggage", fake_evidence(), complete_fn=call, use_cache=False)
        self.assertEqual(first["reply"], second["reply"])
        self.assertEqual(first["grounding_flags"], second["grounding_flags"])

    def test_tests_never_reach_ollama(self):
        payload = '{"reply":"ok","needs_more_information":false,"evidence_used":[]}'
        result = generate_llm("x", "baggage", fake_evidence(),
                              complete_fn=fake_complete(payload), use_cache=False)
        self.assertEqual(result["model"], "mock")


class TestGoldenIsolation(unittest.TestCase):
    def test_no_golden_text_reaches_a_prompt(self):
        golden = pd.read_csv(config.GOLDEN_SET, dtype=str, keep_default_na=False,
                             encoding="utf-8-sig")
        prompt = build_prompt("my bag is lost", "baggage", fake_evidence())
        self.assertFalse(any(text and text in prompt for text in golden["text"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
