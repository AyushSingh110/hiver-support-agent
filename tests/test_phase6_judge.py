"""Phase 6E judge tests. Synthetic fixtures only: no gold labels, no model calls."""
from __future__ import annotations

import hashlib
import json
import unittest

from src import judge as jd
from src import llm

VALID = {"relevance": 4, "groundedness": 3, "helpfulness": 5,
         "information_request_appropriateness": 2, "claim_safety": 1,
         "rationale": "Addresses the bag issue but invents a refund."}


def evidence(n: int = 2) -> list[dict]:
    return [{"rank": i + 1, "similarity": 0.4,
             "historical_customer_message": f"@AmericanAir my bag is lost case {i}",
             "historical_brand_reply": f"@55555{i} Please DM your record locator."}
            for i in range(n)]


def row(evaluation_id: str, replies: dict | None = None) -> dict:
    return {"evaluation_id": evaluation_id,
            "customer_message": "@AmericanAir my bag never arrived",
            "evidence": evidence(),
            "replies": replies or {"llm_grounded": "LLM reply", "baseline_b_template": "B reply",
                                   "baseline_a_echo": "A reply"}}


def item(reply: str = "We are looking into your bag.", system: str = "llm_grounded") -> dict:
    return {"item_id": "j0001", "evaluation_id": "e1", "system": system,
            "customer_message": "@AmericanAir my bag never arrived", "evidence": evidence(),
            "reply": reply}


class Fake:
    """Scripted completion: each call pops the next outcome (text or exception)."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.prompts = []

    def __call__(self, prompt, **kwargs):
        self.prompts.append((prompt, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return {"text": outcome, "model": kwargs.get("model"), "latency_seconds": 0.5,
                "from_cache": False}


class TestPrompt(unittest.TestCase):
    def test_contains_inputs_and_rules(self):
        prompt = jd.build_judge_prompt("@AmericanAir my bag never arrived", evidence(), "Reply text here")
        self.assertIn("my bag never arrived", prompt)
        self.assertIn("Reply text here", prompt)
        self.assertIn("Please DM your record locator.", prompt)
        for phrase in ("Do not assume any airline policy", "do not invent facts",
                       "not facts about this customer", "independently", "use the whole scale",
                       "Return only this JSON object"):
            self.assertIn(phrase, prompt)
        for dimension in jd.DIMENSIONS:
            self.assertIn(f"{dimension}:", prompt)
        self.assertEqual(prompt.count("5 = "), 5)
        self.assertEqual(prompt.count("1 = "), 5)

    def test_sanitises_identifiers(self):
        prompt = jd.build_judge_prompt("@123456 @AmericanAir bag", evidence(), "reply")
        self.assertNotRegex(prompt, r"@\d+")

    def test_never_mentions_systems_labels_or_escalation(self):
        prompt = jd.build_judge_prompt("my bag", evidence(), "reply").lower()
        forbidden = ["baseline", "llm_grounded", "template", "echo", "proposed", "gold",
                     "primary_intent", "predicted", "intent", "escalat", "confidence",
                     "golden", "g1_", "g7_", "grounding_flag", "system:"]
        for word in forbidden:
            self.assertNotIn(word, prompt, word)

    def test_identical_format_for_every_system(self):
        prompts = [jd.build_judge_prompt(r["customer_message"], r["evidence"], "same reply")
                   for r in (row("a"), row("b"))]
        self.assertEqual(prompts[0], prompts[1])

    def test_zero_shot(self):
        self.assertEqual(jd.JUDGE_TEMPLATE.template.count("CANDIDATE REPLY"), 1)
        self.assertEqual(jd.JUDGE_TEMPLATE.template.count("CUSTOMER MESSAGE"), 1)

    def test_prompt_hash_is_deterministic(self):
        expected = hashlib.sha256(jd.JUDGE_TEMPLATE.template.encode("utf-8")).hexdigest()
        self.assertEqual(jd.JUDGE_PROMPT_SHA256, expected)
        self.assertEqual(jd.JUDGE_PROMPT_VERSION, "j1")

    def test_braces_and_dollars_in_inputs_are_literal(self):
        prompt = jd.build_judge_prompt("cost me $200 {evidence}", evidence(), "reply with $x {y}")
        self.assertIn("$200 {evidence}", prompt)
        self.assertIn("reply with $x {y}", prompt)


class TestParser(unittest.TestCase):
    def parse(self, **changes):
        payload = {**VALID, **changes}
        return jd.parse_judgement(json.dumps(payload))

    def test_valid(self):
        parsed = jd.parse_judgement(json.dumps(VALID))
        self.assertEqual(parsed, VALID)

    def test_rejects_malformed_json(self):
        for text in ("not json", "", "{", None):
            with self.assertRaises(ValueError, msg=repr(text)):
                jd.parse_judgement(text)

    def test_rejects_non_object(self):
        for text in ("[1,2]", "5", '"text"', "null"):
            with self.assertRaises(ValueError, msg=text):
                jd.parse_judgement(text)

    def test_rejects_missing_keys(self):
        for key in VALID:
            payload = {k: v for k, v in VALID.items() if k != key}
            with self.assertRaises(ValueError, msg=key):
                jd.parse_judgement(json.dumps(payload))

    def test_rejects_extra_keys(self):
        with self.assertRaises(ValueError):
            self.parse(overall=5)

    def test_rejects_wrong_score_types(self):
        for value in ("4", 4.0, 3.5, True, False, None, [4], {"score": 4}):
            with self.assertRaises(ValueError, msg=repr(value)):
                self.parse(relevance=value)

    def test_rejects_out_of_range(self):
        for value in (0, 6, -1, 10):
            with self.assertRaises(ValueError, msg=value):
                self.parse(claim_safety=value)

    def test_accepts_every_boundary_score(self):
        for value in jd.SCORES:
            self.assertEqual(self.parse(helpfulness=value)["helpfulness"], value)

    def test_rationale_limits(self):
        self.assertEqual(self.parse(rationale=" ".join(["w"] * 40))["rationale"], " ".join(["w"] * 40))
        for value in (" ".join(["w"] * 41), "", "   ", None, 5, ["text"]):
            with self.assertRaises(ValueError, msg=repr(value)):
                self.parse(rationale=value)

    def test_rationale_is_stripped(self):
        self.assertEqual(self.parse(rationale="  fine  ")["rationale"], "fine")


class TestPairing(unittest.TestCase):
    def test_one_item_per_row_and_system(self):
        items = jd.build_judge_items([row("e1"), row("e2")])
        self.assertEqual(len(items), 6)
        self.assertEqual({(i["evaluation_id"], i["system"]) for i in items},
                         {(e, s) for e in ("e1", "e2") for s in jd.SYSTEMS})
        self.assertEqual([i["item_id"] for i in items], [f"j{n:04d}" for n in range(1, 7)])

    def test_shuffle_is_deterministic_and_seeded(self):
        rows = [row(f"e{n}") for n in range(10)]
        order = lambda seed: [(i["evaluation_id"], i["system"]) for i in jd.build_judge_items(rows, seed=seed)]
        self.assertEqual(order(42), order(42))
        self.assertNotEqual(order(42), order(7))
        unshuffled = [(f"e{n}", s) for n in range(10) for s in jd.SYSTEMS]
        self.assertNotEqual(order(42), unshuffled)
        self.assertEqual(sorted(order(42)), sorted(unshuffled))

    def test_system_identity_stays_outside_the_prompt(self):
        replies = {"llm_grounded": "Sorry about the bag.", "baseline_b_template": "Sorry about the bag.",
                   "baseline_a_echo": "Sorry about the bag."}
        items = jd.build_judge_items([row("e1", replies)])
        fake = Fake(*[json.dumps(VALID)] * 3)
        records = jd.run_judge(items, complete_fn=fake, use_cache=False)
        self.assertEqual(len({p for p, _ in fake.prompts}), 1)
        for prompt, _ in fake.prompts:
            for system in jd.SYSTEMS:
                self.assertNotIn(system, prompt)
        self.assertEqual({r["system"] for r in records}, set(jd.SYSTEMS))

    def test_missing_system_reply_raises(self):
        with self.assertRaises(KeyError):
            jd.build_judge_items([row("e1", {"llm_grounded": "x"})])

    def test_duplicate_rows_raise(self):
        with self.assertRaises(ValueError):
            jd.build_judge_items([row("e1"), row("e1")])


class TestJudging(unittest.TestCase):
    def test_judged_record(self):
        fake = Fake(json.dumps(VALID))
        record = jd.judge_item(item(), complete_fn=fake, use_cache=False)
        self.assertEqual(record["status"], jd.JUDGED)
        self.assertEqual(record["scores"], {d: VALID[d] for d in jd.DIMENSIONS})
        self.assertEqual(record["rationale"], VALID["rationale"])
        self.assertEqual(record["judge_prompt_sha256"], jd.JUDGE_PROMPT_SHA256)
        self.assertEqual(fake.prompts[0][1]["model"], "qwen2.5:7b")
        self.assertFalse(fake.prompts[0][1]["use_cache"])

    def test_empty_replies_are_not_judged(self):
        for reply in ("", "   ", None):
            fake = Fake()
            record = jd.judge_item(item(reply=reply), complete_fn=fake)
            self.assertEqual(record["status"], jd.NOT_JUDGED_EMPTY_REPLY)
            self.assertEqual(fake.prompts, [])

    def test_malformed_output_is_recorded_and_not_retried(self):
        fake = Fake('{"relevance": "high"}')
        records = jd.run_judge([item()], complete_fn=fake, use_cache=False)
        self.assertEqual(len(fake.prompts), 1)
        self.assertEqual(records[0]["status"], jd.JUDGE_PARSE_FAILURE)
        self.assertEqual(records[0]["raw_output"], '{"relevance": "high"}')
        self.assertIsNone(records[0]["scores"])
        self.assertEqual(records[0]["attempt"], 1)

    def test_infrastructure_error_retried_once_and_succeeds(self):
        fake = Fake(llm.LLMError("HTTP 500"), json.dumps(VALID))
        records = jd.run_judge([item()], complete_fn=fake, use_cache=False)
        self.assertEqual(len(fake.prompts), 2)
        self.assertEqual(records[0]["status"], jd.JUDGED)
        self.assertEqual(records[0]["attempt"], 2)
        self.assertIn("HTTP 500", records[0]["first_attempt_error"])

    def test_infrastructure_error_retried_only_once(self):
        fake = Fake(MemoryError("x"), OSError("y"))
        records = jd.run_judge([item()], complete_fn=fake, use_cache=False)
        self.assertEqual(len(fake.prompts), 2)
        self.assertEqual(records[0]["status"], jd.JUDGE_INFRASTRUCTURE_ERROR)
        self.assertIn("OSError", records[0]["error"])
        self.assertIn("MemoryError", records[0]["first_attempt_error"])

    def test_programming_errors_are_not_swallowed(self):
        with self.assertRaises(KeyError):
            jd.judge_item(item(), complete_fn=Fake(KeyError("bug")))

    def test_no_row_is_dropped(self):
        items = [dict(item(), item_id=f"j{n}", evaluation_id=f"e{n}") for n in range(4)]
        items[1]["reply"] = ""
        fake = Fake(json.dumps(VALID), "garbage", llm.LLMError("down"), llm.LLMError("down"))
        records = jd.run_judge(items, complete_fn=fake, use_cache=False)
        self.assertEqual([r["item_id"] for r in records], [i["item_id"] for i in items])
        self.assertEqual([r["status"] for r in records],
                         [jd.JUDGED, jd.NOT_JUDGED_EMPTY_REPLY, jd.JUDGE_PARSE_FAILURE,
                          jd.JUDGE_INFRASTRUCTURE_ERROR])


def record(evaluation_id, system, status=jd.JUDGED, **scores):
    return {"evaluation_id": evaluation_id, "system": system, "status": status,
            "scores": {d: scores.get(d, 3) for d in jd.DIMENSIONS} if status == jd.JUDGED else None}


class TestAggregation(unittest.TestCase):
    def test_dimension_summary(self):
        records = [record("e1", "llm_grounded", relevance=5),
                   record("e2", "llm_grounded", relevance=2),
                   record("e3", "llm_grounded", relevance=2),
                   record("e4", "llm_grounded", status=jd.NOT_JUDGED_EMPTY_REPLY),
                   record("e1", "baseline_b_template", relevance=1)]
        s = jd.dimension_summary(records, "llm_grounded")
        rel = s["dimensions"]["relevance"]
        self.assertEqual(rel["n"], 3)
        self.assertEqual(rel["mean"], 3.0)
        self.assertEqual(rel["median"], 2.0)
        self.assertEqual(rel["distribution"], {"1": 0, "2": 2, "3": 0, "4": 0, "5": 1})
        self.assertEqual(s["status"], {"items": 4, "judged": 3, "not_judged_empty_reply": 1,
                                       "judge_parse_failure": 0, "judge_infrastructure_error": 0})
        self.assertNotIn("overall", s)

    def test_empty_summary(self):
        s = jd.dimension_summary([], "baseline_a_echo")
        self.assertEqual(s["dimensions"]["relevance"], {"n": 0, "mean": None, "median": None,
                                                        "distribution": {str(k): 0 for k in jd.SCORES}})

    def test_paired_comparison(self):
        records = [
            record("e1", "llm_grounded", relevance=5), record("e1", "baseline_b_template", relevance=3),
            record("e2", "llm_grounded", relevance=2), record("e2", "baseline_b_template", relevance=4),
            record("e3", "llm_grounded", relevance=3), record("e3", "baseline_b_template", relevance=3),
            record("e4", "llm_grounded", status=jd.JUDGE_PARSE_FAILURE),
            record("e4", "baseline_b_template", relevance=5),
        ]
        c = jd.paired_comparison(records, "llm_grounded", "baseline_b_template", resamples=500)
        rel = c["dimensions"]["relevance"]
        self.assertEqual(c["pairs"], 3)
        self.assertEqual(c["unpaired"], {"llm_grounded": 0, "baseline_b_template": 1})
        self.assertEqual((rel["win"], rel["tie"], rel["loss"]), (1, 1, 1))
        self.assertEqual(rel["mean_difference"], 0.0)
        self.assertLessEqual(rel["interval_95"][0], 0.0)
        self.assertGreaterEqual(rel["interval_95"][1], 0.0)
        self.assertEqual(c["dimensions"]["claim_safety"]["tie"], 3)

    def test_paired_bootstrap_is_deterministic(self):
        records = [record(f"e{n}", s, relevance=(n % 5) + 1 if s == "llm_grounded" else 3)
                   for n in range(20) for s in ("llm_grounded", "baseline_b_template")]
        a = jd.paired_comparison(records, "llm_grounded", "baseline_b_template")
        b = jd.paired_comparison(records, "llm_grounded", "baseline_b_template")
        self.assertEqual(a, b)
        self.assertIn("2000 resamples, seed 42", a["dimensions"]["relevance"]["interval_method"])

    def test_descriptive_comparison_has_no_interval(self):
        records = [record("e1", "llm_grounded"), record("e1", "baseline_a_echo", relevance=1)]
        c = jd.paired_comparison(records, "llm_grounded", "baseline_a_echo", with_interval=False)
        self.assertEqual(c["kind"], "descriptive")
        self.assertNotIn("interval_95", c["dimensions"]["relevance"])
        self.assertEqual(c["dimensions"]["relevance"]["win"], 1)

    def test_no_shared_pairs(self):
        c = jd.paired_comparison([record("e1", "llm_grounded")], "llm_grounded", "baseline_b_template")
        self.assertEqual(c["pairs"], 0)
        self.assertIsNone(c["dimensions"]["relevance"]["mean_difference"])


class TestSanityDesign(unittest.TestCase):
    def test_six_sanity_cases_and_ten_probe_items(self):
        self.assertEqual(len(jd.SANITY_CASES), 6)
        self.assertEqual(len(jd.SANITY_CASES) + len(jd.PROBE_EXTRA_CASES), 10)
        targeted = {dim for case in jd.SANITY_CASES for dim in case["expect"]}
        self.assertEqual(targeted, set(jd.DIMENSIONS))

    def test_expectation_rules(self):
        self.assertTrue(jd._meets(4, ">=4"))
        self.assertFalse(jd._meets(3, ">=4"))
        self.assertTrue(jd._meets(2, "<=2"))
        self.assertFalse(jd._meets(3, "<=2"))

    def test_message_selection_is_deterministic_and_filtered(self):
        import pandas as pd
        corpus = pd.DataFrame({
            "conversation_id": [f"c{n}" for n in range(9)],
            "customer_author_id": [str(n) for n in range(9)],
            "weak_label": ["baggage", "baggage", "flight_delay", "flight_delay",
                           "praise_and_compliment", "praise_and_compliment", "baggage",
                           "flight_delay", "praise_and_compliment"],
            "customer_message": ["my bag is lost at the airport again today help"] * 2
                                + ["my flight is delayed by three hours at the gate now"] * 2
                                + ["thank you to the great crew on my flight this morning"] * 2
                                + ["short", "short", "short"],
        })
        a, b = jd.select_sanity_messages(corpus), jd.select_sanity_messages(corpus)
        self.assertEqual(a, b)
        self.assertEqual(set(a), set(jd.SANITY_SOURCES))
        self.assertTrue(all(v["customer_message"] != "short" for v in a.values()))

    def test_cli_requires_explicit_flag(self):
        self.assertEqual(jd.main([]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
