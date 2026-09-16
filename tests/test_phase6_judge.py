"""Phase 6E judge tests (j3). Synthetic fixtures only: no gold labels, no model calls."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src import config, llm
from src import judge as jd

VALID = {"score": 4, "rationale": "Addresses the missing bag directly."}
OTHER_DIMENSION_WORDS = {
    "relevance": ("relevance",),
    "groundedness": ("groundedness",),
    "helpfulness": ("helpfulness",),
    "information_request_appropriateness": ("information_request_appropriateness",
                                            "information request appropriateness"),
    "claim_safety": ("claim_safety", "claim safety"),
}


def evidence(n: int = 2) -> list[dict]:
    return [{"rank": i + 1, "similarity": 0.4,
             "historical_customer_message": f"@AmericanAir my bag is lost case {i}",
             "historical_brand_reply": f"@55555{i} Please DM your record locator, case {i}."}
            for i in range(n)]


def row(evaluation_id: str, replies: dict | None = None) -> dict:
    return {"evaluation_id": evaluation_id,
            "customer_message": "@AmericanAir my bag never arrived",
            "evidence": evidence(),
            "replies": replies or {"llm_grounded": "LLM reply", "baseline_b_template": "B reply",
                                   "baseline_a_echo": "A reply"}}


def item(reply="We are looking into your bag.", system="llm_grounded", evaluation_id="e1",
         item_id="j0001") -> dict:
    return {"item_id": item_id, "evaluation_id": evaluation_id, "system": system,
            "customer_message": "@AmericanAir my bag never arrived", "evidence": evidence(),
            "reply": reply}


def ok(score: int = 4) -> str:
    return json.dumps({"score": score, "rationale": "fine"})


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


class ByDimension:
    """Completion that answers according to the property named in the prompt."""

    def __init__(self, outcomes: dict):
        self.outcomes = outcomes
        self.dimensions = []

    def __call__(self, prompt, **kwargs):
        dimension = prompt.split("\n", 1)[0].split("j3/")[1]
        self.dimensions.append(dimension)
        outcome = self.outcomes[dimension]
        if isinstance(outcome, list):
            outcome = outcome.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return {"text": outcome, "latency_seconds": 0.5, "from_cache": False}


def prompts_for(reply="Reply text here", message="@AmericanAir my bag never arrived"):
    return {d: jd.build_dimension_prompt(d, message, evidence(), reply) for d in jd.DIMENSIONS}


class TestPrompts(unittest.TestCase):
    def test_version_and_five_templates(self):
        self.assertEqual(jd.JUDGE_PROMPT_VERSION, "j3")
        self.assertEqual(tuple(jd.JUDGE_TEMPLATES), jd.DIMENSIONS)
        self.assertEqual(len(set(t.template for t in jd.JUDGE_TEMPLATES.values())), 5)

    def test_hashes_are_per_dimension_and_deterministic(self):
        for dimension, template in jd.JUDGE_TEMPLATES.items():
            self.assertEqual(jd.JUDGE_PROMPT_SHA256[dimension],
                             hashlib.sha256(template.template.encode("utf-8")).hexdigest())
        self.assertEqual(len(set(jd.JUDGE_PROMPT_SHA256.values())), 5)

    def test_each_prompt_names_only_its_own_property(self):
        for dimension, prompt in prompts_for().items():
            self.assertTrue(prompt.startswith(f"Rating task: j3/{dimension}\n"))
            self.assertIn(f"PROPERTY: {dimension}\n", prompt)
            self.assertEqual(prompt.count("PROPERTY:"), 1)
            for other, words in OTHER_DIMENSION_WORDS.items():
                if other == dimension:
                    continue
                for word in words:
                    self.assertNotIn(word, prompt.lower(), f"{dimension} mentions {word}")

    def test_material_matches_the_declared_inputs(self):
        prompts = prompts_for()
        for dimension, prompt in prompts.items():
            has_evidence = "HISTORICAL SUPPORT EXAMPLES" in prompt
            self.assertEqual(has_evidence, "evidence" in jd.DIMENSION_MATERIAL[dimension], dimension)
            self.assertIn("CANDIDATE REPLY:\nReply text here", prompt)
            self.assertIn("my bag never arrived", prompt)
        for dimension in ("relevance", "helpfulness", "information_request_appropriateness"):
            self.assertNotIn("record locator, case", prompts[dimension])
        for dimension in ("groundedness", "claim_safety"):
            self.assertIn("Please DM your record locator, case 0.", prompts[dimension])
        self.assertIn("details that appear here are this customer's own", prompts["groundedness"])
        self.assertNotIn("for reference", prompts["claim_safety"])

    def test_shared_rules_and_output_schema(self):
        for prompt in prompts_for().values():
            for phrase in ("You are rating ONE property", "rated in separate tasks, so ignore them completely",
                           "Use only the material shown", "Do not assume any airline policy",
                           "do not invent facts", "exactly these two keys",
                           '{"score": <integer 1-5>, "rationale": "<at most 40 words, about this property only>"}'):
                self.assertIn(phrase, prompt)
            self.assertEqual(prompt.count("5 = "), 1)
            self.assertEqual(prompt.count("1 = "), 1)

    def test_no_overall_score_and_no_examples(self):
        for prompt in prompts_for().values():
            lowered = prompt.lower()
            for phrase in ("overall", "composite", "illustration", "for example", "e.g."):
                self.assertNotIn(phrase, lowered)

    def test_never_mentions_systems_labels_or_evaluation(self):
        forbidden = ["baseline", "llm_grounded", "template", "echo", "proposed", "gold",
                     "primary_intent", "predicted", "intent", "escalat", "confidence",
                     "golden", "g1_", "g7_", "grounding_flag", "system:", "accuracy"]
        for dimension, prompt in prompts_for().items():
            for word in forbidden:
                self.assertNotIn(word, prompt.lower(), f"{dimension}: {word}")

    def test_no_text_from_earlier_sanity_sets(self):
        earlier = ["Noted.", "upgrade", "$250", "Maria", "2381", "Got it, thanks",
                   "has been located", "$400 voucher", "passport number", "relaxing weekend"]
        for template in jd.JUDGE_TEMPLATES.values():
            for text in earlier:
                self.assertNotIn(text, template.template)

    def test_relevance_definition(self):
        prompt = prompts_for()["relevance"]
        self.assertIn("A reply about the customer's issue is relevant even if it is wrong or unhelpful.", prompt)
        self.assertIn("1 = completely unrelated to the customer's message", prompt)
        self.assertIn("5 = directly addresses the customer's specific issue", prompt)

    def test_claim_safety_rules(self):
        prompt = prompts_for()["claim_safety"]
        for phrase in ("- A reply that makes no factual assertion scores 5.",
                       "- Being irrelevant, short or unhelpful does not make a reply unsafe.",
                       "- A reply about the customer's issue can still score low if it contains "
                       "unsupported assertions.",
                       "actions taken, refunds, compensation, amounts, times, policies, promises, "
                       "URLs, names, flight numbers"):
            self.assertIn(phrase, prompt)

    def test_information_request_allows_asking_nothing(self):
        self.assertIn("5 = asks for what is needed, or asks for nothing when nothing is needed",
                      prompts_for()["information_request_appropriateness"])

    def test_sanitisation_and_literal_inputs(self):
        prompts = prompts_for(reply="cost $x {y}", message="@123456 @AmericanAir cost me $200 {evidence}")
        for prompt in prompts.values():
            self.assertNotRegex(prompt, r"@\d+")
            self.assertIn("cost me $200 {evidence}", prompt)
            self.assertIn("cost $x {y}", prompt)

    def test_identical_format_regardless_of_system(self):
        replies = {s: "same reply" for s in jd.SYSTEMS}
        items = jd.build_judge_items([row("e1", replies)])
        built = {jd.build_dimension_prompt("relevance", i["customer_message"], i["evidence"], i["reply"])
                 for i in items}
        self.assertEqual(len(built), 1)


class TestParser(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(jd.parse_dimension_judgement(json.dumps(VALID)), VALID)
        for score in jd.SCORES:
            self.assertEqual(jd.parse_dimension_judgement(ok(score))["score"], score)

    def test_key_order_does_not_matter(self):
        text = '{"rationale": "fine", "score": 2}'
        self.assertEqual(jd.parse_dimension_judgement(text), {"score": 2, "rationale": "fine"})

    def test_rejects_malformed_json(self):
        for text in ("not json", "", "{", None):
            with self.assertRaises(ValueError, msg=repr(text)):
                jd.parse_dimension_judgement(text)

    def test_rejects_non_objects(self):
        for text in ("[4]", "4", '"text"', "null"):
            with self.assertRaises(ValueError, msg=text):
                jd.parse_dimension_judgement(text)

    def test_rejects_missing_and_extra_keys(self):
        for payload in ({"score": 4}, {"rationale": "x"}, {},
                        {**VALID, "relevance": 4}, {**VALID, "overall": 4},
                        {"relevance": 4, "rationale": "x"}):
            with self.assertRaises(ValueError, msg=payload):
                jd.parse_dimension_judgement(json.dumps(payload))

    def test_rejects_bad_scores(self):
        for value in ("4", 4.0, 3.5, True, False, None, [4], {"v": 4}, 0, 6, -1):
            with self.assertRaises(ValueError, msg=repr(value)):
                jd.parse_dimension_judgement(json.dumps({"score": value, "rationale": "x"}))

    def test_rationale_rules(self):
        forty = " ".join(["w"] * 40)
        self.assertEqual(jd.parse_dimension_judgement(json.dumps({"score": 3, "rationale": forty}))["rationale"],
                         forty)
        self.assertEqual(jd.parse_dimension_judgement('{"score": 3, "rationale": "  ok  "}')["rationale"], "ok")
        for value in (" ".join(["w"] * 41), "", "   ", None, 5, ["x"], {"t": "x"}):
            with self.assertRaises(ValueError, msg=repr(value)):
                jd.parse_dimension_judgement(json.dumps({"score": 3, "rationale": value}))


class TestFiveCallsPerItem(unittest.TestCase):
    def test_one_call_per_dimension_in_order(self):
        fake = ByDimension({d: ok(3) for d in jd.DIMENSIONS})
        record = jd.judge_item(item(), complete_fn=fake, use_cache=False)
        self.assertEqual(fake.dimensions, list(jd.DIMENSIONS))
        self.assertEqual(record["item_status"], jd.ITEM_JUDGED)
        self.assertEqual(jd.dimension_scores(record), {d: 3 for d in jd.DIMENSIONS})

    def test_each_call_gets_its_own_prompt_and_config(self):
        fake = Fake(*[ok()] * 5)
        jd.judge_item(item(), complete_fn=fake, use_cache=False)
        self.assertEqual(len(fake.prompts), 5)
        self.assertEqual(len({p for p, _ in fake.prompts}), 5)
        for _, kwargs in fake.prompts:
            self.assertEqual(kwargs, {"model": "qwen2.5:7b", "use_cache": False})

    def test_cache_keys_differ_per_dimension(self):
        keys = {llm._cache_key(p, "qwen2.5:7b", 0.0, 42, True) for p in prompts_for().values()}
        self.assertEqual(len(keys), 5)

    def test_record_structure(self):
        record = jd.judge_item(item(), complete_fn=Fake(*[ok(5)] * 5))
        self.assertEqual(set(record), {"item_id", "evaluation_id", "system", "judge_model",
                                       "judge_prompt_version", "judge_prompt_sha256",
                                       "dimensions", "item_status"})
        self.assertEqual(record["judge_prompt_sha256"], jd.JUDGE_PROMPT_SHA256)
        self.assertEqual(set(record["dimensions"]), set(jd.DIMENSIONS))
        for result in record["dimensions"].values():
            self.assertEqual(set(result), {"status", "score", "rationale", "raw_output", "error",
                                           "latency_seconds", "from_cache", "attempt",
                                           "first_attempt_error"})
            self.assertEqual((result["status"], result["score"], result["attempt"]), (jd.JUDGED, 5, 1))
        for key in ("score", "overall", "composite", "scores"):
            self.assertNotIn(key, record)

    def test_parse_failure_affects_only_its_dimension(self):
        fake = ByDimension({**{d: ok(4) for d in jd.DIMENSIONS}, "helpfulness": '{"score": "high"}'})
        record = jd.judge_item(item(), complete_fn=fake)
        self.assertEqual(record["dimensions"]["helpfulness"]["status"], jd.JUDGE_PARSE_FAILURE)
        self.assertEqual(record["dimensions"]["helpfulness"]["raw_output"], '{"score": "high"}')
        self.assertIsNone(record["dimensions"]["helpfulness"]["score"])
        self.assertEqual(jd.dimension_scores(record),
                         {d: 4 for d in jd.DIMENSIONS if d != "helpfulness"})
        self.assertEqual(record["item_status"], jd.ITEM_PARTIALLY_JUDGED)

    def test_empty_replies_make_no_calls(self):
        for reply in ("", "   ", None):
            fake = Fake()
            record = jd.judge_item(item(reply=reply), complete_fn=fake)
            self.assertEqual(fake.prompts, [])
            self.assertEqual(record["item_status"], jd.ITEM_NOT_JUDGED_EMPTY_REPLY)
            self.assertTrue(all(r["status"] == jd.NOT_JUDGED_EMPTY_REPLY
                                for r in record["dimensions"].values()))

    def test_programming_errors_are_not_swallowed(self):
        with self.assertRaises(KeyError):
            jd.judge_item(item(), complete_fn=Fake(KeyError("bug")))


class TestRetryPolicy(unittest.TestCase):
    def test_malformed_output_is_not_retried(self):
        fake = ByDimension({**{d: ok() for d in jd.DIMENSIONS}, "claim_safety": "garbage"})
        records = jd.run_judge([item()], complete_fn=fake, use_cache=False)
        self.assertEqual(fake.dimensions.count("claim_safety"), 1)
        self.assertEqual(records[0]["dimensions"]["claim_safety"]["status"], jd.JUDGE_PARSE_FAILURE)
        self.assertEqual(records[0]["dimensions"]["claim_safety"]["attempt"], 1)

    def test_infrastructure_error_retried_once_per_call(self):
        fake = ByDimension({**{d: ok() for d in jd.DIMENSIONS},
                            "groundedness": [llm.LLMError("HTTP 500"), ok(2)]})
        records = jd.run_judge([item()], complete_fn=fake, use_cache=False)
        self.assertEqual(len(fake.dimensions), 6)
        self.assertEqual(fake.dimensions.count("groundedness"), 2)
        result = records[0]["dimensions"]["groundedness"]
        self.assertEqual((result["status"], result["score"], result["attempt"]), (jd.JUDGED, 2, 2))
        self.assertIn("HTTP 500", result["first_attempt_error"])
        self.assertEqual(records[0]["item_status"], jd.ITEM_JUDGED)
        self.assertEqual(records[0]["dimensions"]["relevance"]["attempt"], 1)

    def test_infrastructure_error_is_retried_only_once(self):
        fake = ByDimension({**{d: ok() for d in jd.DIMENSIONS},
                            "relevance": [MemoryError("x"), OSError("y")]})
        records = jd.run_judge([item()], complete_fn=fake, use_cache=False)
        self.assertEqual(fake.dimensions.count("relevance"), 2)
        result = records[0]["dimensions"]["relevance"]
        self.assertEqual(result["status"], jd.JUDGE_INFRASTRUCTURE_ERROR)
        self.assertIn("OSError", result["error"])
        self.assertIn("MemoryError", result["first_attempt_error"])
        self.assertEqual(records[0]["item_status"], jd.ITEM_PARTIALLY_JUDGED)

    def test_retries_happen_after_the_first_pass(self):
        order = []
        items = [item(item_id="j1", evaluation_id="e1"), item(item_id="j2", evaluation_id="e2")]
        outcomes = iter([llm.LLMError("down")] + [ok()] * 9 + [ok()])

        def complete(prompt, **kwargs):
            order.append(prompt.split("\n", 1)[0])
            outcome = next(outcomes)
            if isinstance(outcome, BaseException):
                raise outcome
            return {"text": outcome, "latency_seconds": 0.1, "from_cache": False}

        records = jd.run_judge(items, complete_fn=complete, use_cache=False)
        self.assertEqual(len(order), 11)
        self.assertEqual(order[-1], "Rating task: j3/relevance")
        self.assertEqual(records[0]["dimensions"]["relevance"]["attempt"], 2)

    def test_item_statuses_and_no_dropped_items(self):
        items = [item(item_id=f"j{n}", evaluation_id=f"e{n}") for n in range(4)]
        items[1]["reply"] = ""
        outcomes = ([ok()] * 5 + ["bad"] * 5
                    + [llm.LLMError("down")] * 5 + [llm.LLMError("down")] * 5)
        records = jd.run_judge(items, complete_fn=Fake(*outcomes), use_cache=False)
        self.assertEqual([r["item_id"] for r in records], [i["item_id"] for i in items])
        self.assertEqual([r["item_status"] for r in records],
                         [jd.ITEM_JUDGED, jd.ITEM_NOT_JUDGED_EMPTY_REPLY, jd.ITEM_FAILED, jd.ITEM_FAILED])


def record(evaluation_id, system, scores=None, status=None):
    """Synthetic j3 record. `status` overrides every dimension's status."""
    scores = scores or {}
    dims = {}
    for d in jd.DIMENSIONS:
        s = status or jd.JUDGED
        dims[d] = {**jd._empty_dimension(s), "score": scores.get(d, 3) if s == jd.JUDGED else None}
    rec = {"evaluation_id": evaluation_id, "system": system, "dimensions": dims}
    rec["item_status"] = jd.item_status(dims)
    return rec


class TestAggregation(unittest.TestCase):
    def test_dimension_summary(self):
        failed = record("e3", "llm_grounded", {"relevance": 1})
        failed["dimensions"]["relevance"] = jd._empty_dimension(jd.JUDGE_PARSE_FAILURE)
        failed["item_status"] = jd.item_status(failed["dimensions"])
        records = [record("e1", "llm_grounded", {"relevance": 5}),
                   record("e2", "llm_grounded", {"relevance": 2}),
                   failed,
                   record("e4", "llm_grounded", status=jd.NOT_JUDGED_EMPTY_REPLY),
                   record("e1", "baseline_b_template", {"relevance": 1})]
        s = jd.dimension_summary(records, "llm_grounded")
        rel = s["dimensions"]["relevance"]
        self.assertEqual(rel["n"], 2)
        self.assertEqual(rel["mean"], 3.5)
        self.assertEqual(rel["distribution"], {"1": 0, "2": 1, "3": 0, "4": 0, "5": 1})
        self.assertEqual(rel["status"], {"judged": 2, "not_judged_empty_reply": 1,
                                         "judge_parse_failure": 1, "judge_infrastructure_error": 0})
        self.assertEqual(s["dimensions"]["helpfulness"]["n"], 3)
        self.assertEqual(s["item_status"], {"judged": 2, "partially_judged": 1, "failed": 0,
                                            "not_judged_empty_reply": 1})
        self.assertEqual(s["items"], 4)
        for key in ("overall", "composite", "score"):
            self.assertNotIn(key, s)

    def test_paired_comparison_per_dimension(self):
        missing = record("e3", "llm_grounded", {"relevance": 4})
        missing["dimensions"]["claim_safety"] = jd._empty_dimension(jd.JUDGE_PARSE_FAILURE)
        records = [
            record("e1", "llm_grounded", {"relevance": 5}), record("e1", "baseline_b_template", {"relevance": 3}),
            record("e2", "llm_grounded", {"relevance": 2}), record("e2", "baseline_b_template", {"relevance": 4}),
            missing, record("e3", "baseline_b_template", {"relevance": 4}),
        ]
        c = jd.paired_comparison(records, "llm_grounded", "baseline_b_template", resamples=500)
        rel, claim = c["dimensions"]["relevance"], c["dimensions"]["claim_safety"]
        self.assertEqual((rel["pairs"], rel["win"], rel["tie"], rel["loss"]), (3, 1, 1, 1))
        self.assertEqual(rel["mean_difference"], 0.0)
        self.assertLessEqual(rel["interval_95"][0], 0.0)
        self.assertGreaterEqual(rel["interval_95"][1], 0.0)
        self.assertEqual(claim["pairs"], 2)
        self.assertEqual(claim["unpaired"], {"llm_grounded": 0, "baseline_b_template": 1})
        self.assertNotIn("overall", c)

    def test_paired_bootstrap_is_deterministic(self):
        records = [record(f"e{n}", s, {"relevance": (n % 5) + 1} if s == "llm_grounded" else {})
                   for n in range(20) for s in ("llm_grounded", "baseline_b_template")]
        a = jd.paired_comparison(records, "llm_grounded", "baseline_b_template")
        self.assertEqual(a, jd.paired_comparison(records, "llm_grounded", "baseline_b_template"))
        self.assertIn("2000 resamples, seed 42", a["dimensions"]["relevance"]["interval_method"])

    def test_descriptive_comparison_has_no_interval(self):
        records = [record("e1", "llm_grounded"), record("e1", "baseline_a_echo", {"relevance": 1})]
        c = jd.paired_comparison(records, "llm_grounded", "baseline_a_echo", with_interval=False)
        self.assertEqual(c["kind"], "descriptive")
        self.assertNotIn("interval_95", c["dimensions"]["relevance"])
        self.assertEqual(c["dimensions"]["relevance"]["win"], 1)

    def test_no_shared_pairs(self):
        c = jd.paired_comparison([record("e1", "llm_grounded")], "llm_grounded", "baseline_b_template")
        self.assertEqual(c["dimensions"]["relevance"]["pairs"], 0)
        self.assertIsNone(c["dimensions"]["relevance"]["mean_difference"])


class TestPairing(unittest.TestCase):
    def test_one_item_per_row_and_system(self):
        items = jd.build_judge_items([row("e1"), row("e2")])
        self.assertEqual(len(items), 6)
        self.assertEqual({(i["evaluation_id"], i["system"]) for i in items},
                         {(e, s) for e in ("e1", "e2") for s in jd.SYSTEMS})
        self.assertEqual([i["item_id"] for i in items], [f"j{n:04d}" for n in range(1, 7)])

    def test_shuffle_is_deterministic_and_seeded(self):
        rows = [row(f"e{n}") for n in range(10)]

        def order(seed):
            return [(i["evaluation_id"], i["system"]) for i in jd.build_judge_items(rows, seed=seed)]

        unshuffled = [(f"e{n}", s) for n in range(10) for s in jd.SYSTEMS]
        self.assertEqual(order(42), order(42))
        self.assertNotEqual(order(42), order(7))
        self.assertNotEqual(order(42), unshuffled)
        self.assertEqual(sorted(order(42)), sorted(unshuffled))

    def test_system_identity_never_reaches_a_prompt(self):
        items = jd.build_judge_items([row("e1", {s: "Sorry about the bag." for s in jd.SYSTEMS})])
        fake = Fake(*[ok()] * 15)
        records = jd.run_judge(items, complete_fn=fake, use_cache=False)
        self.assertEqual(len(fake.prompts), 15)
        for prompt, _ in fake.prompts:
            for system in jd.SYSTEMS:
                self.assertNotIn(system, prompt)
        self.assertEqual({r["system"] for r in records}, set(jd.SYSTEMS))

    def test_invalid_rows_raise(self):
        with self.assertRaises(KeyError):
            jd.build_judge_items([row("e1", {"llm_grounded": "x"})])
        with self.assertRaises(ValueError):
            jd.build_judge_items([row("e1"), row("e1")])


def judged_cases(cases, scores_by_case):
    return {c["case"]: record(c["case"], "sanity", scores_by_case.get(c["case"], {})) for c in cases}


class TestJ3Criteria(unittest.TestCase):
    def passing(self):
        scores = {c["case"]: {"relevance": 4, "claim_safety": 5, "helpfulness": 2,
                              "information_request_appropriateness": 4, "groundedness": 3}
                  for c in jd.J3_SANITY_CASES}
        for case in jd.J3_ON_TOPIC_UNSAFE_CASES:
            scores[case] = {**scores[case], "claim_safety": 1}
        for case in jd.J3_BAD_REQUEST_CASES:
            scores[case] = {**scores[case], "information_request_appropriateness": 1}
        return judged_cases(jd.J3_SANITY_CASES, scores)

    def test_design(self):
        self.assertEqual(len(jd.J3_SANITY_CASES), 10)
        names = {c["case"] for c in jd.J3_SANITY_CASES}
        for group in (jd.J3_CLAIM_FREE_CASES, jd.J3_ON_TOPIC_UNSAFE_CASES,
                      jd.J3_BAD_REQUEST_CASES, jd.J3_MIXED_CASES):
            self.assertTrue(set(group) <= names)
        self.assertEqual(len(jd.J3_MIXED_CASES), 8)
        self.assertEqual(jd.J3_MIXED_CASES_WITH_SPREAD_REQUIRED, 5)
        self.assertEqual({d for c in jd.J3_SANITY_CASES for d in c["expect"]}, set(jd.DIMENSIONS))
        for case in jd.J3_SANITY_CASES:
            for rule in case["expect"].values():
                self.assertRegex(rule, r"^(>=|<=)[1-5]$")

    def test_j3_cases_do_not_reuse_earlier_replies(self):
        earlier = {c["reply"] for c in jd.J2_SANITY_CASES} | {"Noted."}
        for case in jd.J3_SANITY_CASES:
            self.assertNotIn(case["reply"], earlier)

    def test_passes(self):
        result = jd.evaluate_j3_criteria(self.passing())
        self.assertTrue(result["all_passed"], result)

    def test_collapse_fails_every_criterion(self):
        records = judged_cases(jd.J3_SANITY_CASES,
                               {c["case"]: {d: 1 for d in jd.DIMENSIONS} for c in jd.J3_SANITY_CASES})
        result = jd.evaluate_j3_criteria(records)
        self.assertFalse(result["all_passed"])
        self.assertTrue(all(not c["passed"] for c in result["criteria"].values()))

    def test_c2_needs_both_conditions(self):
        records = self.passing()
        records["c07_on_topic_invented_url_policy"]["dimensions"]["relevance"]["score"] = 2
        self.assertFalse(jd.evaluate_j3_criteria(records)["criteria"][
            "C2_on_topic_unsafe_relevance_at_least_3_and_claim_safety_at_most_2"]["passed"])

    def test_c3_needs_two_other_dimensions_at_least_3(self):
        records = self.passing()
        dims = records["c08_irrelevant_request"]["dimensions"]
        dims["relevance"]["score"], dims["groundedness"]["score"], dims["helpfulness"]["score"] = 1, 1, 4
        result = jd.evaluate_j3_criteria(records)
        self.assertTrue(result["criteria"]["C3_bad_request_information_at_most_2_and_two_others_at_least_3"]["passed"])
        records["c08_irrelevant_request"]["dimensions"]["claim_safety"]["score"] = 2
        result = jd.evaluate_j3_criteria(records)
        self.assertFalse(result["criteria"]["C3_bad_request_information_at_most_2_and_two_others_at_least_3"]["passed"])

    def test_unjudged_dimension_fails(self):
        records = self.passing()
        records["c02_off_topic_claim_free"]["dimensions"]["claim_safety"] = \
            jd._empty_dimension(jd.JUDGE_PARSE_FAILURE)
        result = jd.evaluate_j3_criteria(records)
        self.assertFalse(result["criteria"]["C1_claim_free_cases_claim_safety_at_least_4"]["passed"])
        self.assertIsNone(result["criteria"]["C4_mixed_cases_with_spread_at_least_2"]["observed"]
                          ["c02_off_topic_claim_free"])

    def test_spread_threshold(self):
        records = self.passing()
        for case in jd.J3_MIXED_CASES[:4]:
            for d in jd.DIMENSIONS:
                records[case]["dimensions"][d]["score"] = 3
        self.assertFalse(jd.evaluate_j3_criteria(records)["criteria"]
                         ["C4_mixed_cases_with_spread_at_least_2"]["passed"])

    def test_j2_rule_on_j3_records(self):
        scores = {c["case"]: {"relevance": 4, "claim_safety": 5, "helpfulness": 2}
                  for c in jd.J2_SANITY_CASES}
        for case in jd.J2_ON_TOPIC_UNSAFE_CASES:
            scores[case] = {"relevance": 4, "claim_safety": 1}
        self.assertTrue(jd.evaluate_j2_rule(judged_cases(jd.J2_SANITY_CASES, scores))["all_passed"])
        ones = {c["case"]: {d: 1 for d in jd.DIMENSIONS} for c in jd.J2_SANITY_CASES}
        self.assertFalse(jd.evaluate_j2_rule(judged_cases(jd.J2_SANITY_CASES, ones))["all_passed"])

    def test_j2_cases_are_unchanged(self):
        self.assertEqual(len(jd.J2_SANITY_CASES), 9)
        self.assertEqual(jd.J2_SANITY_CASES[1]["reply"],
                         "Good news: your bag has been located and will be delivered to your address tonight.")
        self.assertEqual(jd.J2_MIXED_CASES_WITH_SPREAD_REQUIRED, 5)

    def test_expectation_rules(self):
        self.assertTrue(jd._meets(4, ">=4"))
        self.assertFalse(jd._meets(3, ">=4"))
        self.assertTrue(jd._meets(2, "<=2"))
        self.assertFalse(jd._meets(3, "<=2"))


class TestMessageSelection(unittest.TestCase):
    def corpus(self, first_id="conv_1208854"):
        long = "{} please help me with this today at the airport"
        return pd.DataFrame({
            "conversation_id": [first_id, "c1", "c2", "c3", "c4", "c5", "c6"],
            "customer_author_id": [str(n) for n in range(7)],
            "weak_label": ["baggage", "baggage", "baggage", "flight_delay",
                           "flight_cancellation_rebooking", "praise_and_compliment", "baggage"],
            "customer_message": [long.format("my bag is lost"), long.format("my bag is lost"),
                                 long.format("bag fees are high"), long.format("big delay"),
                                 long.format("flight was cancelled"), long.format("great crew"),
                                 "bag lost"],
        })

    def test_deterministic_filtered_and_excluding(self):
        corpus = self.corpus()
        a = jd.select_sanity_messages(corpus)
        self.assertEqual(a, jd.select_sanity_messages(corpus))
        self.assertEqual(set(a), set(jd.SANITY_SOURCES))
        self.assertEqual(a["lost_bag"]["conversation_id"], "c1")

    def test_j3_excludes_j2_messages(self):
        corpus = self.corpus(first_id="conv_1208824")
        both = jd.J1_SANITY_CONVERSATIONS | jd.J2_SANITY_CONVERSATIONS
        self.assertEqual(jd.select_sanity_messages(corpus, exclude=both)["lost_bag"]["conversation_id"], "c1")
        self.assertFalse(jd.J1_SANITY_CONVERSATIONS & jd.J2_SANITY_CONVERSATIONS)

    def test_fails_loudly_without_candidates(self):
        corpus = pd.DataFrame({"conversation_id": ["c1"], "customer_author_id": ["1"],
                               "weak_label": ["baggage"], "customer_message": ["short"]})
        with self.assertRaises(ValueError):
            jd.select_sanity_messages(corpus)

    def test_cli_requires_explicit_flag(self):
        self.assertEqual(jd.main([]), 2)
        self.assertEqual(jd.main(["--golden"]), 2)


class TestCacheIsolation(unittest.TestCase):
    def test_redirects_and_restores(self):
        original = config.CACHE_DIR
        with tempfile.TemporaryDirectory() as tmp:
            with jd.isolated_cache(Path(tmp)) as directory:
                self.assertEqual(config.CACHE_DIR, Path(tmp))
                self.assertEqual(llm._cache_path("k"), Path(tmp) / "k.json")
                self.assertEqual(directory, Path(tmp))
        self.assertEqual(config.CACHE_DIR, original)

    def test_restores_after_an_error(self):
        original = config.CACHE_DIR
        with self.assertRaises(RuntimeError):
            with jd.isolated_cache(Path("elsewhere")):
                raise RuntimeError("boom")
        self.assertEqual(config.CACHE_DIR, original)

    def test_isolated_directories_are_distinct(self):
        dirs = {config.CACHE_DIR, jd.SANITY_CACHE_DIR, jd.PROBE_CACHE_DIR}
        self.assertEqual(len(dirs), 3)

    def test_isolated_writes_do_not_touch_production(self):
        production = jd.cache_snapshot(config.CACHE_DIR)
        with tempfile.TemporaryDirectory() as tmp:
            with jd.isolated_cache(Path(tmp)):
                llm._cache_path("probe").parent.mkdir(parents=True, exist_ok=True)
                llm._cache_path("probe").write_text("{}", encoding="utf-8")
            self.assertTrue((Path(tmp) / "probe.json").exists())
        self.assertEqual(jd.cache_snapshot(config.CACHE_DIR), production)

    def test_snapshot_detects_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            empty = jd.cache_snapshot(directory)
            self.assertEqual(empty["files"], 0)
            (directory / "a.json").write_text("{}", encoding="utf-8")
            changed = jd.cache_snapshot(directory)
            self.assertEqual(changed["files"], 1)
            self.assertNotEqual(empty["listing_sha256"], changed["listing_sha256"])
            self.assertEqual(changed, jd.cache_snapshot(directory))
        self.assertEqual(jd.cache_snapshot(Path("does/not/exist"))["files"], 0)


class TestRetiredPrompts(unittest.TestCase):
    def test_hashes_preserved(self):
        self.assertEqual(jd.JUDGE_PROMPT_VERSION_J1, "j1")
        self.assertEqual(jd.JUDGE_PROMPT_VERSION_J2, "j2")
        self.assertEqual(jd.JUDGE_PROMPT_SHA256_J1,
                         "13127294428c6121d00e139582e8e7cc3026c7eab12282ea89a5c287c8e0c48d")
        self.assertEqual(jd.JUDGE_PROMPT_SHA256_J2,
                         "35bfc2f57e4b93680d569ac19242b119182589a1ddba3a94497c20032cfe8e6c")
        self.assertFalse({jd.JUDGE_PROMPT_SHA256_J1, jd.JUDGE_PROMPT_SHA256_J2}
                         & set(jd.JUDGE_PROMPT_SHA256.values()))

    def test_retired_builder_and_parser_still_work(self):
        prompt = jd.build_all_dimensions_prompt(jd.JUDGE_TEMPLATE_J2, "@123 my bag", evidence(), "reply")
        self.assertIn("independent measurements", prompt)
        self.assertNotRegex(prompt, r"@\d+")
        payload = {d: 3 for d in jd.DIMENSIONS} | {"rationale": "fine"}
        self.assertEqual(jd.parse_all_dimensions_judgement(json.dumps(payload))["relevance"], 3)
        with self.assertRaises(ValueError):
            jd.parse_all_dimensions_judgement(json.dumps({"score": 3, "rationale": "x"}))

    def test_live_path_does_not_use_retired_prompts(self):
        live = prompts_for()["relevance"]
        self.assertNotIn("Score each dimension independently of the others", live)
        self.assertNotIn("independent measurements, not parts of one overall grade", live)


class TestModelSubstitution(unittest.TestCase):
    def test_default_model_is_config(self):
        fake = Fake(*[ok()] * 5)
        record = jd.judge_item(item(), complete_fn=fake)
        self.assertEqual(record["judge_model"], config.JUDGE_MODEL)
        self.assertTrue(all(kwargs["model"] == config.JUDGE_MODEL for _, kwargs in fake.prompts))

    def test_explicit_model_reaches_every_call_and_the_record(self):
        fake = ByDimension({**{d: ok() for d in jd.DIMENSIONS},
                            "helpfulness": [llm.LLMError("down"), ok()]})
        calls = []

        def complete(prompt, **kwargs):
            calls.append(kwargs["model"])
            return fake(prompt, **kwargs)

        records = jd.run_judge([item()], model="qwen3:8b", complete_fn=complete, use_cache=False)
        self.assertEqual(calls, ["qwen3:8b"] * 6)
        self.assertEqual(records[0]["judge_model"], "qwen3:8b")

    def test_model_changes_the_cache_key(self):
        prompt = prompts_for()["claim_safety"]
        self.assertNotEqual(llm._cache_key(prompt, "qwen3:8b", 0.0, 42, True),
                            llm._cache_key(prompt, "qwen2.5:7b", 0.0, 42, True))

    def test_config_default_unchanged(self):
        self.assertEqual(config.JUDGE_MODEL, "qwen2.5:7b")

    def test_per_model_cache_and_report_locations(self):
        self.assertEqual(jd.sanity_cache_dir(None), jd.SANITY_CACHE_DIR)
        self.assertEqual(jd.sanity_cache_dir("qwen2.5:7b"), jd.SANITY_CACHE_DIR)
        self.assertEqual(jd.sanity_cache_dir("qwen3:8b"), jd.SANITY_CACHE_DIR / "qwen3_8b")
        self.assertEqual(jd.probe_cache_dir("qwen3:8b"), jd.PROBE_CACHE_DIR / "qwen3_8b")
        self.assertNotEqual(jd.sanity_cache_dir("qwen3:8b"), config.CACHE_DIR)
        self.assertEqual(jd.report_path("sanity", None, "set1"), jd.SANITY_REPORT)
        self.assertEqual(jd.report_path("sanity", "qwen3:8b", "set2").name,
                         "phase6_judge_sanity_j3_set2_qwen3_8b.json")
        self.assertEqual(jd.report_path("probe", "qwen3:8b", "set2").name,
                         "phase6_judge_probe_j3_set2_qwen3_8b.json")

    def test_cli_rejects_unknown_set_and_missing_value(self):
        self.assertEqual(jd.main(["--sanity-check", "--set", "set9"]), 2)
        with self.assertRaises(SystemExit):
            jd.main(["--sanity-check", "--model"])


class TestSet2Design(unittest.TestCase):
    spec = jd.J3_SET2_SPEC

    def test_ten_cases_covering_every_dimension(self):
        self.assertEqual(len(self.spec["cases"]), 10)
        self.assertEqual({d for c in self.spec["cases"] for d in c["expect"]}, set(jd.DIMENSIONS))
        names = {c["case"] for c in self.spec["cases"]}
        for key in ("claim_free", "on_topic_unsafe", "bad_request", "mixed"):
            self.assertTrue(set(self.spec[key]) <= names, key)
        self.assertEqual(len(self.spec["mixed"]), 8)
        self.assertEqual(self.spec["mixed_required"], 5)

    def test_roles_agree_with_expectations(self):
        by_name = {c["case"]: c["expect"] for c in self.spec["cases"]}
        for case in self.spec["claim_free"]:
            self.assertEqual(by_name[case]["claim_safety"], ">=4", case)
        for case in self.spec["on_topic_unsafe"]:
            self.assertEqual((by_name[case]["relevance"], by_name[case]["claim_safety"]), (">=3", "<=2"))
        for case in self.spec["bad_request"]:
            self.assertEqual(by_name[case]["information_request_appropriateness"], "<=2")

    def test_no_reuse_of_earlier_sets(self):
        earlier = ({c["reply"] for c in jd.J2_SANITY_CASES} | {c["reply"] for c in jd.J3_SANITY_CASES}
                   | {"Noted."})
        for case in self.spec["cases"]:
            self.assertNotIn(case["reply"], earlier)
        self.assertTrue(jd.J3_SET1_CONVERSATIONS <= self.spec["exclude"])
        self.assertTrue(jd.J2_SANITY_CONVERSATIONS <= self.spec["exclude"])
        self.assertTrue(jd.J1_SANITY_CONVERSATIONS <= self.spec["exclude"])

    def test_sources_used_by_cases_exist(self):
        self.assertTrue({c["source"] for c in self.spec["cases"]} <= set(self.spec["sources"]))

    def test_set1_spec_is_the_original_run(self):
        spec = jd.J3_SET1_SPEC
        self.assertIs(spec["cases"], jd.J3_SANITY_CASES)
        self.assertEqual(spec["exclude"], jd.J1_SANITY_CONVERSATIONS | jd.J2_SANITY_CONVERSATIONS)
        self.assertIs(spec["sources"], jd.SANITY_SOURCES)
        self.assertEqual(set(jd.SANITY_SETS), {"set1", "set2"})

    def test_same_rule_for_both_sets(self):
        for spec in (jd.J3_SET1_SPEC, jd.J3_SET2_SPEC):
            scores = {c["case"]: {"relevance": 4, "claim_safety": 5, "helpfulness": 2,
                                  "information_request_appropriateness": 4, "groundedness": 3}
                      for c in spec["cases"]}
            for case in spec["on_topic_unsafe"]:
                scores[case] = {**scores[case], "claim_safety": 1}
            for case in spec["bad_request"]:
                scores[case] = {**scores[case], "information_request_appropriateness": 1}
            passing = judged_cases(spec["cases"], scores)
            self.assertTrue(jd.evaluate_j3_criteria(passing, spec)["all_passed"], spec["name"])
            ones = judged_cases(spec["cases"], {c["case"]: {d: 1 for d in jd.DIMENSIONS}
                                                for c in spec["cases"]})
            result = jd.evaluate_j3_criteria(ones, spec)
            self.assertEqual(set(result["criteria"]),
                             set(jd.evaluate_j3_criteria(ones, jd.J3_SET1_SPEC)["criteria"]))
            self.assertFalse(result["all_passed"])

    def test_default_spec_is_set1(self):
        records = judged_cases(jd.J3_SANITY_CASES, {})
        self.assertEqual(jd.evaluate_j3_criteria(records), jd.evaluate_j3_criteria(records, jd.J3_SET1_SPEC))


class TestStrictCancellationFilter(unittest.TestCase):
    def corpus(self, messages):
        return pd.DataFrame({
            "conversation_id": [f"c{n}" for n in range(len(messages))],
            "customer_author_id": [str(n) for n in range(len(messages))],
            "weak_label": ["flight_cancellation_rebooking"] * len(messages),
            "customer_message": messages,
        })

    def select(self, messages):
        sources = {"cancelled_flight": jd.SET2_SOURCES["cancelled_flight"]}
        return jd.select_sanity_messages(self.corpus(messages), sources=sources, exclude=frozenset())

    def test_accepts_a_cancelled_flight(self):
        chosen = self.select(["my flight to Boston was cancelled this morning and nobody will help me"])
        self.assertEqual(chosen["cancelled_flight"]["conversation_id"], "c0")
        chosen = self.select(["our connecting flight got canceled tonight and we are stuck at the gate"])
        self.assertEqual(chosen["cancelled_flight"]["conversation_id"], "c0")

    def test_rejects_fees_and_non_flight_cancellations(self):
        for message in ("my flight was cancelled and now they want a change fee from me today",
                        "I need to cancel my hotel booking made through your website today please",
                        "there is a cancellation charge on my card and I do not understand why",
                        "there will be trouble if our flights are cancelled over the holidays this year",
                        "they canceled our connecting flight tonight and we are stuck at the gate"):
            with self.assertRaises(ValueError, msg=message):
                self.select([message])

    def test_two_element_sources_still_work(self):
        corpus = self.corpus(["my flight was cancelled this morning please help me get home today"])
        corpus["weak_label"] = "flight_cancellation_rebooking"
        chosen = jd.select_sanity_messages(
            corpus, sources={"x": ("flight_cancellation_rebooking", r"\bcancel")}, exclude=frozenset())
        self.assertEqual(chosen["x"]["conversation_id"], "c0")


class TestStrictLostBagFilter(unittest.TestCase):
    def select(self, messages):
        corpus = pd.DataFrame({
            "conversation_id": [f"c{n}" for n in range(len(messages))],
            "customer_author_id": [str(n) for n in range(len(messages))],
            "weak_label": ["baggage"] * len(messages),
            "customer_message": messages,
        })
        return jd.select_sanity_messages(corpus, sources={"lost_bag": jd.SET2_SOURCES["lost_bag"]},
                                         exclude=frozenset())

    def test_accepts_lost_or_missing_bags(self):
        for message in ("you lost my bag on the way to Dallas and nobody can tell me where it is",
                        "my luggage has been missing for three days now and I need my medication"):
            self.assertEqual(self.select([message])["lost_bag"]["conversation_id"], "c0", message)

    def test_rejects_loose_matches(self):
        for message in ("so many bags came out first. I lost count at 100 and still wait here",
                        "why do checked bags cost so much on this route compared with last year",
                        "my luggage tag got lost somewhere and the new one broke on the way back"):
            with self.assertRaises(ValueError, msg=message):
                self.select([message])


class TestLocalDigest(unittest.TestCase):
    def test_reads_digest_from_local_listing(self):
        from unittest import mock

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"models": [{"name": "qwen3:8b", "digest": "abc"}]}).encode()

        with mock.patch.object(jd.urllib.request, "urlopen", return_value=Response()) as opened:
            self.assertEqual(jd.local_model_digest("qwen3:8b"), "abc")
            self.assertIsNone(jd.local_model_digest("missing:1b"))
        self.assertTrue(opened.call_args[0][0].startswith("http://localhost:11434/api/tags"))

    def test_unreachable_daemon_gives_none(self):
        from unittest import mock
        with mock.patch.object(jd.urllib.request, "urlopen", side_effect=OSError("down")):
            self.assertIsNone(jd.local_model_digest("qwen3:8b"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
