from __future__ import annotations

import unittest

import pandas as pd

from src import config
from src.leakage import (
    assert_isolated,
    exclude_golden,
    golden_conversation_ids,
    golden_customer_ids,
)
from src.taxonomy import ALL_LABELS, normalise, weak_label


def read_golden() -> pd.DataFrame:
    return pd.read_csv(config.GOLDEN_SET, dtype=str, keep_default_na=False, encoding="utf-8-sig")


class TestTaxonomy(unittest.TestCase):
    def test_label_set_matches_frozen_taxonomy(self):
        self.assertEqual(len(ALL_LABELS), 13)
        taxonomy = (config.GOLDEN_DIR / "taxonomy_v2.md").read_text(encoding="utf-8")
        for label in ALL_LABELS:
            self.assertIn(label, taxonomy, f"{label} missing from the frozen taxonomy")

    def test_normalise_strips_urls_mentions_and_case(self):
        self.assertEqual(normalise("@AmericanAir My BAG is lost https://t.co/abc"), "my bag is lost")

    def test_weak_label_assigns_unambiguous_message(self):
        label, scores = weak_label("My bag never arrived and my luggage is missing")
        self.assertEqual(label, "baggage")
        self.assertGreaterEqual(scores["baggage"], 2)

    def test_weak_label_abstains_on_tie(self):
        label, scores = weak_label("flight delayed and my bag is missing")
        self.assertIsNone(label)
        self.assertEqual(set(scores), {"flight_delay", "baggage"})

    def test_weak_label_abstains_when_nothing_matches(self):
        label, scores = weak_label("hello there")
        self.assertIsNone(label)
        self.assertEqual(scores, {})


class TestLeakageControl(unittest.TestCase):
    def test_golden_id_files_load_expected_counts(self):
        self.assertEqual(len(golden_conversation_ids()), 248)
        self.assertEqual(len(golden_customer_ids()), 248)

    def test_removes_golden_conversation(self):
        golden_id = sorted(golden_conversation_ids())[0]
        frame = pd.DataFrame({
            "conversation_id": [golden_id, "conv_not_golden_xyz"],
            "customer_author_id": ["999999991", "999999992"],
            "text": ["something", "something else"],
        })
        filtered, report = exclude_golden(frame)
        self.assertNotIn(golden_id, set(filtered["conversation_id"]))
        self.assertEqual(report["removed"]["conversation_id"], 1)
        self.assertEqual(report["rows_out"], 1)

    def test_removes_golden_customer(self):
        golden_customer = sorted(golden_customer_ids())[0]
        frame = pd.DataFrame({
            "conversation_id": ["conv_not_golden_a", "conv_not_golden_b"],
            "customer_author_id": [golden_customer, "999999992"],
            "text": ["alpha text here", "beta text here"],
        })
        filtered, report = exclude_golden(frame)
        self.assertEqual(report["removed"]["customer_author_id"], 1)
        self.assertEqual(filtered["customer_author_id"].tolist(), ["999999992"])

    def test_removes_exact_text_duplicate(self):
        copied = read_golden()["text"].iat[0]
        frame = pd.DataFrame({
            "conversation_id": ["conv_not_golden_c"],
            "customer_author_id": ["999999993"],
            "text": [copied],
        })
        filtered, report = exclude_golden(frame)
        self.assertEqual(report["removed"]["exact_text"], 1)
        self.assertTrue(filtered.empty)

    def test_catches_near_duplicate_with_different_url(self):
        base = read_golden()["text"].iat[0]
        frame = pd.DataFrame({
            "conversation_id": ["conv_not_golden_d"],
            "customer_author_id": ["999999994"],
            "text": [base + " https://t.co/DIFFERENTURL"],
        })
        filtered, report = exclude_golden(frame)
        removed = (report["removed"]["exact_text"] or 0) + (report["removed"]["near_duplicate"] or 0)
        self.assertGreaterEqual(removed, 1)
        self.assertTrue(filtered.empty)

    def test_keeps_unrelated_rows(self):
        frame = pd.DataFrame({
            "conversation_id": ["conv_zzz_1", "conv_zzz_2"],
            "customer_author_id": ["888888881", "888888882"],
            "text": ["completely unrelated message about zebras",
                     "another unrelated message about telescopes"],
        })
        filtered, report = exclude_golden(frame)
        self.assertEqual(report["rows_removed"], 0)
        self.assertEqual(len(filtered), 2)

    def test_reports_unenforceable_risks(self):
        frame = pd.DataFrame({"conversation_id": ["conv_zzz_3"], "text": ["hello"]})
        _, report = exclude_golden(frame)
        self.assertEqual(len(report["unenforceable_risks"]), 3)

    def test_requires_conversation_column(self):
        with self.assertRaises(KeyError):
            exclude_golden(pd.DataFrame({"text": ["x"]}))

    def test_assert_isolated_raises_on_leak(self):
        golden_id = sorted(golden_conversation_ids())[0]
        with self.assertRaises(AssertionError):
            assert_isolated(pd.DataFrame({"conversation_id": [golden_id]}))

    def test_assert_isolated_passes_on_clean_frame(self):
        assert_isolated(pd.DataFrame({"conversation_id": ["conv_zzz_4"]}))


class TestFrozenArtifacts(unittest.TestCase):
    def test_golden_set_is_still_248_rows(self):
        golden = read_golden()
        self.assertEqual(len(golden), 248)
        self.assertEqual(set(golden["batch"]), {"b01", "b02"})
        self.assertEqual(int((golden["batch"] == "b02").sum()), 100)

    def test_every_golden_label_is_in_taxonomy(self):
        self.assertTrue(set(read_golden()["primary_intent"]) <= set(ALL_LABELS))



class TestRetrieval(unittest.TestCase):
    """Small deterministic harness: a hand-built corpus, no dependence on the full build."""

    @classmethod
    def setUpClass(cls):
        from src.retrieve import Retriever
        golden = read_golden()
        cls.golden_conv = golden["conversation_id"].iat[0]
        cls.golden_customer = golden["customer_author_id"].iat[0]
        cls.golden_text = golden["text"].iat[0]

        cls.corpus = pd.DataFrame({
            "conversation_id": ["conv_t1", "conv_t2", "conv_t3", "conv_t4"],
            "tweet_id": [1, 2, 3, 4],
            "customer_author_id": ["700001", "700002", "700003", "700004"],
            "customer_message": [
                "my bag is lost and my luggage never arrived at the airport",
                "my flight was delayed three hours on the tarmac",
                "my bag is lost and my luggage never arrived at the airport",
                "thanks for the wonderful crew on my flight today",
            ],
            "brand_reply": ["we can help with the bag", "sorry about the delay",
                            "please send bag details", "glad you enjoyed it"],
            "weak_label": ["baggage", "flight_delay", "baggage", "praise_and_compliment"],
            "n_turns": [2, 4, 2, 2],
            "n_brand_turns": [1, 2, 1, 1],
            "customer_followups": [0, 1, 0, 0],
        })
        cls.Retriever = Retriever
        cls.retriever = Retriever(cls.corpus)

    def test_retrieval_is_deterministic(self):
        query = "lost luggage at the airport"
        first = self.retriever.retrieve(query, k=3)
        second = self.retriever.retrieve(query, k=3)
        self.assertEqual([e["conversation_id"] for e in first],
                         [e["conversation_id"] for e in second])
        self.assertEqual([e["similarity"] for e in first], [e["similarity"] for e in second])

    def test_ranks_relevant_evidence_first(self):
        top = self.retriever.retrieve("my luggage is lost", k=1)
        self.assertEqual(len(top), 1)
        self.assertEqual(top[0]["historical_weak_intent"], "baggage")

    def test_tie_ordering_is_stable_by_conversation_id(self):
        # conv_t1 and conv_t3 have identical text, so their scores tie exactly.
        results = self.retriever.retrieve("my bag is lost and my luggage never arrived", k=2)
        ids = [e["conversation_id"] for e in results]
        self.assertEqual(ids, sorted(ids))

    def test_self_exclusion_by_conversation_id(self):
        results = self.retriever.retrieve(
            "my bag is lost and my luggage never arrived at the airport",
            k=4, exclude_conversation_id="conv_t1",
        )
        self.assertNotIn("conv_t1", [e["conversation_id"] for e in results])

    def test_same_customer_exclusion(self):
        results = self.retriever.retrieve(
            "my bag is lost and my luggage never arrived at the airport",
            k=4, exclude_customer_id="700001",
        )
        self.assertNotIn("conv_t1", [e["conversation_id"] for e in results])

    def test_empty_query_returns_nothing(self):
        self.assertEqual(self.retriever.retrieve("", k=3), [])
        self.assertEqual(self.retriever.retrieve("@AmericanAir https://t.co/x", k=3), [])

    def test_query_with_no_lexical_overlap_returns_nothing(self):
        self.assertEqual(self.retriever.retrieve("zebra telescope quantum", k=3), [])

    def test_output_schema(self):
        from src.retrieve import EVIDENCE_FIELDS
        result = self.retriever.retrieve("lost luggage", k=1)[0]
        self.assertEqual(sorted(result.keys()), sorted(EVIDENCE_FIELDS))
        self.assertEqual(result["retrieved_by"], "tfidf")
        self.assertIn(result["evidence_shape"], {"two_turn", "multi_turn"})

    def test_random_baseline_is_reproducible(self):
        first = self.retriever.retrieve_random(k=3, seed=7)
        second = self.retriever.retrieve_random(k=3, seed=7)
        self.assertEqual([e["conversation_id"] for e in first],
                         [e["conversation_id"] for e in second])

    def test_random_baseline_differs_by_seed(self):
        seeds = {tuple(e["conversation_id"] for e in self.retriever.retrieve_random(k=2, seed=s))
                 for s in range(12)}
        self.assertGreater(len(seeds), 1)

    def test_random_baseline_respects_exclusions(self):
        for seed in range(15):
            results = self.retriever.retrieve_random(
                k=4, seed=seed, exclude_conversation_id="conv_t2", exclude_customer_id="700003")
            ids = [e["conversation_id"] for e in results]
            self.assertNotIn("conv_t2", ids)
            self.assertNotIn("conv_t3", ids)

    def test_random_baseline_uses_same_pool_and_schema(self):
        from src.retrieve import EVIDENCE_FIELDS
        result = self.retriever.retrieve_random(k=1, seed=3)[0]
        self.assertEqual(sorted(result.keys()), sorted(EVIDENCE_FIELDS))
        self.assertEqual(result["retrieved_by"], "random")
        self.assertIn(result["conversation_id"], set(self.corpus["conversation_id"]))

    def test_retriever_rejects_empty_corpus(self):
        with self.assertRaises(ValueError):
            self.Retriever(self.corpus.iloc[0:0])

    def test_golden_text_is_excluded_by_shared_leakage_function(self):
        # A corpus row copying a golden message must be removed before indexing.
        contaminated = pd.concat([
            self.corpus,
            pd.DataFrame({
                "conversation_id": ["conv_leak"], "tweet_id": [99],
                "customer_author_id": ["700099"], "customer_message": [self.golden_text],
                "brand_reply": ["x"], "weak_label": ["baggage"], "n_turns": [2],
                "n_brand_turns": [1], "customer_followups": [0],
            }),
        ], ignore_index=True)
        contaminated["text"] = contaminated["customer_message"]
        filtered, report = exclude_golden(contaminated)
        self.assertNotIn("conv_leak", set(filtered["conversation_id"]))
        self.assertGreaterEqual(
            (report["removed"]["exact_text"] or 0) + (report["removed"]["near_duplicate"] or 0), 1)

if __name__ == "__main__":
    unittest.main(verbosity=2)
