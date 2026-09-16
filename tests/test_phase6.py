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


if __name__ == "__main__":
    unittest.main(verbosity=2)
