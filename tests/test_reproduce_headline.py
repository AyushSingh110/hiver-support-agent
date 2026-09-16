"""The reproduction runner's final summary reads its values and keeps the required caveats."""
import unittest

from src.reproduce_headline import headline_summary


def pct(value):
    return {"pct": value, "count": 7, "total": 9}


def synthetic_inputs():
    def classifier(accuracy):
        return {"metrics": {"accuracy": accuracy, "macro_f1": accuracy / 10}}

    def subset(offset):
        return {
            "classifier": {"baseline_0_majority_class": classifier(0.1111 + offset),
                           "baseline_1_tfidf_logreg": classifier(0.6666 + offset)},
            "retrieval": {system: {"gold_intent_vs_weak_label_agreement": {
                "agreement_pct_of_weakly_labelled_items": value + offset,
                "queries_with_any_match": pct(value + 1 + offset)}}
                for system, value in (("tfidf", 71.25), ("random", 3.5))},
        }

    all_rows, b02 = subset(0), subset(0.002)
    evaluation = {
        "classifier": {"all_248": all_rows["classifier"], "independent_b02_100": b02["classifier"]},
        "retrieval": {"all_248": all_rows["retrieval"], "independent_b02_100": b02["retrieval"]},
        "generation": {"all_248": {
            "parsed": pct(88.88), "all_checks_passed": pct(77.77),
            "flags": {"G2_unsupported_url": pct(0) | {"count": 13},
                      "G7_excessive_copying": pct(0) | {"count": 17}},
            "needs_more_information_true": pct(44.44)}},
        "escalation": {"all_248": {
            "escalate": pct(33.33), "auto_handle": pct(66.67),
            "escalation_rate_when_classifier_correct": pct(12.34),
            "escalation_rate_when_classifier_wrong": pct(56.78)}},
    }
    corpus = {"eligibility": {"start": 5432}, "retrieval_corpus_rows": 4321}
    phase5 = {"n": 31, "agreement": {"raw_agreement_pct": 81.1, "cohens_kappa": 0.7777},
              "bootstrap": {"ci_lower": 0.6111, "ci_upper": 0.9111}}
    round1 = {"provenance": {"counts": {"rated_responses": 101}},
              "test_retest": {"status": "not_completed"}}
    return evaluation, corpus, 6543, phase5, round1, 12.34, False


class HeadlineSummaryTests(unittest.TestCase):
    def setUp(self):
        self.text = headline_summary(*synthetic_inputs())
        self.lines = self.text.splitlines()

    def section(self, number):
        start = next(i for i, line in enumerate(self.lines) if line.startswith(f"{number}. "))
        end = next((i for i, line in enumerate(self.lines[start + 1:], start + 1)
                    if line[:1].isdigit() or line.startswith("=")), len(self.lines))
        return "\n".join(self.lines[start:end])

    def test_values_come_from_inputs_and_land_in_their_sections(self):
        expected = {
            1: ["6,543", "5,432", "4,321"],
            2: ["accuracy 0.1111", "accuracy 0.6666", "macro-F1 0.0667",
                "accuracy 0.1131", "accuracy 0.6686"],
            3: ["71.25%", "72.25%", "3.5%", "4.5%", "71.252%", "3.502%"],
            4: ["88.88%", "77.77%", "13 / 17", "44.44%"],
            5: ["33.33% / 66.67%", "12.34%", "56.78%"],
            6: ["n=31", "81.1%", "kappa 0.7777", "[0.6111, 0.9111]", "101 ratable", "not completed"],
            7: ["12.3 s", "byte-identical to the recorded run: False"],
        }
        for number, values in expected.items():
            section = self.section(number)
            for value in values:
                self.assertIn(value, section, f"section {number}")

    def test_required_caveats_are_present(self):
        for phrase in ("REPORT.md is authoritative", "not retrieval accuracy",
                       "pattern checks", "not correctness", "intra-annotator",
                       "single-annotator, developer-made, AI-assisted",
                       "not independent human ratings", "failed its pre-declared validation",
                       "not used (D46)"):
            self.assertIn(phrase, self.text)

    def test_misleading_labels_are_absent(self):
        generation, retrieval = self.section(4).lower(), self.section(3).lower()
        self.assertNotIn("accuracy", generation)
        self.assertEqual(retrieval.count("accuracy"), 1)
        self.assertIn("not retrieval accuracy", retrieval)
        rating_line = next(line for line in self.lines if "round-1 reply ratings" in line)
        self.assertIn("(not independent human ratings)", rating_line)
        self.assertEqual(self.text.count("independent human"), 1)
        self.assertNotIn("inter-annotator agreement", self.text)


if __name__ == "__main__":
    unittest.main()
