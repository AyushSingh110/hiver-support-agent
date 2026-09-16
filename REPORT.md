# Hiver take-home — AI support agent for @AmericanAir

One brand from *Customer Support on Twitter*. The system classifies each incoming message,
drafts a reply grounded in how the brand answered similar messages, and decides
**auto-handle or escalate, with a reason**. The priority was evidence over features: every
number below comes from a frozen, reproducible run (`README.md`, "Reproduction"). Decision
references (Dnn) point to `docs/DECISION_LOG.md`.

## 1. Problem framing: what "good" means

A support agent that answers fluently but invents a refund is worse than one that says
nothing. So "good" means, in order:

1. **Never state what the evidence does not support**, such as actions taken, amounts,
   policies or links.
2. **Hand anything uncertain to a human**, with a reason a reviewer can audit.
3. **Be useful when it does answer:** relevant, with a concrete next step, and asking only
   for what is needed.
4. **Be measurable.** A held-out, leakage-controlled set of manually annotated intent labels, baselines for every
   component, and honest statements of what each number does *not* show.

## 2. What was deliberately not built

- **No fine-tuning, no embeddings, no hosted LLM.** TF-IDF plus a local `llama3.1:8b` via
  Ollama keeps everything reproducible offline (D24).
- **No UI or serving layer.** The deliverable is the pipeline and its evaluation.
- **No LLM-as-judge metric.** It was built and failed its own pre-declared validation, so no
  judge score is reported (§10, D46).
- **No outcome prediction.** The data has no resolution labels, and a two-turn exchange
  (61% of the corpus) never shows whether the customer was helped.

## 3. Why AmericanAir

The 25 largest brands went through four gates (volume, public resolution, language,
support purity), then a weighted score (D22, D23).

- **AmericanAir** has 24,429 clean conversations, 21.1% DM deflection, an 8.3% URL rate and
  99.2% English, so it answers **in prose, in public**.
- **AmazonHelp** is three times larger but 18.1% non-English, with 65.6% of replies being
  links.
- **AppleSupport** pushes 64.4% of conversations to DMs, where the resolution is invisible.
- **The weighted score ranked AmericanAir sixth, and I did not adjust the weights to change
  that.** The top scorer, GWRHelp, is a regional rail operator with about four intents.

## 4. System architecture

```
customer message ──► sanitise ──► intent: TF-IDF + logistic regression (weak labels) ─┐
        │                                                                             │
        └──► retrieval: TF-IDF over 23,722 historical conversations, k = 5 ───────────┤
                                                                                      ▼
            generation: llama3.1:8b, temperature 0, seed 42, JSON {reply, needs_more_information, evidence_used}
                                                                                      ▼
            deterministic checks G1–G7: empty · unsupported URL · unsupported number ·
            action claim · resolution claim · leaked @id · copying (≥ 0.90 similarity)
                                                                                      ▼
            escalation esc-v1: 7 ordered rules ──► auto_handle | escalate + reason codes
```

- **Escalation rules**, in order:
  1. generation unusable;
  2. a G2–G7 flag;
  3. an invalid evidence citation;
  4. fewer than 2 evidence items;
  5. **explicit human-review policy** (a staff complaint, or general dissatisfaction; a
     policy choice, not learned; D44);
  6. intent confidence < 0.1913;
  7. top-1 similarity < 0.1981 **and** a substantive reply.
- **A reply that asks for details is not escalated for weak evidence** (D45).
- **Both thresholds are 20th percentiles** on a golden-free pool of 2,000 conversations.
  They set how much traffic each rule flags; they were not tuned for accuracy, because no
  escalation outcome labels exist (D43).

## 5. Golden set and leakage controls

- **Labels:** 248 messages whose intent labels were **manually annotated and decided by one
  annotator** under a 13-label taxonomy (v2):
  - **148 pilot rows (b01), labelled first and relabelled under v2**, whose definitions were
    written after reading them;
  - **100 fresh rows (b02)**, labelled under the frozen v2: **the cleaner independent
    subset**.
- **Label reliability:** an intra-annotator retest of 45 items after a short 12–14-hour gap
  gave 84.4% raw agreement, **Cohen's κ 0.8281**, 95% CI [0.7039, 0.9254]. This is the same
  annotator, not inter-annotator agreement (D40).
- **Leakage controls:**
  - golden conversations, **all conversations of golden customers**, exact normalised text
    and near-duplicates are removed from the retrieval and training corpus (D39);
  - that removes 456 rows, leaving 24,178 → 23,722;
  - the evaluation harness re-checks all of this and stops on any violation;
  - gold labels are read only by an explicitly confirmed golden run.

## 6. Intent classification

- **Training data:** the classifier trains on **rule-based weak labels** only: 9,423 corpus
  messages (39.7%) matched by keyword rules (D41).
- **Its ceiling:** the rules cover 10 of the 13 labels, so **41 of the 248 golden rows
  (`OTHER`, `UNCLEAR`, `non_support_commentary`) cannot be predicted**. They stay in the
  headline figures.

| System | Accuracy, all 248 [95% CI] | Macro-F1, all 248 | Accuracy, b02 100 [95% CI] | Macro-F1, b02 |
| --- | --- | --- | --- | --- |
| Majority class | 0.117 [0.081, 0.157] | 0.016 | 0.100 [0.05, 0.16] | 0.014 |
| **TF-IDF + LR** | **0.472 [0.411, 0.532]** | **0.358** | **0.450 [0.35, 0.55]** | **0.322** |

- **Predictable labels only** (secondary view): accuracy 0.565 on 207 rows, 0.549 on b02's
  82 rows.
- **Per intent:** `general_dissatisfaction` has precision 1.00 but recall 0.17, and
  `boarding_and_gate` has F1 0.00 on b02. `flight_delay` and `flight_cancellation_rebooking`
  are the strongest (F1 ≈ 0.62).

## 7. Retrieval vs random evidence

Evidence carries only rule-based weak labels, so the measure below compares **gold query
intent with weak evidence labels**. **It is not retrieval accuracy.**

| | Gold intent vs weak label (all 248) | Queries with any gold match | b02 |
| --- | --- | --- | --- |
| **TF-IDF, k=5** | **39.83%** (188/472 weakly labelled items) | 38.3% | 40.80% |
| Random, same pool and exclusions | 12.13% (61/503) | 20.6% | 12.04% |

- **Evidence quality is modest:** top-1 cosine similarity has a median of 0.259, and only
  38% of retrieved items carry any weak label.
- **How k was chosen:** k=5 came from a rule declared in advance (use k=3 only if it is ≥5
  points more consistent). **That rule was applied to golden results** (D42), so the golden
  set informed a model choice, and the reported golden numbers inherit that.
  - Re-applied on the golden-free development pool, the same rule also picks k=5 (k=3
    73.65% vs k=5 72.95%). k stayed frozen.

## 8. Generation and G1–G7

These are the recorded 248 generations (`artifacts/phase6/`); no reply was regenerated
for evaluation.

| | LLM | Baseline B (fixed generic reply) | Baseline A (top-1 echo) |
| --- | --- | --- | --- |
| Output parsed | 238 / 248 (95.97%) | — | — |
| G1 / G2 / G3–G6 / G7 | 10 / 1 / 0 / 1 | 0 | G7 on 247 |
| **All checks passed** | **236 / 248 (95.16%)** | 248 (100%) | 1 (0.4%) |

- **What the pass rates don't mean:**
  - **95.16% is a pattern-check pass rate, not correctness.**
  - Baseline B passes because a fixed sentence contains nothing the checks look for.
  - Baseline A fails because it copies by construction.
- **Parse failures:** all 10 put an intent name or a phrase in `evidence_used` (§11).
- **Output format:** 40.8% of parsed outputs gave ranks as strings. These were converted and
  counted, never hidden. Every citation fell within range.
- **`needs_more_information`** was true in 56.7% of parsed outputs.
- **Latency:** median 12.8 s per reply on a local laptop, which says nothing about
  production throughput.
- **Run robustness:**
  - 49 rows were retried once after infrastructure failures (47 out-of-memory errors);
  - regenerating 5 fixed rows reproduced 4 byte for byte, the exception being b01_0001.

## 9. Escalation

**80 of the 248 rows escalated (32.3%); b02: 34%.** There are no outcome labels, so this is
system behaviour, not success.

| Reason code (rows can have several) | Rows |
| --- | --- |
| low intent confidence | 49 (19.8%) |
| explicit human-review policy | 15 (6.0%) |
| weak grounding for a substantive reply | 11 (4.4%) |
| generation unusable (parse failure) | 10 (4.0%) |
| grounding check failed (G2 or G7) | 2 (0.8%) |

- **By classifier outcome:** 20.5% of rows escalated when the classifier was right,
  42.8% when it was wrong.
- **Unpredictable gold labels:** 43.9% of the 41 such rows escalated. The only route for
  them is low confidence.
- **Clarifying questions:** 135 replies (54.4%) ask the customer for details. They carry the
  informational code `awaiting_customer_details`, which never escalates on its own.

## 10. Reply-quality evaluation status

**LLM-as-judge: failed, not used.** Four configurations each failed pre-declared checks on
non-golden synthetic cases:
- j1, j2 and j3 with qwen2.5:7b;
- j3 with qwen3:8b.

No judge ever reliably marked invented refunds, policies or other customers' details as
unsafe (D46).

**Reply ratings (round 1 only): single-annotator, developer-made, AI-assisted.** One
annotator rated 40 sampled golden items × 3 systems on a 1–5 rubric: 119 ratings, since one
LLM reply was empty (D47, D48). **This is not an independent human evaluation.** Read these
with care:
- **The rater is the system's developer, and the ratings were AI-assisted.**
- Blinding was weak: Baseline B is a fixed sentence and Baseline A repeats evidence item 1.
- They are **not independent human ratings and not validated.** No reply-quality agreement
  statistic exists. The Phase 5 κ in §5 measures intent labels, not reply ratings.
- **The blinded retest was built but not completed before submission** (the 72-hour minimum
  gap had not passed).

| Mean score (1–5) | Relevance | Helpfulness | Groundedness | Info request | Claim safety |
| --- | --- | --- | --- | --- | --- |
| LLM (n=39) | 4.85 | 4.41 | 4.64 | 4.72 | 4.79 |
| Baseline A (n=40) | 3.73 | 3.05 | 3.35 | 3.85 | 3.98 |
| Baseline B (n=40) | 3.50 | 1.90 | 4.88 | 2.70 | 5.00 |

- **These comparisons are descriptive, not a ranking.**
- **The most useful finding is a gap in the deterministic checks:** **2 of the 27
  auto-handled LLM replies were rated claim safety ≤2, and G2–G7 flagged neither.**

## 11. Top 5 failure modes

1. **Soft promises and invented procedures pass the checks.**
   - b02_0096 (auto-handled): *"We'd like to help you with a more comprehensive
     compensation"*, in reply to a customer disputing a partial compensation offer.
   - b02_0025 (auto-handled): an invented account of how check-in staff handle unpaid bag
     fees.
   - *Hypothesis:* G4 matches completed actions ("we have refunded"), not offers or process
     descriptions, and the model fills gaps with plausible airline policy.
2. **Invented URL and policy.** b01_0057 gives `aa.com/i18n/travel-info/baggage` and
   baggage-allowance rules found in no evidence. G2 caught the URL (so the row escalated);
   nothing catches the rules.
   - *Hypothesis:* prior knowledge from pre-training overrides the rule "only use what is
     shown".
3. **Schema confusion.** 10 outputs put `"flight_delay"` or `"historical example 1"` in
   `evidence_used`, so the whole reply was discarded (for example b01_0007).
   - *Hypothesis:* the prompt shows a labelled "PREDICTED ISSUE TYPE" next to numbered
     evidence, and a small model conflates the two.
4. **The weak-label ceiling.** 41 gold rows can never be predicted, and
   `general_dissatisfaction` recall is 0.17. Its rules match only explicit phrases ("worst
   airline", "never again"), while real dissatisfaction is phrased freely.
5. **Lexical retrieval.**
   - The median top-1 similarity is 0.26.
   - Templated check-in posts get near-duplicate matches from other customers (b01_0019,
     0.94).
   - Paraphrases share no tokens, so relevant history is missed, and the reply leans on the
     model's prior instead.

## 12. What is misleading about my headline number?

- **"Intent accuracy 47%"**
  - It is measured against **one annotator's** labels, and 148 of the 248 rows are pilot
    rows **relabelled under definitions written after reading them**. The clean b02 figure
    is 45%, with a 20-point-wide interval.
  - It **mixes two failures:** 41 rows are impossible by construction, and the classifier
    learned from keyword rules, so it partly re-learns the rules.
- **"95% pass all safety checks"**
  - This counts **absent patterns, not correct replies.**
  - The developer-made reply ratings found unsafe auto-handled replies inside that 95%.
  - The G7 row and Baseline B's 100% are artefacts of how those checks and that baseline
    are built.
- **"Retrieval 3× better than random"**
  - It compares gold labels with **weak** labels on the 38% of evidence that has any label,
    and k was picked on golden results.
- **"LLM replies rated 4.4–4.9"**
  - These are AI-assisted ratings by the developer, with weak blinding and no completed
    retest. They are not independent evidence.
- **"32% escalated"**
  - This is a traffic level set by percentile thresholds. With no outcome labels, it does
    not say whether the right 32% were escalated.

## 13. One more week

1. **A second, independent annotator** on b02 and on the reply ratings, **and the completed
   blinded retest.**
2. **Claim detection beyond regex:** a small extraction step that lists every promise, offer
   and procedure in a reply and checks each against the evidence. Evaluate it on the
   cases flagged in the reply ratings first.
3. **Prompt schema fix:** remove the intent label from the generation prompt and rerun on
   the development pool. That tests failure mode 3 without touching the golden set.
4. **Retrieval ablation:** embeddings vs TF-IDF on the development pool, under the same
   leakage rules.
5. **Escalation outcome labels:** a human labels "should this have been escalated?" on a
   blinded sample, which would turn §9 from a description into an error rate.

## 14. Key non-obvious decisions

| Decision | Why it matters |
| --- | --- |
| Broadcasts excluded by participant structure, not fan-out (D15) | a fan-out threshold would have deleted real support threads |
| Brand gates before scoring; the score not overridden (D22, D23) | avoided picking a volume-rich but unusable brand |
| Only hand-made artefacts tracked; generated data ignored (D13, D27, with D49 as the exception) | reproducibility without committing 500 MB |
| Revision triggers declared before labelling (D29, D33) | taxonomy v2 changes rest on evidence, not taste |
| Two-layer leakage: conversation **and** customer (D39) | the same customer's other threads would otherwise leak |
| Weak supervision, with its ceiling reported (D41) | no golden label trains anything |
| k chosen by a pre-declared rule, and the golden exposure disclosed (D42) | the naive "best k" would have been k=1 |
| Thresholds as traffic sizes on a golden-free pool (D43) | no accuracy claim without outcome labels |
| Human-review policy stated as policy (D44) | policy is not presented as a learned signal |
| All escalation reasons collected; "needs details" ≠ "unsafe" (D45) | a clarifying reply is not escalated for thin evidence |
| Recorded generations are the run of record; retries logged (6C, D49) | the evaluation measures a fixed, auditable run |
| The LLM judge dropped after failing its own validation (D46) | no unvalidated score in the headline |
| Reply-rating analysis fixed before rating; chronology disclosed (D47, D48) | the deviations are on the record, not hidden |
| Data files stored byte-exact (D49) | recorded SHA-256 values hold on any clone |
