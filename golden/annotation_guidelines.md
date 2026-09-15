# Annotation guidelines

How to label a batch. Read **`taxonomy_v2.md`** alongside this — it is the frozen
label set and its tie-breakers decide every hard case.

> **v2 is frozen.** The labels and definitions will not change while a batch is being
> annotated, so labels can never be tuned to the evaluation. If you meet a message the
> taxonomy genuinely cannot handle, use `OTHER` and leave a note — do not invent a
> label.

**Time estimate:** ~150 messages at 10–20 seconds each, so 30–50 minutes. Take breaks;
fatigue shows up as label drift, and the test–retest check will find it.

---

## The workflow

1. Open `b01_pilot_blank.csv`.
2. **Save it immediately as `b01_pilot_labelled.csv`.** Work in that copy.
3. Fill the six label columns. Leave the first six columns untouched.
4. Tell me when it's done. I never write to `*_labelled.csv` — your work cannot be
   overwritten by a re-run.

**Open it in a spreadsheet, not a plain text editor.** The file is UTF-8 with BOM so
Excel renders emoji correctly, and every field is quoted so commas inside tweets do
not break the columns.

---

## Columns you fill

| Column | Values | Required |
| --- | --- | --- |
| `primary_intent` | one of the **11** intent names, or `OTHER`, or `UNCLEAR` | **yes** |
| `secondary_intent` | another intent name | only for genuine multi-intent |
| `confidence` | `high` / `medium` / `low` | **yes** |
| `is_ambiguous` | `yes` or blank | when two readings are defensible |
| `needs_discussion` | `yes` or blank | when you want me to look |
| `notes` | free text | whenever useful, and always with `OTHER` |

Type intent names **exactly** as in `taxonomy_v2.md` (lowercase, underscores). The
analysis tool validates spelling and will list anything it does not recognise rather
than guessing.

---

## Core rules

**1. Label the *request*, not the tone.**
Most messages are annoyed. Anger is not an intent. *"Your bag handling is a disgrace,
where is my suitcase"* is `baggage`, not `general_dissatisfaction`. Only use
`general_dissatisfaction` when there is **no actionable request at all**.

**2. Label what the customer wants, not what the airline should do.**
*"My flight is cancelled"* is `flight_cancellation_rebooking` even if they never
explicitly ask to be rebooked.

**3. One primary intent, even when it feels close.**
Pick the dominant one and set `is_ambiguous = y`. Do not leave `primary_intent` blank
— a blank row is dropped from the analysis and your reading is lost. If you truly
cannot choose, that is what `UNCLEAR` is for.

**4. Use `secondary_intent` sparingly.**
Only when a message carries two genuinely separate requests: *"Flight delayed 3 hrs
AND they lost my bag"* → primary `flight_delay`, secondary `baggage`. Not for "this
is a bit like that other intent" — that is `is_ambiguous`.

**5. `confidence` is about *your* certainty, not message clarity.**
Use `low` freely. It is a signal, not a failure. If an intent collects many `low`
ratings, its **definition** is unclear and I will rewrite it.

**6. `OTHER` vs `UNCLEAR` — different things.**
- `OTHER` = "I know exactly what they want; the taxonomy has no slot for it."
  **Always add a note naming the missing intent.** This is the single most valuable
  signal in the pilot.
- `UNCLEAR` = "I cannot tell what they want." A property of the message.

**7. Do not skip hard examples.**
Short ones, emoji-only ones, angry ones, non-English ones, ones that are only a
mention and a link. They are real traffic. Skipping them would quietly inflate every
downstream metric. Label them `UNCLEAR` if that is the honest answer.

**8. Judge the opening message alone.**
Do not look up the rest of the conversation. The live agent will only have this
message, so labelling with extra context would produce a taxonomy the system cannot
reproduce.

---

## Boundaries and edge cases

### The v2 boundary questions

When two intents both look plausible, these decide it. They are the same rules as in
`taxonomy_v2.md`.

| Pair | Ask |
| --- | --- |
| delay vs cancellation | Still on the original flight? |
| baggage vs booking fees | Is a bag involved? *(bag wins, even for fees)* |
| seating vs boarding | Where you sit, or when you board? |
| loyalty vs booking fees | The programme/account, or one booking? |
| **praise vs commentary (first)** | **Is the author a customer, or AA staff/crew?** |
| praise vs commentary (second) | Is service being evaluated? |
| dissatisfaction vs OTHER | Is there an actionable issue? |
| OTHER vs commentary | Is there a support issue at all? |
| OTHER vs UNCLEAR | Can you tell what they want? |

**The author-identity rule, new in v2.** If a message clearly reads as written by AA
staff, crew or the company itself — *"our employees"*, *"privilege to serve"*, *"our
team"* — label it `non_support_commentary` **even when it praises AA**. It is not
customer feedback. This is a **textual heuristic**: the dataset has no author-role
metadata, so judge only from self-identifying wording, and do not infer someone's
employer from enthusiasm alone.

---

## Edge cases

| Situation | Do this |
| --- | --- |
| Baggage **fee** complaint | `baggage` (v2 rule: bag wins). Flag `is_ambiguous` only if genuinely torn |
| Praise that also reports a problem | Label the problem; praise is incidental |
| Pure thanks after a resolved issue | `praise_and_compliment` |
| Only a mention and a URL, no words | `UNCLEAR` |
| Non-English, and you can read it | Label the intent normally |
| Non-English, and you cannot | `UNCLEAR` + note the language |
| A question about someone else's flight | Label the underlying intent |
| Journalist, plane-spotter or marketing chatter | `OTHER` + note (likely a real gap) |
| Two customers arguing, brand incidental | `OTHER` + note |
| Threat of legal action or a safety incident | Label the intent + `needs_discussion = y` |

---

## What I will do with this

After you return the file I will report the distribution, confidence spread, and the
signals below.

**These thresholds are now quality-review signals, not taxonomy revision triggers.**
They served as revision triggers during the pilot, when the taxonomy was still a
candidate. **v2 is frozen**, so crossing one of these no longer changes the label set
mid-batch — it flags where labelling was hard and what to examine once the batch is
complete.

| Signal | Threshold | What it now means |
| --- | --- | --- |
| `OTHER` rate | > 10% | Real issues are falling outside the taxonomy — review after the batch |
| Intent share of the random stratum | < 2% | Rare label; note it, but do not drop on one batch |
| Non-`high` confidence within an intent | > 30% | That definition is hard to apply in practice |
| Pair co-occurring as primary/secondary | > 15% | Those two intents overlap in practice |
| `UNCLEAR` rate | > 15% | Opening messages alone may lack sufficient context |

Any change to the taxonomy that these signals suggest happens **after** a batch is
finished and is recorded as a decision — never silently, and never mid-annotation.

---

## What I will not do

- Assign, infer, suggest or auto-fill any label.
- Call an LLM at any point in this phase.
- Use your golden labels for training, retrieval, prompt examples or threshold
  tuning. They are for evaluation only.
- Overwrite your `*_labelled.csv`.

---

## About the sample

150 messages in two strata, recorded per row in `sampling_stratum`:

- **`random` (~120)** — uniform random, month-proportional across Oct/Nov/Dec 2017 so
  no single incident dominates. **Only this stratum gives honest frequency
  estimates.**
- **`targeted` (~30)** — keyword-probed to guarantee rare intents appear at all. Pure
  random might return zero `loyalty_and_lounge` examples. **Deliberately
  unrepresentative** and never used for frequency estimates.

The strata are kept separate in every report. Sampling is not based on Phase 4
cluster assignments, since that partition was measurably unreliable.
