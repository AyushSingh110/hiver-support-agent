# golden/

Hand-labelled evaluation data for the AmericanAir support agent.

## Why this directory is tracked in Git

Everything under `data/` and `reports/` is gitignored, because it can be regenerated
from `src/` and the raw dataset. **This directory is the exception.**

These labels are produced by a person reading messages one at a time. If they are
lost, they cannot be recreated — not by re-running code, not by an LLM, not from
the raw data. They are the most valuable and least reproducible artifact in the
project, and the only thing the evaluation ultimately rests on.

So `golden/` is committed. It is small (a few hundred KB) and irreplaceable.

## Files

| File | Written by | Purpose |
| --- | --- | --- |
| `taxonomy_v1.md` | hand | Candidate taxonomy, frozen for the pilot. **Not final** |
| `annotation_guidelines.md` | hand | Labelling rules and edge cases |
| `b01_pilot_blank.csv` | `src/build_annotation_batch.py` | 150 messages, label columns empty |
| `b01_pilot_labelled.csv` | **the annotator** | The filled-in copy |

Later, after the taxonomy is frozen: `taxonomy_v2.md` and the golden set itself.

## Rules the tooling enforces

- The blank batch is written with **every label column empty**. No code path fills
  them, and no LLM is called anywhere in this phase.
- `src/build_annotation_batch.py` **refuses to write** to any `*_labelled.csv` path,
  and refuses to overwrite an existing blank batch. Human work cannot be clobbered
  by a re-run.
- The analysis tool opens labelled files **read-only**.
- Original tweet text is never modified. `text_display` collapses newlines to ` / `
  so one message occupies one CSV row; the untouched original stays in
  `data/processed/conversation_turns.parquet`, joinable on `conversation_id`.

## How golden labels may and may not be used

**May:** evaluation only — intent accuracy, macro-F1, per-intent precision and
recall, confusion matrices, escalation metrics.

**May not:** training, retrieval corpus membership, few-shot prompt examples,
threshold tuning, or any other model-development step. The conversation IDs here
become an exclusion list that the retrieval corpus must subtract.

This separation is what makes the reported numbers mean anything.

## Reproducing the sample

```bash
python -m src.build_annotation_batch --batch b01
```

Deterministic given the same processed data and seed. The exact draw, seed, month
distribution and keyword probes are recorded in
`reports/phase5_b01_sampling_manifest.json`.

Re-running will **not** overwrite an existing blank batch — delete it deliberately if
you really intend to redraw, and be aware that doing so changes which messages the
annotator is asked to label.
