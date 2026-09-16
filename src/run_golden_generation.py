"""Held-out golden generation run. Data collection only - no tuning, no evaluation."""
import sys, io, json, time, traceback
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, r"C:/Users/ASUS/Desktop/hiver-support-agent")

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config, llm
from src.leakage import assert_isolated
from src.retrieve import Retriever, load_corpus
from src.taxonomy import ALL_LABELS, normalise
from src.generate_reply import (PROMPT_VERSION, build_prompt, generate_llm,
                                generate_baseline_echo, generate_baseline_template)

OUT = config.CORPUS_DIR / "golden_generation_records.jsonl"

golden = pd.read_csv(config.GOLDEN_SET, dtype=str, keep_default_na=False, encoding="utf-8-sig")
corpus = load_corpus()
assert_isolated(corpus)

training = pd.read_parquet(config.CORPUS_DIR / "weak_training_set.parquet")
assert_isolated(training)
training = training[training["weak_label"].isin(ALL_LABELS)]

# Intent model: weak labels only. Gold labels are never used for fitting.
clf = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.6,
                              sublinear_tf=True, max_features=50_000)),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=config.RANDOM_SEED)),
])
clf.fit(training["customer_message"].map(normalise), training["weak_label"])
print(f"classifier fitted on {len(training)} weakly-labelled rows", flush=True)

retriever = Retriever(corpus)
started_all = time.perf_counter()
n_fail = 0

with open(OUT, "w", encoding="utf-8") as fh:
    for i, row in enumerate(golden.itertuples(index=False), start=1):
        rec = {
            "annotation_id": row.annotation_id,
            "batch": row.batch,
            "conversation_id": row.conversation_id,
            "customer_author_id": row.customer_author_id,
            "customer_message": row.text,
            "eval_only_gold_intent": row.primary_intent,      # evaluation only, never used here
            "eval_only_gold_secondary": row.secondary_intent,
            "eval_only_gold_ambiguous": row.ambiguous,
            "model": config.GENERATION_MODEL,
            "prompt_version": PROMPT_VERSION,
            "retrieval_k": config.RETRIEVAL_TOP_K,
            "temperature": 0.0,
            "seed": config.RANDOM_SEED,
        }
        try:
            message = row.text
            rec["predicted_intent"] = clf.predict([normalise(message)])[0]

            evidence = retriever.retrieve(
                message, config.RETRIEVAL_TOP_K,
                exclude_conversation_id=row.conversation_id,
                exclude_customer_id=row.customer_author_id,
            )
            rec["evidence"] = [
                {"rank": e["rank"], "conversation_id": e["conversation_id"],
                 "tweet_id": int(e["tweet_id"]), "similarity": e["similarity"],
                 "weak_intent": e["historical_weak_intent"],
                 "evidence_shape": e["evidence_shape"], "n_turns": e["n_turns"]}
                for e in evidence
            ]
            rec["evidence_count"] = len(evidence)
            rec["top1_similarity"] = evidence[0]["similarity"] if evidence else None
            rec["top5_similarities"] = [e["similarity"] for e in evidence]

            result = generate_llm(message, rec["predicted_intent"], evidence)
            rec["llm"] = result
            # Cache hit, so no second inference: retained for audit.
            try:
                raw = llm.complete(build_prompt(message, rec["predicted_intent"], evidence),
                                   model=config.GENERATION_MODEL, use_cache=True)
                rec["raw_model_output"] = raw["text"]
            except Exception as err:
                rec["raw_model_output"] = None
                rec["raw_capture_error"] = str(err)

            rec["baseline_b_template"] = generate_baseline_template(message, evidence)
            rec["baseline_a_echo"] = generate_baseline_echo(message, evidence)
            rec["row_error"] = None
        except Exception as err:
            n_fail += 1
            rec["row_error"] = f"{type(err).__name__}: {err}"
            rec["row_traceback"] = traceback.format_exc()[-1500:]
            print(f"[FAIL {i}] {rec['annotation_id']} {rec['row_error']}", flush=True)

        fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        fh.flush()
        if i % 10 == 0 or i == len(golden):
            print(f"[{i}/{len(golden)}] elapsed {time.perf_counter()-started_all:.0f}s "
                  f"failures={n_fail}", flush=True)

print(f"DONE rows={len(golden)} failures={n_fail} "
      f"wall_clock_seconds={time.perf_counter()-started_all:.1f}", flush=True)
print("written", OUT, flush=True)
