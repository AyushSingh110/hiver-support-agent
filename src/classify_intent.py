from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from src import config
from src.leakage import assert_isolated
from src.taxonomy import ALL_LABELS, normalise

TFIDF_SETTINGS = {
    "ngram_range": (1, 2),
    "min_df": 3,
    "max_df": 0.6,
    "sublinear_tf": True,
    "max_features": 50_000,
}


def load_training() -> pd.DataFrame:
    path = config.CORPUS_DIR / "weak_training_set.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Weak training set missing at {path}; run src.build_corpus first")
    training = pd.read_parquet(path)
    assert_isolated(training)
    return training


def load_golden() -> pd.DataFrame:
    return pd.read_csv(config.GOLDEN_SET, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def build_pipeline(seed: int) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_SETTINGS)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
    ])


def score(name: str, truth: np.ndarray, predicted: np.ndarray) -> dict:
    report = classification_report(
        truth, predicted, labels=ALL_LABELS, output_dict=True, zero_division=0
    )
    return {
        "system": name,
        "n": int(len(truth)),
        "accuracy": round(float((truth == predicted).mean()), 4),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 4),
        "per_intent": {
            label: {
                "support": int(report[label]["support"]),
                "precision": round(float(report[label]["precision"]), 4),
                "recall": round(float(report[label]["recall"]), 4),
                "f1": round(float(report[label]["f1-score"]), 4),
            }
            for label in ALL_LABELS
        },
    }


def non_zero_confusion(truth: np.ndarray, predicted: np.ndarray) -> list[dict]:
    matrix = confusion_matrix(truth, predicted, labels=ALL_LABELS)
    return [
        {"true": ALL_LABELS[i], "predicted": ALL_LABELS[j], "count": int(matrix[i][j])}
        for i in range(len(ALL_LABELS))
        for j in range(len(ALL_LABELS))
        if matrix[i][j] > 0
    ]


def evaluate_all(golden: pd.DataFrame, predictions: dict[str, np.ndarray]) -> dict:
    truth = golden["primary_intent"].to_numpy()
    b02 = (golden["batch"] == "b02").to_numpy()
    results = {}
    for name, predicted in predictions.items():
        results[name] = {
            "all_248": score(name, truth, predicted),
            "independent_b02_100": score(name, truth[b02], predicted[b02]),
            "confusion_all_248": non_zero_confusion(truth, predicted),
        }
    return results


def main() -> int:
    print("[1/4] Loading weakly-labelled training data ...", flush=True)
    training = load_training()
    trainable = training[training["weak_label"].isin(ALL_LABELS)]
    print(f"      {len(trainable):,} rows across {trainable['weak_label'].nunique()} intents",
          flush=True)

    print("[2/4] Loading the frozen golden set ...", flush=True)
    golden = load_golden()
    if len(golden) != 248:
        raise ValueError(f"Golden set must be 248 rows, found {len(golden)}")

    print("[3/4] Fitting baselines ...", flush=True)
    x_train = trainable["customer_message"].map(normalise)
    y_train = trainable["weak_label"].to_numpy()
    x_golden = golden["text"].map(normalise)

    majority = trainable["weak_label"].value_counts().idxmax()
    predictions = {
        "baseline_0_majority_class": np.array([majority] * len(golden)),
    }

    pipeline = build_pipeline(config.RANDOM_SEED)
    pipeline.fit(x_train, y_train)
    predictions["baseline_1_tfidf_logreg"] = pipeline.predict(x_golden)

    print("[4/4] Scoring and writing results ...", flush=True)
    results = evaluate_all(golden, predictions)

    trainable_labels = sorted(set(y_train))
    unreachable = [label for label in ALL_LABELS if label not in trainable_labels]
    golden_unreachable = int(golden["primary_intent"].isin(unreachable).sum())

    stats = {
        "taxonomy_version": "v2",
        "training_rows": len(trainable),
        "training_label_distribution": trainable["weak_label"].value_counts().to_dict(),
        "majority_class": majority,
        "tfidf_settings": TFIDF_SETTINGS,
        "labels_absent_from_training": unreachable,
        "golden_rows_with_unreachable_label": golden_unreachable,
        "golden_rows_with_unreachable_label_b02": int(
            golden.loc[golden["batch"] == "b02", "primary_intent"].isin(unreachable).sum()
        ),
        "results": results,
        "notes": (
            "Trained only on weak rule-based labels from the leakage-filtered corpus. "
            "The golden set was never used for fitting or threshold selection."
        ),
    }
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "phase6_intent_baselines.json").write_text(
        json.dumps(stats, indent=2, default=str), encoding="utf-8"
    )

    for name, result in results.items():
        every = result["all_248"]
        clean = result["independent_b02_100"]
        print(f"\n  {name}")
        print(f"    all 248 : accuracy {every['accuracy']:.4f}  macro-F1 {every['macro_f1']:.4f}")
        print(f"    b02 100 : accuracy {clean['accuracy']:.4f}  macro-F1 {clean['macro_f1']:.4f}")
    print(f"\nResults: {config.REPORTS_DIR / 'phase6_intent_baselines.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
