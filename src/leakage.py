from __future__ import annotations

from functools import lru_cache

import pandas as pd

from src import config
from src.taxonomy import normalise

# Enforceable layers only. See docs/DECISION_LOG.md D39 for what cannot be detected:
# one human with several accounts, the same incident from different customers, and
# cross-thread continuation invisible to the identifiers.
UNENFORCEABLE_RISKS = [
    "one human operating multiple anonymised accounts",
    "semantically identical incidents reported by different customers",
    "cross-thread continuation not captured by conversation or customer identifiers",
]


def _read_id_file(path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


@lru_cache(maxsize=1)
def golden_conversation_ids() -> frozenset[str]:
    return frozenset(_read_id_file(config.GOLDEN_CONVERSATION_EXCLUSIONS))


@lru_cache(maxsize=1)
def golden_customer_ids() -> frozenset[str]:
    return frozenset(_read_id_file(config.GOLDEN_CUSTOMER_EXCLUSIONS))


@lru_cache(maxsize=1)
def golden_normalised_texts() -> tuple[frozenset[str], tuple[frozenset[str], ...]]:
    golden = pd.read_csv(config.GOLDEN_SET, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    normalised = [normalise(text) for text in golden["text"]]
    token_sets = tuple(frozenset(text.split()) for text in normalised if text)
    return frozenset(t for t in normalised if t), token_sets


def _max_jaccard(tokens: frozenset[str], golden_tokens: tuple[frozenset[str], ...],
                 threshold: float) -> float:
    if not tokens:
        return 0.0
    size = len(tokens)
    # Jaccard >= t forces both set sizes into [t*size, size/t]; cheap pruning.
    low, high = threshold * size, size / threshold
    best = 0.0
    for other in golden_tokens:
        if not low <= len(other) <= high:
            continue
        shared = len(tokens & other)
        if not shared:
            continue
        score = shared / (size + len(other) - shared)
        if score > best:
            best = score
            if best >= 1.0:
                break
    return best


def exclude_golden(
    frame: pd.DataFrame,
    *,
    conversation_column: str = "conversation_id",
    customer_column: str | None = "customer_author_id",
    text_column: str | None = "text",
    check_near_duplicates: bool = True,
    threshold: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Remove every row that leaks the golden set, and report what was removed.

    Four enforceable layers: conversation id, customer id, exact normalised text,
    near-duplicate text. Returns the filtered frame and an audit dict so callers can
    assert isolation rather than assume it.
    """
    threshold = config.NEAR_DUPLICATE_THRESHOLD if threshold is None else threshold
    if conversation_column not in frame.columns:
        raise KeyError(f"{conversation_column!r} not in frame; cannot enforce leakage control")

    report = {"rows_in": len(frame), "threshold": threshold, "removed": {}}
    keep = pd.Series(True, index=frame.index)

    by_conversation = frame[conversation_column].isin(golden_conversation_ids())
    report["removed"]["conversation_id"] = int((keep & by_conversation).sum())
    keep &= ~by_conversation

    if customer_column and customer_column in frame.columns:
        by_customer = frame[customer_column].astype(str).isin(golden_customer_ids())
        report["removed"]["customer_author_id"] = int((keep & by_customer).sum())
        keep &= ~by_customer
    else:
        report["removed"]["customer_author_id"] = None

    if text_column and text_column in frame.columns:
        exact_texts, golden_tokens = golden_normalised_texts()
        normalised = frame.loc[keep, text_column].map(normalise)

        by_exact = normalised.isin(exact_texts)
        exact_ids = normalised.index[by_exact]
        report["removed"]["exact_text"] = int(len(exact_ids))
        keep.loc[exact_ids] = False

        if check_near_duplicates:
            remaining = normalised.loc[keep.loc[normalised.index]]
            scores = remaining.map(lambda t: _max_jaccard(frozenset(t.split()), golden_tokens, threshold))
            near_ids = scores.index[scores >= threshold]
            report["removed"]["near_duplicate"] = int(len(near_ids))
            report["near_duplicate_examples"] = [
                {"index": str(i), "jaccard": round(float(scores[i]), 4)} for i in near_ids[:10]
            ]
            keep.loc[near_ids] = False
        else:
            report["removed"]["near_duplicate"] = None
    else:
        report["removed"]["exact_text"] = None
        report["removed"]["near_duplicate"] = None

    filtered = frame[keep]
    report["rows_out"] = len(filtered)
    report["rows_removed"] = len(frame) - len(filtered)
    report["unenforceable_risks"] = UNENFORCEABLE_RISKS
    return filtered, report


def assert_isolated(frame: pd.DataFrame, *, conversation_column: str = "conversation_id") -> None:
    """Fail loudly if any golden conversation survived into a development set."""
    overlap = set(frame[conversation_column]) & golden_conversation_ids()
    if overlap:
        raise AssertionError(
            f"{len(overlap)} golden conversations leaked into a development set: "
            f"{sorted(overlap)[:5]}"
        )
