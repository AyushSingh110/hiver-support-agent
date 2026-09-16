"""Phase 6E LLM-as-judge (prompt version j3: one call per dimension).

Model-based evaluation, not human evaluation. No human reply ratings exist. The judge
failed its pre-declared non-golden validation with every prompt and model configuration
tested (j1, j2 and j3 with qwen2.5:7b; j3 with qwen3:8b; docs/DECISION_LOG.md D46), so it is
not used in the final evaluation and has never been run on golden rows.

Baseline B (a fixed sentence) and Baseline A (a verbatim echo of evidence) are
recognisable from their content, so blinding is partial: the judge is never told which
system wrote a reply, but it may infer it.

j1 and j2 scored all five dimensions in one call and let one poor property pull the
others down (docs/DECISION_LOG.md). j3 asks about one dimension per call, and each call
sees only the material that dimension needs.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.request
from contextlib import contextmanager
from string import Template

import numpy as np

from src import config, llm
from src.generate_reply import format_evidence, sanitise_evidence_text

JUDGE_PROMPT_VERSION = "j3"
DIMENSIONS = ("relevance", "groundedness", "helpfulness",
              "information_request_appropriateness", "claim_safety")
RATIONALE_MAX_WORDS = 40
SCORES = (1, 2, 3, 4, 5)
SEED = config.RANDOM_SEED
BOOTSTRAP_RESAMPLES = 2_000

SYSTEMS = ("llm_grounded", "baseline_b_template", "baseline_a_echo")

JUDGED = "judged"
NOT_JUDGED_EMPTY_REPLY = "not_judged_empty_reply"
JUDGE_PARSE_FAILURE = "judge_parse_failure"
JUDGE_INFRASTRUCTURE_ERROR = "judge_infrastructure_error"
DIMENSION_STATUSES = (JUDGED, NOT_JUDGED_EMPTY_REPLY, JUDGE_PARSE_FAILURE, JUDGE_INFRASTRUCTURE_ERROR)

ITEM_JUDGED = "judged"
ITEM_PARTIALLY_JUDGED = "partially_judged"
ITEM_FAILED = "failed"
ITEM_NOT_JUDGED_EMPTY_REPLY = "not_judged_empty_reply"

# Errors raised while reaching the model, as opposed to errors in what it returned.
INFRASTRUCTURE_ERRORS = (llm.LLMError, OSError, MemoryError, ValueError)

SANITY_REPORT = config.REPORTS_DIR / "phase6_judge_sanity_j3.json"
PROBE_REPORT = config.REPORTS_DIR / "phase6_judge_probe_j3.json"
SANITY_CACHE_DIR = config.PROJECT_ROOT / "data" / "llm_cache_judge_sanity"
PROBE_CACHE_DIR = config.PROJECT_ROOT / "data" / "llm_cache_judge_probe"


# --------------------------------------------------------------------------- j3 prompts

_HEADER = (
    "Rating task: j3/{dimension}\n"
    "You are rating ONE property of a draft reply written for an airline's customer-support team "
    "on Twitter. Other properties of the reply are rated in separate tasks, so ignore them "
    "completely. Use only the material shown. Do not assume any airline policy that is not shown, "
    "and do not invent facts.\n"
)
_MESSAGE = "CUSTOMER MESSAGE:\n$customer_message\n"
_MESSAGE_FOR_REFERENCE = (
    "CUSTOMER MESSAGE (for reference: details that appear here are this customer's own and are "
    "not borrowed):\n$customer_message\n"
)
_EVIDENCE = (
    "HISTORICAL SUPPORT EXAMPLES (how the support team replied to other customers with similar "
    "messages; they are not facts about this customer):\n$evidence\n"
)
_REPLY = "CANDIDATE REPLY:\n$reply\n"
_FOOTER = (
    "Return only this JSON object, with exactly these two keys:\n"
    '{"score": <integer 1-5>, "rationale": "<at most 40 words, about this property only>"}'
)

# Material each dimension may consider; nothing else reaches that call.
DIMENSION_MATERIAL = {
    "relevance": ("customer_message", "reply"),
    "groundedness": ("customer_message", "evidence", "reply"),
    "helpfulness": ("customer_message", "reply"),
    "information_request_appropriateness": ("customer_message", "reply"),
    "claim_safety": ("customer_message", "evidence", "reply"),
}

_DEFINITIONS = {
    "relevance": """Question: does the candidate reply address what this customer wrote?
Judge only whether the subject of the reply matches the customer's message. Do not consider whether the reply is accurate, safe, supported, useful, or whether it asks for information. A reply about the customer's issue is relevant even if it is wrong or unhelpful.
5 = directly addresses the customer's specific issue
4 = addresses the issue, with minor gaps
3 = generic, or only partly about the issue
2 = mostly about something else
1 = completely unrelated to the customer's message""",
    "groundedness": """Question: is the reply's approach consistent with how the support team replied in the historical examples, without borrowing another customer's specific details?
Compare the reply with the historical examples. Details that appear in this customer's own message are not borrowed. Do not consider whether the reply is on topic, useful or complete.
5 = the same kind of approach as the examples, and no details taken from other customers
4 = mostly the same kind of approach
3 = neither follows nor departs from the examples
2 = a noticeably different approach, or one minor detail taken from another customer
1 = contradicts the examples, or presents another customer's names, flight numbers or other specifics as this customer's""",
    "helpfulness": """Question: does the reply give this customer useful practical value or an appropriate next step?
Consider only what the customer can do, or can expect, after reading the reply. Do not consider whether statements in the reply are supported; that is rated separately. Do not consider tone or style.
5 = a clear, appropriate next step or useful information for this customer
4 = useful but slightly vague
3 = some value, but no clear next step
2 = little practical value
1 = no practical value for this customer""",
    "information_request_appropriateness": """Question: is what the reply asks the customer to provide appropriate and necessary for this customer's message?
Identify what the reply asks the customer for (it may ask for nothing) and what this message makes necessary (it may need nothing). Judge only how well the two match. Do not consider the rest of the reply's content or quality.
5 = asks for what is needed, or asks for nothing when nothing is needed
4 = a small unnecessary request or a small omission
3 = a somewhat unnecessary request, or a needed request is missing
2 = a clearly unnecessary request, or a clearly needed request is missing
1 = asks for sensitive or unrelated personal details the situation does not call for""",
    "claim_safety": """Question: does the candidate reply state anything as fact that is not supported by the customer message or the historical examples?
Check each concrete assertion in the reply: actions taken, refunds, compensation, amounts, times, policies, promises, URLs, names, flight numbers and other specific facts.
- A reply that makes no factual assertion scores 5.
- Being irrelevant, short or unhelpful does not make a reply unsafe.
- A reply about the customer's issue can still score low if it contains unsupported assertions.
- Apologies, expressions of sympathy and requests for information are not factual assertions.
Do not consider whether the reply is on topic, useful or well written.
5 = no unsupported assertion
4 = one vague or minor unsupported assertion
3 = an unsupported specific assertion of limited consequence
2 = an unsupported assertion that could mislead the customer
1 = asserts an action, refund, compensation, policy, amount or URL that nothing shown supports""",
}


def _compose(dimension: str) -> str:
    blocks = {"customer_message": _MESSAGE_FOR_REFERENCE if dimension == "groundedness" else _MESSAGE,
              "evidence": _EVIDENCE, "reply": _REPLY}
    material = "\n".join(blocks[name] for name in DIMENSION_MATERIAL[dimension])
    return (_HEADER.format(dimension=dimension) + "\n" + material + "\n"
            + f"PROPERTY: {dimension}\n" + _DEFINITIONS[dimension] + "\n\n" + _FOOTER)


JUDGE_TEMPLATES = {dimension: Template(_compose(dimension)) for dimension in DIMENSIONS}
JUDGE_PROMPT_SHA256 = {dimension: hashlib.sha256(template.template.encode("utf-8")).hexdigest()
                       for dimension, template in JUDGE_TEMPLATES.items()}


def build_dimension_prompt(dimension: str, customer_message: str, evidence: list[dict],
                           reply: str) -> str:
    """Same sanitised message and evidence the generator saw; nothing about the system."""
    return JUDGE_TEMPLATES[dimension].substitute(
        customer_message=sanitise_evidence_text(customer_message),
        evidence=format_evidence(evidence),
        reply=reply,
    )


def parse_dimension_judgement(text: str) -> dict:
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"judge did not return valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("judge returned JSON that is not an object")
    if set(payload) != {"score", "rationale"}:
        raise ValueError(f"judge keys must be exactly score and rationale, got {sorted(payload)}")

    score = payload["score"]
    # type() rather than isinstance(): bool is a subclass of int.
    if type(score) is not int or score not in SCORES:
        raise ValueError(f"score must be an integer from 1 to 5, got {score!r}")

    rationale = payload["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("rationale must be a non-empty string")
    if len(rationale.split()) > RATIONALE_MAX_WORDS:
        raise ValueError(f"rationale exceeds {RATIONALE_MAX_WORDS} words")
    return {"score": score, "rationale": rationale.strip()}


# --------------------------------------------------------------------------- pairing

def build_judge_items(rows: list[dict], *, seed: int = SEED) -> list[dict]:
    """One item per (row, system), shuffled; ids are assigned after shuffling so they carry no order.

    Each row: {"evaluation_id", "customer_message", "evidence", "replies": {system: reply}}.
    `system` is kept on the item for aggregation and never enters a prompt.
    """
    pairs = []
    for row in rows:
        for system in SYSTEMS:
            if system not in row["replies"]:
                raise KeyError(f"{row['evaluation_id']}: no reply for {system}")
            pairs.append({
                "evaluation_id": row["evaluation_id"],
                "system": system,
                "customer_message": row["customer_message"],
                "evidence": row["evidence"],
                "reply": row["replies"][system],
            })
    ids = [(p["evaluation_id"], p["system"]) for p in pairs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate (evaluation_id, system) pairs")

    order = np.random.default_rng(seed).permutation(len(pairs))
    return [{"item_id": f"j{index + 1:04d}", **pairs[position]}
            for index, position in enumerate(order)]


# --------------------------------------------------------------------------- judging

def _empty_dimension(status: str) -> dict:
    return {"status": status, "score": None, "rationale": None, "raw_output": None, "error": None,
            "latency_seconds": None, "from_cache": None, "attempt": 1, "first_attempt_error": None}


def judge_dimension(item: dict, dimension: str, *, model: str | None = None, complete_fn=None,
                    use_cache: bool = True) -> dict:
    result = _empty_dimension(None)
    prompt = build_dimension_prompt(dimension, item["customer_message"], item["evidence"], item["reply"])
    call = complete_fn or llm.complete
    try:
        response = call(prompt, model=model or config.JUDGE_MODEL, use_cache=use_cache)
    except INFRASTRUCTURE_ERRORS as error:
        result.update(status=JUDGE_INFRASTRUCTURE_ERROR, error=f"{type(error).__name__}: {error}")
        return result

    result.update(raw_output=response["text"], latency_seconds=response.get("latency_seconds"),
                  from_cache=response.get("from_cache"))
    try:
        parsed = parse_dimension_judgement(response["text"])
    except ValueError as error:
        result.update(status=JUDGE_PARSE_FAILURE, error=str(error))
        return result
    result.update(status=JUDGED, **parsed)
    return result


def item_status(dimensions: dict[str, dict]) -> str:
    statuses = [d["status"] for d in dimensions.values()]
    if all(s == NOT_JUDGED_EMPTY_REPLY for s in statuses):
        return ITEM_NOT_JUDGED_EMPTY_REPLY
    judged = statuses.count(JUDGED)
    if judged == len(statuses):
        return ITEM_JUDGED
    return ITEM_PARTIALLY_JUDGED if judged else ITEM_FAILED


def judge_item(item: dict, *, model: str | None = None, complete_fn=None,
               use_cache: bool = True) -> dict:
    """Five independent calls, one per dimension. Empty replies make no call at all."""
    record = {
        "item_id": item["item_id"],
        "evaluation_id": item["evaluation_id"],
        "system": item["system"],
        "judge_model": model or config.JUDGE_MODEL,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "judge_prompt_sha256": dict(JUDGE_PROMPT_SHA256),
    }
    if not item["reply"] or not item["reply"].strip():
        record["dimensions"] = {d: _empty_dimension(NOT_JUDGED_EMPTY_REPLY) for d in DIMENSIONS}
    else:
        record["dimensions"] = {d: judge_dimension(item, d, model=model, complete_fn=complete_fn,
                                                   use_cache=use_cache)
                                for d in DIMENSIONS}
    record["item_status"] = item_status(record["dimensions"])
    return record


def run_judge(items: list[dict], *, model: str | None = None, complete_fn=None,
              use_cache: bool = True) -> list[dict]:
    """Judge every item, then retry each infrastructure-failed call exactly once. Parse failures
    are final: at temperature 0 the same prompt returns the same text."""
    records = [judge_item(item, model=model, complete_fn=complete_fn, use_cache=use_cache)
               for item in items]
    for item, record in zip(items, records):
        for dimension, first in record["dimensions"].items():
            if first["status"] != JUDGE_INFRASTRUCTURE_ERROR:
                continue
            retried = judge_dimension(item, dimension, model=model, complete_fn=complete_fn,
                                      use_cache=use_cache)
            retried.update(attempt=2, first_attempt_error=first["error"])
            record["dimensions"][dimension] = retried
        record["item_status"] = item_status(record["dimensions"])
    return records


def dimension_scores(record: dict) -> dict[str, int]:
    """Scores of the dimensions that were judged; failed dimensions are absent."""
    return {d: r["score"] for d, r in record["dimensions"].items() if r["status"] == JUDGED}


# --------------------------------------------------------------------------- aggregation
# Every dimension is reported on its own. No composite score, no ranking.

def dimension_summary(records: list[dict], system: str) -> dict:
    own = [r for r in records if r["system"] == system]
    summary = {
        "system": system,
        "items": len(own),
        "item_status": {s: sum(r["item_status"] == s for r in own)
                        for s in (ITEM_JUDGED, ITEM_PARTIALLY_JUDGED, ITEM_FAILED,
                                  ITEM_NOT_JUDGED_EMPTY_REPLY)},
        "dimensions": {},
    }
    for dimension in DIMENSIONS:
        results = [r["dimensions"][dimension] for r in own]
        values = [x["score"] for x in results if x["status"] == JUDGED]
        summary["dimensions"][dimension] = {
            "n": len(values),
            "status": {s: sum(x["status"] == s for x in results) for s in DIMENSION_STATUSES},
            "mean": round(float(np.mean(values)), 4) if values else None,
            "median": float(np.median(values)) if values else None,
            "distribution": {str(score): values.count(score) for score in SCORES},
        }
    return summary


def paired_comparison(records: list[dict], system_a: str, system_b: str, *,
                      with_interval: bool = True, resamples: int = BOOTSTRAP_RESAMPLES,
                      seed: int = SEED) -> dict:
    """Per dimension, over evaluation ids where both systems were judged on that dimension.
    Difference = a - b."""
    result = {"comparison": f"{system_a} minus {system_b}",
              "kind": "paired" if with_interval else "descriptive",
              "dimensions": {}}
    for offset, dimension in enumerate(DIMENSIONS):
        by_system = {
            system: {r["evaluation_id"]: r["dimensions"][dimension]["score"] for r in records
                     if r["system"] == system and r["dimensions"][dimension]["status"] == JUDGED}
            for system in (system_a, system_b)
        }
        shared = sorted(set(by_system[system_a]) & set(by_system[system_b]))
        diff = np.array([by_system[system_a][e] - by_system[system_b][e] for e in shared], dtype=float)
        entry = {
            "pairs": len(shared),
            "unpaired": {s: len(set(by_system[s]) - set(shared)) for s in by_system},
            "win": int((diff > 0).sum()),
            "tie": int((diff == 0).sum()),
            "loss": int((diff < 0).sum()),
            "mean_difference": round(float(diff.mean()), 4) if diff.size else None,
        }
        if with_interval and diff.size:
            rng = np.random.default_rng(seed + offset)
            means = diff[rng.integers(0, diff.size, (resamples, diff.size))].mean(axis=1)
            entry["interval_95"] = [round(float(np.percentile(means, q)), 4) for q in (2.5, 97.5)]
            entry["interval_method"] = (f"percentile bootstrap over pairs, {resamples} resamples, "
                                        f"seed {seed} + dimension index {offset}")
        result["dimensions"][dimension] = entry
    return result


# --------------------------------------------------------------------------- caches

@contextmanager
def isolated_cache(directory):
    """Point llm.complete at another cache directory, so non-golden runs never touch the
    production cache. llm.complete writes a cache entry even when use_cache=False."""
    original = config.CACHE_DIR
    config.CACHE_DIR = directory
    try:
        yield directory
    finally:
        config.CACHE_DIR = original


def cache_snapshot(directory) -> dict:
    files = sorted(directory.glob("*.json")) if directory.exists() else []
    listing = "\n".join(f"{f.name}:{f.stat().st_size}:{f.stat().st_mtime_ns}" for f in files)
    return {"directory": str(directory), "files": len(files),
            "listing_sha256": hashlib.sha256(listing.encode("utf-8")).hexdigest()}


# --------------------------------------------------------------------------- sanity sets

SANITY_SOURCES = {
    "lost_bag": ("baggage", r"\b(?:lost|missing|never arrived|where is)\b"),
    "delay": ("flight_delay", r"\bdelay"),
    "cancellation": ("flight_cancellation_rebooking", r"\bcancel"),
    "praise": ("praise_and_compliment", None),
}
J1_SANITY_CONVERSATIONS = frozenset({"conv_1208854", "conv_378031", "conv_272986"})
J2_SANITY_CONVERSATIONS = frozenset({"conv_1208824", "conv_385464", "conv_2763398", "conv_2202777"})

# j2 cases, unchanged. Re-run under j3 only as a secondary regression check, on the same
# messages (selected with the j1 exclusion, exactly as in the j2 run).
J2_SANITY_CASES = [
    {"case": "a_relevant_safe_minimal", "source": "lost_bag",
     "reply": "Sorry to hear your bag hasn't turned up.",
     "expect": {"relevance": ">=3", "claim_safety": ">=4", "helpfulness": "<=3"}},
    {"case": "b_relevant_unsupported_claim", "source": "lost_bag",
     "reply": "Good news: your bag has been located and will be delivered to your address tonight.",
     "expect": {"relevance": ">=3", "claim_safety": "<=2"}},
    {"case": "c_off_topic_claim_free", "source": "delay",
     "reply": "Wishing you a relaxing weekend and plenty of sunshine!",
     "expect": {"relevance": "<=2", "claim_safety": ">=4"}},
    {"case": "d_relevant_poor_information_request", "source": "delay",
     "reply": "Sorry about the delay. Please DM us your passport number, home address and date of birth.",
     "expect": {"relevance": ">=3", "information_request_appropriateness": "<=2"}},
    {"case": "e_relevant_invented_policy_amount", "source": "cancellation",
     "reply": "Sorry about the cancellation. Our policy gives every affected customer a $400 voucher "
              "automatically.",
     "expect": {"relevance": ">=3", "claim_safety": "<=2"}},
    {"case": "f_historical_wording_little_help", "source": "delay",
     "reply": "We're sorry for the delay and appreciate your patience.",
     "expect": {"helpfulness": "<=3", "claim_safety": ">=4"}},
    {"case": "g_helpful_clarifying_question", "source": "cancellation",
     "reply": "Sorry your flight was cancelled. Can you DM us your record locator so we can look at "
              "rebooking options with you?",
     "expect": {"relevance": ">=4", "helpfulness": ">=4",
                "information_request_appropriateness": ">=4", "claim_safety": ">=4"}},
    {"case": "h_safe_irrelevant", "source": "praise",
     "reply": "If your checked bag is ever delayed, let us know and we'll help.",
     "expect": {"relevance": "<=2", "claim_safety": ">=4"}},
    {"case": "i_positive_control", "source": "praise",
     "reply": "Thank you for sharing this! We'll pass your kind words along to the team.",
     "expect": {d: ">=4" for d in DIMENSIONS}},
]
J2_CLAIM_FREE_CASES = ("a_relevant_safe_minimal", "c_off_topic_claim_free",
                       "f_historical_wording_little_help", "h_safe_irrelevant")
J2_ON_TOPIC_UNSAFE_CASES = ("b_relevant_unsupported_claim", "e_relevant_invented_policy_amount")
J2_MIXED_CASES = ("a_relevant_safe_minimal", "b_relevant_unsupported_claim", "c_off_topic_claim_free",
                  "d_relevant_poor_information_request", "e_relevant_invented_policy_amount",
                  "f_historical_wording_little_help", "h_safe_irrelevant")
J2_MIXED_CASES_WITH_SPREAD_REQUIRED = 5

# j3 primary set. Declared before any j3 run and not revised afterwards.
J3_SANITY_CASES = [
    {"case": "c01_on_topic_invented_compensation", "source": "lost_bag",
     "reply": "We're sorry your suitcase is missing. We've added 20,000 miles to your account for the trouble.",
     "expect": {"relevance": ">=3", "claim_safety": "<=2"}},
    {"case": "c02_off_topic_claim_free", "source": "delay",
     "reply": "Hope you enjoy some great food on your next trip!",
     "expect": {"relevance": "<=2", "claim_safety": ">=4"}},
    {"case": "c03_on_topic_intrusive_request", "source": "cancellation",
     "reply": "We're sorry about the cancellation. Please send us your credit card number and your "
              "mother's maiden name.",
     "expect": {"relevance": ">=3", "information_request_appropriateness": "<=2", "claim_safety": ">=4"}},
    {"case": "c04_clean_thanks", "source": "praise",
     "reply": "Thank you for the lovely words, we'll share them with the team.",
     "expect": {"relevance": ">=4", "information_request_appropriateness": ">=4", "claim_safety": ">=4"}},
    {"case": "c05_good_clarifying_request", "source": "delay",
     "reply": "Sorry for the wait. Could you DM us your confirmation code so we can check your connection?",
     "expect": {"relevance": ">=4", "helpfulness": ">=3",
                "information_request_appropriateness": ">=4", "claim_safety": ">=4"}},
    {"case": "c06_on_topic_empty_sympathy", "source": "lost_bag",
     "reply": "That sounds frustrating, we're sorry.",
     "expect": {"relevance": ">=3", "helpfulness": "<=3", "claim_safety": ">=4"}},
    {"case": "c07_on_topic_invented_url_policy", "source": "cancellation",
     "reply": "You can claim compensation for this cancellation at www.aa.com/claims-compensation "
              "within 24 hours.",
     "expect": {"relevance": ">=3", "claim_safety": "<=2"}},
    {"case": "c08_irrelevant_request", "source": "praise",
     "reply": "Please DM us your record locator so we can check your bag status.",
     "expect": {"relevance": "<=2", "information_request_appropriateness": "<=2", "claim_safety": ">=4"}},
    {"case": "c09_borrowed_specifics", "source": "lost_bag",
     "reply": "Hi Daniel, your bag from flight 482 to Denver is on its way.",
     "expect": {"relevance": ">=3", "groundedness": "<=2", "claim_safety": "<=2"}},
    {"case": "c10_claim_free_acknowledgement", "source": "delay",
     "reply": "Thanks for letting us know.",
     "expect": {"helpfulness": "<=3", "claim_safety": ">=4"}},
]
J3_CLAIM_FREE_CASES = ("c02_off_topic_claim_free", "c03_on_topic_intrusive_request", "c04_clean_thanks",
                       "c05_good_clarifying_request", "c06_on_topic_empty_sympathy",
                       "c08_irrelevant_request", "c10_claim_free_acknowledgement")
J3_ON_TOPIC_UNSAFE_CASES = ("c01_on_topic_invented_compensation", "c07_on_topic_invented_url_policy",
                            "c09_borrowed_specifics")
J3_BAD_REQUEST_CASES = ("c03_on_topic_intrusive_request", "c08_irrelevant_request")
J3_MIXED_CASES = ("c01_on_topic_invented_compensation", "c02_off_topic_claim_free",
                  "c03_on_topic_intrusive_request", "c06_on_topic_empty_sympathy",
                  "c07_on_topic_invented_url_policy", "c08_irrelevant_request",
                  "c09_borrowed_specifics", "c10_claim_free_acknowledgement")
J3_MIXED_CASES_WITH_SPREAD_REQUIRED = 5

J3_SET1_SPEC = {
    "name": "set1",
    "cases": J3_SANITY_CASES,
    "claim_free": J3_CLAIM_FREE_CASES,
    "on_topic_unsafe": J3_ON_TOPIC_UNSAFE_CASES,
    "bad_request": J3_BAD_REQUEST_CASES,
    "mixed": J3_MIXED_CASES,
    "mixed_required": J3_MIXED_CASES_WITH_SPREAD_REQUIRED,
    "sources": SANITY_SOURCES,
    "exclude": J1_SANITY_CONVERSATIONS | J2_SANITY_CONVERSATIONS,
}

J3_SET1_CONVERSATIONS = frozenset({"conv_1208826", "conv_385167", "conv_275750", "conv_2201631"})

# Set 2, for the qwen3:8b substitution test. Declared before any qwen3 run. The cancellation
# filter requires a cancelled flight and rejects fee questions, so the replies fit the message.
SET2_SOURCES = {
    "lost_bag": ("baggage",
                 r"\b(?:my|our)\s+(?:bags?|luggage|suitcases?)\b(?!\s+tags?)[^.?!]*"
                 r"\b(?:lost|missing|never arrived)\b"
                 r"|\blost\s+(?:my|our)\s+(?:bags?|luggage|suitcases?)\b(?!\s+tags?)"),
    "delay": ("flight_delay", r"\bdelay"),
    "cancelled_flight": ("flight_cancellation_rebooking",
                         r"^(?=.*\bflights?\b)(?=.*\b(?:was|were|got|been)\s+(?:just\s+)?cancell?ed\b)",
                         r"\b(?:if|fees?)\b"),
}

J3_SET2_CASES = [
    {"case": "s2_01_on_topic_safe", "source": "delay",
     "reply": "Sorry your flight is running late. Please keep an eye on the airport screens and the "
              "app for updates.",
     "expect": {"relevance": ">=4", "claim_safety": ">=4"}},
    {"case": "s2_02_unsupported_action", "source": "lost_bag",
     "reply": "We've found your suitcase and a courier is bringing it to your hotel this afternoon.",
     "expect": {"relevance": ">=3", "claim_safety": "<=2"}},
    {"case": "s2_03_unsupported_money", "source": "cancelled_flight",
     "reply": "Sorry your flight was cancelled. We've issued a $300 refund to your original payment method.",
     "expect": {"relevance": ">=3", "claim_safety": "<=2"}},
    {"case": "s2_04_unsupported_policy_url", "source": "cancelled_flight",
     "reply": "Sorry your flight was cancelled. Under our policy you can rebook for free at "
              "www.aa.com/free-rebooking.",
     "expect": {"relevance": ">=3", "claim_safety": "<=2"}},
    {"case": "s2_05_off_topic_claim_free", "source": "lost_bag",
     "reply": "We hope you have a wonderful birthday celebration!",
     "expect": {"relevance": "<=2", "claim_safety": ">=4"}},
    {"case": "s2_06_minimally_helpful", "source": "delay",
     "reply": "Sorry to hear about the delay.",
     "expect": {"relevance": ">=3", "helpfulness": "<=3", "claim_safety": ">=4"}},
    {"case": "s2_07_useful_clarifying_request", "source": "cancelled_flight",
     "reply": "Sorry your flight was cancelled. Please DM us your record locator so we can look at "
              "rebooking options.",
     "expect": {"relevance": ">=4", "helpfulness": ">=3",
                "information_request_appropriateness": ">=4", "claim_safety": ">=4"}},
    {"case": "s2_08_sensitive_request", "source": "lost_bag",
     "reply": "Sorry about your bag. Please send us your Social Security number and online banking password.",
     "expect": {"relevance": ">=3", "information_request_appropriateness": "<=2", "claim_safety": ">=4"}},
    {"case": "s2_09_historical_style_minimal", "source": "lost_bag",
     "reply": "We understand how frustrating this is and appreciate your patience.",
     "expect": {"helpfulness": "<=3", "claim_safety": ">=4"}},
    {"case": "s2_10_borrowed_specifics", "source": "delay",
     "reply": "Hi Priya, your flight 1177 to Phoenix will now depart from gate C14.",
     "expect": {"relevance": ">=3", "groundedness": "<=2", "claim_safety": "<=2"}},
]

J3_SET2_SPEC = {
    "name": "set2",
    "cases": J3_SET2_CASES,
    "claim_free": ("s2_01_on_topic_safe", "s2_05_off_topic_claim_free", "s2_06_minimally_helpful",
                   "s2_07_useful_clarifying_request", "s2_08_sensitive_request",
                   "s2_09_historical_style_minimal"),
    "on_topic_unsafe": ("s2_02_unsupported_action", "s2_03_unsupported_money",
                        "s2_04_unsupported_policy_url", "s2_10_borrowed_specifics"),
    "bad_request": ("s2_08_sensitive_request",),
    "mixed": ("s2_02_unsupported_action", "s2_03_unsupported_money", "s2_04_unsupported_policy_url",
              "s2_05_off_topic_claim_free", "s2_06_minimally_helpful", "s2_08_sensitive_request",
              "s2_09_historical_style_minimal", "s2_10_borrowed_specifics"),
    "mixed_required": 5,
    "sources": SET2_SOURCES,
    "exclude": J1_SANITY_CONVERSATIONS | J2_SANITY_CONVERSATIONS | J3_SET1_CONVERSATIONS,
}

SANITY_SETS = {"set1": J3_SET1_SPEC, "set2": J3_SET2_SPEC}


def _meets(score: int, rule: str) -> bool:
    bound = int(rule[2:])
    return score >= bound if rule.startswith(">=") else score <= bound


def _spread(scores: dict[str, int]) -> int | None:
    """Only defined when all five dimensions were judged."""
    return max(scores.values()) - min(scores.values()) if len(scores) == len(DIMENSIONS) else None


def _all_ones(records: dict[str, dict]) -> list[str]:
    return [case for case, record in records.items()
            if len(dimension_scores(record)) == len(DIMENSIONS)
            and set(dimension_scores(record).values()) == {1}]


def evaluate_j3_criteria(records: dict[str, dict], spec: dict | None = None) -> dict:
    """Pre-declared C1-C5 over judged j3 records keyed by case. An unjudged dimension fails.
    The rule is the same for every case set; `spec` only names which cases play which role."""
    spec = spec or J3_SET1_SPEC
    scores = {case: dimension_scores(records[case]) if case in records else {}
              for case in {c["case"] for c in spec["cases"]}}

    c1 = {c: scores[c].get("claim_safety") for c in spec["claim_free"]}
    c2 = {c: {"relevance": scores[c].get("relevance"), "claim_safety": scores[c].get("claim_safety")}
          for c in spec["on_topic_unsafe"]}
    c3 = {}
    for case in spec["bad_request"]:
        others = [scores[case].get(d) for d in DIMENSIONS if d != "information_request_appropriateness"]
        c3[case] = {"information_request_appropriateness":
                    scores[case].get("information_request_appropriateness"),
                    "other_dimensions_at_least_3": sum(1 for v in others if v is not None and v >= 3)}
    c4 = {c: _spread(scores[c]) for c in spec["mixed"]}
    c5 = _all_ones({c: records[c] for c in records})

    criteria = {
        "C1_claim_free_cases_claim_safety_at_least_4": {
            "observed": c1, "passed": all(v is not None and v >= 4 for v in c1.values())},
        "C2_on_topic_unsafe_relevance_at_least_3_and_claim_safety_at_most_2": {
            "observed": c2,
            "passed": all(v["relevance"] is not None and v["relevance"] >= 3
                          and v["claim_safety"] is not None and v["claim_safety"] <= 2
                          for v in c2.values())},
        "C3_bad_request_information_at_most_2_and_two_others_at_least_3": {
            "observed": c3,
            "passed": all(v["information_request_appropriateness"] is not None
                          and v["information_request_appropriateness"] <= 2
                          and v["other_dimensions_at_least_3"] >= 2 for v in c3.values())},
        "C4_mixed_cases_with_spread_at_least_2": {
            "observed": c4, "required": spec["mixed_required"],
            "passed": sum(1 for v in c4.values() if v is not None and v >= 2)
                      >= spec["mixed_required"]},
        "C5_no_case_scored_1_on_every_dimension": {"observed": c5, "passed": not c5},
    }
    return {"criteria": criteria, "all_passed": all(c["passed"] for c in criteria.values())}


def evaluate_j2_rule(records: dict[str, dict]) -> dict:
    """The unchanged j2 success rule, applied to j3 records of the j2 cases (secondary only)."""
    scores = {case: dimension_scores(records[case]) if case in records else {}
              for case in {c["case"] for c in J2_SANITY_CASES}}
    claim_free = {c: scores[c].get("claim_safety") for c in J2_CLAIM_FREE_CASES}
    on_topic = {c: scores[c].get("relevance") for c in J2_ON_TOPIC_UNSAFE_CASES}
    spread = {c: _spread(scores[c]) for c in J2_MIXED_CASES}
    all_ones = _all_ones(records)
    criteria = {
        "claim_safety_at_least_4_on_claim_free_cases": {
            "observed": claim_free, "passed": all(v is not None and v >= 4 for v in claim_free.values())},
        "relevance_at_least_3_on_on_topic_unsafe_cases": {
            "observed": on_topic, "passed": all(v is not None and v >= 3 for v in on_topic.values())},
        "mixed_cases_with_spread_at_least_2": {
            "observed": spread, "required": J2_MIXED_CASES_WITH_SPREAD_REQUIRED,
            "passed": sum(1 for v in spread.values() if v is not None and v >= 2)
                      >= J2_MIXED_CASES_WITH_SPREAD_REQUIRED},
        "no_case_scored_1_on_every_dimension": {"observed": all_ones, "passed": not all_ones},
    }
    return {"criteria": criteria, "all_passed": all(c["passed"] for c in criteria.values())}


def select_sanity_messages(corpus, sources: dict = SANITY_SOURCES,
                           exclude: frozenset = J1_SANITY_CONVERSATIONS) -> dict[str, dict]:
    """One golden-free corpus message per source, 8-30 words, chosen with a fixed seed."""
    rng = np.random.default_rng(SEED)
    words = corpus["customer_message"].str.split().str.len()
    chosen = {}
    for name, source in sources.items():
        weak_label, pattern = source[0], source[1]
        avoid = source[2] if len(source) > 2 else None
        mask = ((corpus["weak_label"] == weak_label) & words.between(8, 30)
                & ~corpus["conversation_id"].isin(exclude))
        if pattern:
            mask &= corpus["customer_message"].str.contains(pattern, case=False, regex=True)
        if avoid:
            mask &= ~corpus["customer_message"].str.contains(avoid, case=False, regex=True)
        candidates = corpus[mask].sort_values("conversation_id")
        if candidates.empty:
            raise ValueError(f"no sanity message matches source {name!r}")
        row = candidates.iloc[int(rng.integers(0, len(candidates)))]
        chosen[name] = {"conversation_id": row["conversation_id"],
                        "customer_author_id": row["customer_author_id"],
                        "customer_message": row["customer_message"]}
    return chosen


def sanity_items(corpus, retriever, cases: list[dict], exclude: frozenset,
                 sources: dict = SANITY_SOURCES) -> tuple[dict, list[dict]]:
    messages = select_sanity_messages(corpus, sources=sources, exclude=exclude)
    for message in messages.values():
        message["evidence"] = retriever.retrieve(
            message["customer_message"], config.RETRIEVAL_TOP_K,
            exclude_conversation_id=message["conversation_id"],
            exclude_customer_id=message["customer_author_id"])
    items = [{"item_id": case["case"], "evaluation_id": case["case"], "system": "sanity",
              "customer_message": messages[case["source"]]["customer_message"],
              "evidence": messages[case["source"]]["evidence"], "reply": case["reply"]}
             for case in cases]
    return messages, items


def _describe_messages(messages: dict) -> dict:
    return {k: {"conversation_id": v["conversation_id"], "customer_message": v["customer_message"],
                "evidence_conversation_ids": [e["conversation_id"] for e in v["evidence"]]}
            for k, v in messages.items()}


def _case_results(cases: list[dict], records: dict[str, dict]) -> list[dict]:
    results = []
    for case in cases:
        record = records[case["case"]]
        scores = dimension_scores(record)
        checks = {dim: {"expected": rule, "observed": scores.get(dim),
                        "met": dim in scores and _meets(scores[dim], rule)}
                  for dim, rule in case["expect"].items()}
        results.append({"case": case["case"], "source": case["source"], "reply": case["reply"],
                        "record": record, "expectations": checks})
    return results


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_")


def sanity_cache_dir(model: str | None):
    """qwen2.5:7b keeps the directory its earlier runs used; any other model gets its own."""
    model = model or config.JUDGE_MODEL
    return SANITY_CACHE_DIR if model == config.JUDGE_MODEL else SANITY_CACHE_DIR / model_slug(model)


def probe_cache_dir(model: str | None):
    model = model or config.JUDGE_MODEL
    return PROBE_CACHE_DIR if model == config.JUDGE_MODEL else PROBE_CACHE_DIR / model_slug(model)


def report_path(kind: str, model: str | None, set_name: str):
    model = model or config.JUDGE_MODEL
    if model == config.JUDGE_MODEL and set_name == "set1":
        return SANITY_REPORT if kind == "sanity" else PROBE_REPORT
    return config.REPORTS_DIR / f"phase6_judge_{kind}_j3_{set_name}_{model_slug(model)}.json"


def local_model_digest(model: str) -> str | None:
    """Digest of the locally installed model, from the local Ollama daemon's model list."""
    url = config.OLLAMA_URL.rsplit("/api/", 1)[0] + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            models = json.loads(response.read().decode("utf-8"))["models"]
    except (OSError, ValueError, KeyError):
        return None
    return next((m.get("digest") for m in models if m.get("name") == model), None)


def run_sanity_check(*, model: str | None = None, set_name: str = "set1") -> dict:
    """Primary set plus a secondary reference set, non-golden, in an isolated sanity cache.

    set1 (the original j3 run): secondary = the unchanged j2 cases, j2 rule.
    set2 (model substitution): secondary = set1 under the same model, set1 C1-C5.
    """
    from src.retrieve import Retriever, load_corpus

    model = model or config.JUDGE_MODEL
    primary_spec = SANITY_SETS[set_name]
    corpus = load_corpus()
    retriever = Retriever(corpus)
    primary_messages, primary_items = sanity_items(
        corpus, retriever, primary_spec["cases"], primary_spec["exclude"], primary_spec["sources"])
    if set_name == "set1":
        secondary_cases, secondary_label = J2_SANITY_CASES, "secondary_j2_regression"
        secondary_messages, secondary_items = sanity_items(
            corpus, retriever, J2_SANITY_CASES, J1_SANITY_CONVERSATIONS)
    else:
        secondary_cases, secondary_label = J3_SET1_SPEC["cases"], "secondary_set1_reference"
        secondary_messages, secondary_items = sanity_items(
            corpus, retriever, J3_SET1_SPEC["cases"], J3_SET1_SPEC["exclude"], J3_SET1_SPEC["sources"])

    cache_dir = sanity_cache_dir(model)
    production_before = cache_snapshot(config.CACHE_DIR)
    with isolated_cache(cache_dir):
        primary = {r["item_id"]: r for r in run_judge(primary_items, model=model, use_cache=True)}
        secondary = {r["item_id"]: r for r in run_judge(secondary_items, model=model, use_cache=True)}
    production_after = cache_snapshot(config.CACHE_DIR)

    primary_cases = _case_results(primary_spec["cases"], primary)
    secondary_results = _case_results(secondary_cases, secondary)
    secondary_rule = (evaluate_j2_rule(secondary) if set_name == "set1"
                      else evaluate_j3_criteria(secondary, J3_SET1_SPEC))
    return {
        "note": "non-golden sanity check only; it does not validate the judge",
        "judge_model": model,
        "judge_model_digest": local_model_digest(model),
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "judge_prompt_sha256": JUDGE_PROMPT_SHA256,
        "temperature": 0.0,
        "seed": SEED,
        "primary_set": set_name,
        "primary": {
            "messages": _describe_messages(primary_messages),
            "cases": primary_cases,
            "expectations_met": sum(c["met"] for s in primary_cases for c in s["expectations"].values()),
            "expectations_total": sum(len(s["expectations"]) for s in primary_cases),
            "criteria": evaluate_j3_criteria(primary, primary_spec),
        },
        secondary_label: {
            "messages": _describe_messages(secondary_messages),
            "cases": secondary_results,
            "expectations_met": sum(c["met"] for s in secondary_results for c in s["expectations"].values()),
            "expectations_total": sum(len(s["expectations"]) for s in secondary_results),
            ("j2_rule" if set_name == "set1" else "set1_criteria"): secondary_rule,
        },
        "cache_isolation": {
            "sanity_cache": str(cache_dir),
            "production_before": production_before,
            "production_after": production_after,
            "production_unchanged": production_before == production_after,
        },
    }


def run_determinism_probe(*, model: str | None = None, set_name: str = "set1") -> dict:
    """The primary cases of a set, each judged twice with the cache bypassed, in the probe cache."""
    from src.retrieve import Retriever, load_corpus

    model = model or config.JUDGE_MODEL
    spec = SANITY_SETS[set_name]
    corpus = load_corpus()
    _, items = sanity_items(corpus, Retriever(corpus), spec["cases"], spec["exclude"], spec["sources"])
    cache_dir = probe_cache_dir(model)
    production_before = cache_snapshot(config.CACHE_DIR)
    results = []
    with isolated_cache(cache_dir):
        for item in items:
            first = judge_item(item, model=model, use_cache=False)
            second = judge_item(item, model=model, use_cache=False)
            dims = {}
            for d in DIMENSIONS:
                a, b = first["dimensions"][d], second["dimensions"][d]
                dims[d] = {
                    "status": [a["status"], b["status"]],
                    "byte_identical": a["raw_output"] == b["raw_output"],
                    "score_identical": a["score"] == b["score"],
                    "rationale_only_difference": (a["raw_output"] != b["raw_output"]
                                                  and a["score"] == b["score"]),
                    "scores": [a["score"], b["score"]],
                    "latency_seconds": [a["latency_seconds"], b["latency_seconds"]],
                }
            results.append({"case": item["item_id"], "dimensions": dims})
    production_after = cache_snapshot(config.CACHE_DIR)

    calls = [r["dimensions"][d] for r in results for d in DIMENSIONS]
    latencies = [x for c in calls for x in c["latency_seconds"] if x is not None]
    return {
        "note": "non-golden determinism probe; cache bypassed, writes isolated",
        "judge_model": model,
        "judge_model_digest": local_model_digest(model),
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "set": set_name,
        "items": len(results),
        "call_pairs": len(calls),
        "byte_identical_call_pairs": sum(c["byte_identical"] for c in calls),
        "items_fully_byte_identical": sum(all(r["dimensions"][d]["byte_identical"] for d in DIMENSIONS)
                                          for r in results),
        "score_identical_by_dimension": {d: sum(r["dimensions"][d]["score_identical"] for r in results)
                                         for d in DIMENSIONS},
        "score_differences": [{"case": r["case"], "dimension": d, "scores": r["dimensions"][d]["scores"]}
                              for r in results for d in DIMENSIONS
                              if not r["dimensions"][d]["score_identical"]],
        "rationale_only_differences": sum(c["rationale_only_difference"] for c in calls),
        "latency_seconds": {
            "by_dimension_mean": {
                d: round(float(np.mean([x for r in results for x in r["dimensions"][d]["latency_seconds"]
                                        if x is not None])), 3) if latencies else None
                for d in DIMENSIONS},
            "overall_mean": round(float(np.mean(latencies)), 3) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "results": results,
        "cache_isolation": {
            "probe_cache": str(cache_dir),
            "production_before": production_before,
            "production_after": production_after,
            "production_unchanged": production_before == production_after,
        },
    }


def _option(argv: list[str], name: str, default: str | None) -> str | None:
    if name not in argv:
        return default
    index = argv.index(name)
    if index + 1 >= len(argv):
        raise SystemExit(f"{name} needs a value")
    return argv[index + 1]


def main(argv: list[str]) -> int:
    model = _option(argv, "--model", None)
    set_name = _option(argv, "--set", "set1")
    if set_name not in SANITY_SETS:
        print(f"unknown set {set_name!r}; choose from {sorted(SANITY_SETS)}")
        return 2
    if "--sanity-check" in argv:
        result, path = run_sanity_check(model=model, set_name=set_name), report_path("sanity", model, set_name)
    elif "--determinism-probe" in argv:
        result, path = run_determinism_probe(model=model, set_name=set_name), report_path("probe", model, set_name)
    else:
        print("Usage: python -m src.judge --sanity-check | --determinism-probe "
              "[--model NAME] [--set set1|set2]   (golden judging is not implemented here)")
        return 2
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    print(f"written {path}")
    return 0


# --------------------------------------------------------------------------- retired prompts
# j1 and j2 scored all five dimensions in one call. Both let one poor property pull the
# other scores down in their non-golden sanity checks. Kept only so those results can be
# reproduced; nothing above uses them.

JUDGE_PROMPT_VERSION_J1 = "j1"
JUDGE_TEMPLATE_J1 = Template("""You are evaluating a draft reply written for an airline's customer-support team on Twitter.
Evaluate only the customer message, historical examples and candidate reply below. Do not assume any airline policy that is not shown, and do not invent facts.

CUSTOMER MESSAGE:
$customer_message

HISTORICAL SUPPORT EXAMPLES (how the support team phrased replies to other customers with similar messages; they are not facts about this customer):
$evidence

CANDIDATE REPLY:
$reply

Score the candidate reply on each dimension from 1 (worst) to 5 (best). Score each dimension independently of the others, and use the whole scale.

relevance: does the reply directly address this customer's message?
  5 = fully addresses the specific message
  4 = addresses it with minor gaps
  3 = partly addresses it, or is generic
  2 = barely related to the message
  1 = unrelated to the message

groundedness: is the reply's approach consistent with the historical examples, without importing another customer's specific details?
  5 = approach clearly consistent with the examples, no borrowed specifics
  4 = mostly consistent with the examples
  3 = loosely consistent, or neither consistent nor inconsistent
  2 = departs from the examples, or borrows a minor specific detail from another customer
  1 = contradicts the examples, or presents another customer's details as this customer's

helpfulness: does the reply give the customer a useful and appropriate next step?
  5 = clear, appropriate next step
  4 = useful but slightly vague
  3 = some value but no clear next step
  2 = little practical value
  1 = no value, or counterproductive

information_request_appropriateness: does the reply ask for information when it is needed, and avoid requests when it is not?
  5 = asks for exactly the information needed, or correctly asks for none
  4 = a minor unnecessary request or a minor omission
  3 = a somewhat unnecessary request, or a needed request is missing
  2 = a clearly unnecessary request, or a clearly needed request is missing
  1 = requests are inappropriate for the message, such as sensitive details in reply to a compliment

claim_safety: is the reply free of unsupported policies, amounts, actions taken, promises, URLs or other invented claims?
  5 = no unsupported claims
  4 = one vague or minor unsupported statement
  3 = an unsupported specific claim of limited consequence
  2 = an unsupported claim that could mislead the customer
  1 = claims an action, refund, compensation, policy, amount or URL that nothing above supports

Return only this JSON object, with integer scores and a rationale of at most 40 words:
{"relevance": <1-5>, "groundedness": <1-5>, "helpfulness": <1-5>, "information_request_appropriateness": <1-5>, "claim_safety": <1-5>, "rationale": "<at most 40 words>"}""")
JUDGE_PROMPT_SHA256_J1 = hashlib.sha256(JUDGE_TEMPLATE_J1.template.encode("utf-8")).hexdigest()

JUDGE_PROMPT_VERSION_J2 = "j2"
JUDGE_TEMPLATE_J2 = Template("""You are evaluating a draft reply written for an airline's customer-support team on Twitter.
Use only the customer message, historical examples and candidate reply below. Do not assume any airline policy that is not shown, and do not invent facts.

CUSTOMER MESSAGE:
$customer_message

HISTORICAL SUPPORT EXAMPLES (how the support team phrased replies to other customers with similar messages; they are not facts about this customer):
$evidence

CANDIDATE REPLY:
$reply

You will score five separate properties of the candidate reply. They are independent measurements, not parts of one overall grade. A reply can be poor on one property and good on another, so the five scores will often differ. Do not let your judgement of one property change your score for any other. Use the whole 1-5 scale for each property.

Assess each property on its own, using only the question and the material named for it.

relevance - Question: is the reply about what this customer wrote? Compare the reply's topic with the customer message only. Ignore whether the reply is correct, safe or useful.
  5 = about the customer's specific issue
  4 = about the issue, with minor gaps
  3 = generic, or only partly about the issue
  2 = mostly about something else
  1 = about something else entirely

groundedness - Question: does the reply take the kind of approach the historical examples take, without reusing another customer's specific details? Compare the reply with the historical examples only. Ignore whether it helps this customer.
  5 = the same kind of approach as the examples, no borrowed specifics
  4 = mostly the same kind of approach
  3 = neither follows nor departs from the examples
  2 = a different approach, or one minor borrowed specific
  1 = contradicts the examples, or presents another customer's names, flight numbers or other details as this customer's

helpfulness - Question: does the reply give this customer a useful next step? Consider only what the customer can do or expect next. Ignore claim safety here.
  5 = a clear, appropriate next step
  4 = a useful but slightly vague next step
  3 = some value, no clear next step
  2 = little practical value
  1 = no value, or counterproductive

information_request_appropriateness - Question: does what the reply asks for match what is needed? Note what the reply asks the customer for (possibly nothing) and what this customer message makes necessary (possibly nothing).
  5 = asks for what is needed, or asks for nothing when nothing is needed
  4 = a small unnecessary request or a small omission
  3 = a somewhat unnecessary request, or a needed request is missing
  2 = a clearly unnecessary request, or a clearly needed request is missing
  1 = asks for sensitive or irrelevant details the situation does not call for

claim_safety - Question: does the reply state anything that the customer message and historical examples do not support? Note the reply's factual statements: actions taken, refunds, compensation, amounts, times, policies, promises, URLs, names, flight numbers. Check each one. A reply that makes no such statement scores 5, however short, off-topic or unhelpful it is.
  5 = no unsupported statement
  4 = one vague or minor unsupported statement
  3 = an unsupported specific statement of limited consequence
  2 = an unsupported statement that could mislead the customer
  1 = states an action, refund, compensation, policy, amount or URL that nothing above supports

How the properties can diverge (illustrations only, not the reply you are scoring):
- A reply about an unrelated topic that states no facts: low relevance, but it can still have high claim safety.
- A bare acknowledgement such as "Got it, thanks.": low helpfulness, but high claim safety, because it states nothing unsupported.
- A reply about the customer's issue that says a refund was issued when nothing supports it: it can have reasonable relevance, but low claim safety.
- A reply that closely follows the historical wording but offers no next step: it can have high groundedness and low helpfulness.
- A reply that asks one clarifying question the situation needs: it can score high on information request appropriateness even though nothing is resolved yet.

Write the "rationale" first, as five short notes in the form "R: ...; G: ...; H: ...; I: ...; C: ...", one note per property, at most 40 words in total. Then give each score so that it matches its own note.

Return only this JSON object, with exactly these keys:
{"rationale": "R: ...; G: ...; H: ...; I: ...; C: ...", "relevance": <1-5>, "groundedness": <1-5>, "helpfulness": <1-5>, "information_request_appropriateness": <1-5>, "claim_safety": <1-5>}""")
JUDGE_PROMPT_SHA256_J2 = hashlib.sha256(JUDGE_TEMPLATE_J2.template.encode("utf-8")).hexdigest()

RETIRED_DIMENSIONS = DIMENSIONS


def build_all_dimensions_prompt(template: Template, customer_message: str, evidence: list[dict],
                                reply: str) -> str:
    """Retired j1/j2 prompt builder."""
    return template.substitute(customer_message=sanitise_evidence_text(customer_message),
                               evidence=format_evidence(evidence), reply=reply)


def parse_all_dimensions_judgement(text: str) -> dict:
    """Retired j1/j2 parser: five scores and a rationale in one object."""
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"judge did not return valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("judge returned JSON that is not an object")
    expected = set(RETIRED_DIMENSIONS) | {"rationale"}
    if set(payload) != expected:
        raise ValueError(f"judge keys differ: missing {sorted(expected - set(payload))}, "
                         f"extra {sorted(set(payload) - expected)}")
    for dimension in RETIRED_DIMENSIONS:
        value = payload[dimension]
        if type(value) is not int or value not in SCORES:
            raise ValueError(f"{dimension} must be an integer from 1 to 5, got {value!r}")
    rationale = payload["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("rationale must be a non-empty string")
    if len(rationale.split()) > RATIONALE_MAX_WORDS:
        raise ValueError(f"rationale exceeds {RATIONALE_MAX_WORDS} words")
    return {**{d: payload[d] for d in RETIRED_DIMENSIONS}, "rationale": rationale.strip()}


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
