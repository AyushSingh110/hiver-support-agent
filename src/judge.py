"""Phase 6E LLM-as-judge.

Model-based evaluation, not human evaluation. No human reply ratings exist, so the judge
is unvalidated; the non-golden sanity check below only shows it behaves plausibly on
obvious cases. Baseline B (a fixed sentence) and Baseline A (a verbatim echo of evidence)
are recognisable from their content, so blinding is partial: the judge is never told
which system wrote a reply, but it may infer it.
"""
from __future__ import annotations

import hashlib
import json
import sys
from string import Template

import numpy as np

from src import config, llm
from src.generate_reply import format_evidence, sanitise_evidence_text

JUDGE_PROMPT_VERSION = "j1"
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

# Errors raised while reaching the model, as opposed to errors in what it returned.
INFRASTRUCTURE_ERRORS = (llm.LLMError, OSError, MemoryError, ValueError)

SANITY_REPORT = config.REPORTS_DIR / "phase6_judge_sanity.json"

JUDGE_TEMPLATE = Template("""You are evaluating a draft reply written for an airline's customer-support team on Twitter.
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

JUDGE_PROMPT_SHA256 = hashlib.sha256(JUDGE_TEMPLATE.template.encode("utf-8")).hexdigest()


def build_judge_prompt(customer_message: str, evidence: list[dict], reply: str) -> str:
    """Same sanitised message and evidence formatting the generator saw; nothing about the system."""
    return JUDGE_TEMPLATE.substitute(
        customer_message=sanitise_evidence_text(customer_message),
        evidence=format_evidence(evidence),
        reply=reply,
    )


def parse_judgement(text: str) -> dict:
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"judge did not return valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("judge returned JSON that is not an object")

    expected = set(DIMENSIONS) | {"rationale"}
    if set(payload) != expected:
        raise ValueError(f"judge keys differ: missing {sorted(expected - set(payload))}, "
                         f"extra {sorted(set(payload) - expected)}")

    for dimension in DIMENSIONS:
        value = payload[dimension]
        # type() rather than isinstance(): bool is a subclass of int.
        if type(value) is not int or value not in SCORES:
            raise ValueError(f"{dimension} must be an integer from 1 to 5, got {value!r}")

    rationale = payload["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("rationale must be a non-empty string")
    if len(rationale.split()) > RATIONALE_MAX_WORDS:
        raise ValueError(f"rationale exceeds {RATIONALE_MAX_WORDS} words")

    return {**{d: payload[d] for d in DIMENSIONS}, "rationale": rationale.strip()}


def build_judge_items(rows: list[dict], *, seed: int = SEED) -> list[dict]:
    """One item per (row, system), shuffled; ids are assigned after shuffling so they carry no order.

    Each row: {"evaluation_id", "customer_message", "evidence", "replies": {system: reply}}.
    `system` is kept on the item for aggregation and never enters the prompt.
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


def judge_item(item: dict, *, complete_fn=None, use_cache: bool = True) -> dict:
    record = {
        "item_id": item["item_id"],
        "evaluation_id": item["evaluation_id"],
        "system": item["system"],
        "judge_model": config.JUDGE_MODEL,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "judge_prompt_sha256": JUDGE_PROMPT_SHA256,
        "status": None,
        "scores": None,
        "rationale": None,
        "raw_output": None,
        "error": None,
        "latency_seconds": None,
        "from_cache": None,
    }
    if not item["reply"] or not item["reply"].strip():
        record["status"] = NOT_JUDGED_EMPTY_REPLY
        return record

    prompt = build_judge_prompt(item["customer_message"], item["evidence"], item["reply"])
    call = complete_fn or llm.complete
    try:
        response = call(prompt, model=config.JUDGE_MODEL, use_cache=use_cache)
    except INFRASTRUCTURE_ERRORS as error:
        record["status"] = JUDGE_INFRASTRUCTURE_ERROR
        record["error"] = f"{type(error).__name__}: {error}"
        return record

    record.update(raw_output=response["text"], latency_seconds=response.get("latency_seconds"),
                  from_cache=response.get("from_cache"))
    try:
        parsed = parse_judgement(response["text"])
    except ValueError as error:
        record["status"] = JUDGE_PARSE_FAILURE
        record["error"] = str(error)
        return record

    record["status"] = JUDGED
    record["rationale"] = parsed.pop("rationale")
    record["scores"] = parsed
    return record


def run_judge(items: list[dict], *, complete_fn=None, use_cache: bool = True) -> list[dict]:
    """Judge every item, then retry infrastructure errors exactly once. Parse failures are final:
    at temperature 0 the same prompt returns the same text."""
    first = [judge_item(item, complete_fn=complete_fn, use_cache=use_cache) for item in items]
    records = []
    for item, record in zip(items, first):
        if record["status"] == JUDGE_INFRASTRUCTURE_ERROR:
            retried = judge_item(item, complete_fn=complete_fn, use_cache=use_cache)
            retried["attempt"] = 2
            retried["first_attempt_error"] = record["error"]
            records.append(retried)
        else:
            record["attempt"] = 1
            record["first_attempt_error"] = None
            records.append(record)
    return records


# Aggregation. No composite score, no ranking.

def status_counts(records: list[dict], system: str) -> dict:
    statuses = [r["status"] for r in records if r["system"] == system]
    return {"items": len(statuses),
            **{s: statuses.count(s) for s in (JUDGED, NOT_JUDGED_EMPTY_REPLY,
                                              JUDGE_PARSE_FAILURE, JUDGE_INFRASTRUCTURE_ERROR)}}


def dimension_summary(records: list[dict], system: str) -> dict:
    judged = [r["scores"] for r in records if r["system"] == system and r["status"] == JUDGED]
    summary = {"system": system, "status": status_counts(records, system), "dimensions": {}}
    for dimension in DIMENSIONS:
        values = [s[dimension] for s in judged]
        summary["dimensions"][dimension] = {
            "n": len(values),
            "mean": round(float(np.mean(values)), 4) if values else None,
            "median": float(np.median(values)) if values else None,
            "distribution": {str(score): values.count(score) for score in SCORES},
        }
    return summary


def paired_comparison(records: list[dict], system_a: str, system_b: str, *,
                      with_interval: bool = True, resamples: int = BOOTSTRAP_RESAMPLES,
                      seed: int = SEED) -> dict:
    """Per dimension, over evaluation ids where both systems were judged. Difference = a - b."""
    by_system = {
        system: {r["evaluation_id"]: r["scores"] for r in records
                 if r["system"] == system and r["status"] == JUDGED}
        for system in (system_a, system_b)
    }
    shared = sorted(set(by_system[system_a]) & set(by_system[system_b]))
    result = {
        "comparison": f"{system_a} minus {system_b}",
        "kind": "paired" if with_interval else "descriptive",
        "pairs": len(shared),
        "unpaired": {s: len(set(by_system[s]) - set(shared)) for s in by_system},
        "dimensions": {},
    }
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(shared), (resamples, len(shared))) if with_interval and shared else None
    for dimension in DIMENSIONS:
        diff = np.array([by_system[system_a][e][dimension] - by_system[system_b][e][dimension]
                         for e in shared], dtype=float)
        entry = {
            "win": int((diff > 0).sum()),
            "tie": int((diff == 0).sum()),
            "loss": int((diff < 0).sum()),
            "mean_difference": round(float(diff.mean()), 4) if diff.size else None,
        }
        if draws is not None:
            means = diff[draws].mean(axis=1)
            entry["interval_95"] = [round(float(np.percentile(means, q)), 4) for q in (2.5, 97.5)]
            entry["interval_method"] = f"percentile bootstrap over pairs, {resamples} resamples, seed {seed}"
        result["dimensions"][dimension] = entry
    return result


# Non-golden sanity check and determinism probe.

SANITY_SOURCES = ("baggage", "flight_delay", "praise_and_compliment")

# Declared before the first run. Each case targets one or two dimensions.
SANITY_CASES = [
    {"case": "S1_reasonable", "source": "baggage",
     "reply": "Sorry to hear about your bag. Please DM us your bag tag number and record locator "
              "so we can look into this for you.",
     "expect": {"relevance": ">=4", "claim_safety": ">=4"}},
    {"case": "S2_off_topic", "source": "baggage",
     "reply": "Congratulations on your upgrade! Enjoy the extra legroom and the meal service on "
              "your flight today.",
     "expect": {"relevance": "<=2"}},
    {"case": "S3_invented_refund_and_action", "source": "baggage",
     "reply": "We have refunded $250 to your card and your bag will be delivered to your home "
              "tomorrow by 5pm.",
     "expect": {"claim_safety": "<=2"}},
    {"case": "S4_unnecessary_information_request", "source": "praise_and_compliment",
     "reply": "Please DM us your full name, date of birth, record locator and ticket number so we "
              "can verify your account.",
     "expect": {"information_request_appropriateness": "<=2"}},
    {"case": "S5_borrowed_specifics", "source": "flight_delay",
     "reply": "Sorry Maria, your flight 2381 from Phoenix was delayed by the thunderstorm, and your "
              "hotel voucher is ready at gate B7.",
     "expect": {"groundedness": "<=2", "claim_safety": "<=2"}},
    {"case": "S6_unhelpful", "source": "flight_delay",
     "reply": "Noted.",
     "expect": {"helpfulness": "<=2"}},
]

PROBE_EXTRA_CASES = [
    {"case": "P7_praise_thanks", "source": "praise_and_compliment",
     "reply": "Thank you for the kind words! We'll be sure to pass them along to the crew."},
    {"case": "P8_top1_echo_baggage", "source": "baggage", "reply": None},
    {"case": "P9_top1_echo_delay", "source": "flight_delay", "reply": None},
    {"case": "P10_top1_echo_praise", "source": "praise_and_compliment", "reply": None},
]


def _meets(score: int, rule: str) -> bool:
    bound = int(rule[2:])
    return score >= bound if rule.startswith(">=") else score <= bound


def select_sanity_messages(corpus) -> dict[str, dict]:
    """One golden-free corpus message per source intent, chosen by seed, 8-30 words long."""
    rng = np.random.default_rng(SEED)
    chosen = {}
    words = corpus["customer_message"].str.split().str.len()
    for source in SANITY_SOURCES:
        candidates = corpus[(corpus["weak_label"] == source) & words.between(8, 30)]
        candidates = candidates.sort_values("conversation_id")
        row = candidates.iloc[int(rng.integers(0, len(candidates)))]
        chosen[source] = {"conversation_id": row["conversation_id"],
                          "customer_author_id": row["customer_author_id"],
                          "customer_message": row["customer_message"]}
    return chosen


def _sanity_items(corpus, retriever) -> tuple[dict, list[dict]]:
    messages = select_sanity_messages(corpus)
    for message in messages.values():
        message["evidence"] = retriever.retrieve(
            message["customer_message"], config.RETRIEVAL_TOP_K,
            exclude_conversation_id=message["conversation_id"],
            exclude_customer_id=message["customer_author_id"])
    items = []
    for case in SANITY_CASES + PROBE_EXTRA_CASES:
        message = messages[case["source"]]
        reply = case["reply"]
        if reply is None:
            reply = sanitise_evidence_text(message["evidence"][0]["historical_brand_reply"])
        items.append({"item_id": case["case"], "evaluation_id": case["case"], "system": "sanity",
                      "customer_message": message["customer_message"],
                      "evidence": message["evidence"], "reply": reply})
    return messages, items


def run_sanity_check_and_probe() -> dict:
    """Non-golden only: corpus messages (golden-free, isolation asserted on load)."""
    from src.retrieve import Retriever, load_corpus

    corpus = load_corpus()
    messages, items = _sanity_items(corpus, Retriever(corpus))
    by_case = {item["item_id"]: item for item in items}

    sanity = []
    for case in SANITY_CASES:
        record = judge_item(by_case[case["case"]], use_cache=True)
        checks = {}
        if record["status"] == JUDGED:
            checks = {dim: {"expected": rule, "observed": record["scores"][dim],
                            "met": _meets(record["scores"][dim], rule)}
                      for dim, rule in case["expect"].items()}
        sanity.append({"case": case["case"], "reply": case["reply"], "record": record,
                       "expectations": checks})

    probe = []
    for item in items:
        first = judge_item(item, use_cache=False)
        second = judge_item(item, use_cache=False)
        probe.append({
            "case": item["item_id"],
            "status": [first["status"], second["status"]],
            "byte_identical": first["raw_output"] == second["raw_output"],
            "scores": [first["scores"], second["scores"]],
            "latency_seconds": [first["latency_seconds"], second["latency_seconds"]],
        })

    return {
        "note": "non-golden sanity check only; it does not validate the judge",
        "judge_model": config.JUDGE_MODEL,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "judge_prompt_sha256": JUDGE_PROMPT_SHA256,
        "temperature": 0.0,
        "seed": SEED,
        "messages": {k: {"conversation_id": v["conversation_id"],
                         "customer_message": v["customer_message"],
                         "evidence_conversation_ids": [e["conversation_id"] for e in v["evidence"]]}
                     for k, v in messages.items()},
        "sanity_cases": sanity,
        "expectations_met": sum(c["met"] for s in sanity for c in s["expectations"].values()),
        "expectations_total": sum(len(s["expectations"]) for s in sanity),
        "determinism_probe": {
            "items": len(probe),
            "calls_per_item": 2,
            "cache": "bypassed",
            "byte_identical_items": sum(p["byte_identical"] for p in probe),
            "results": probe,
        },
    }


def main(argv: list[str]) -> int:
    if "--sanity-check" not in argv:
        print("Usage: python -m src.judge --sanity-check   (golden judging is not implemented here)")
        return 2
    result = run_sanity_check_and_probe()
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SANITY_REPORT.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    print(f"written {SANITY_REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
