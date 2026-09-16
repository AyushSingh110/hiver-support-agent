from __future__ import annotations

import json
import re

from src import config, llm

PROMPT_VERSION = "p1"

MENTION_PATTERN = re.compile(r"@\w+")
ANON_MENTION_PATTERN = re.compile(r"@\d+")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
CURRENCY_PATTERN = re.compile(r"[\$£€]\s?\d[\d,.]*")
PERCENT_PATTERN = re.compile(r"\b\d{1,3}\s?%")
PHONE_PATTERN = re.compile(r"\b1[-.\s]?800[-.\s]?\S{3,}|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b")
LONG_DIGITS_PATTERN = re.compile(r"\b\d{4,}\b")

ACTION_CLAIM_PATTERN = re.compile(
    r"\b(?:we|i)\s*(?:'ve|have|has)?\s*"
    r"(?:refunded|rebooked|cancelled|canceled|issued|credited|updated|changed|"
    r"processed|upgraded|transferred|booked)\b"
    r"|\byour\s+(?:reservation|booking|ticket|account|refund|seat)\s+has\s+been\b"
    r"|\byou\s+will\s+receive\b",
    re.I,
)
RESOLUTION_CLAIM_PATTERN = re.compile(
    r"\b(?:this|the)\s+(?:issue|problem|matter)\s+(?:is|has been)\s+"
    r"(?:now\s+)?(?:resolved|fixed|sorted|settled)\b"
    r"|\bis\s+now\s+(?:resolved|fixed|sorted)\b",
    re.I,
)

MAX_EVIDENCE_CHARS = 400

PROMPT_TEMPLATE = """You are drafting a public reply for the American Airlines support team on Twitter.

CUSTOMER MESSAGE:
{customer_message}

PREDICTED ISSUE TYPE: {intent}

HISTORICAL EXAMPLES of how this team handled similar messages:
{evidence}

Rules:
1. Reply to this customer's message, not to the historical examples.
2. Treat the historical examples as guidance on approach and tone. They are other
   customers' cases, never facts about this customer.
3. Do not invent policies, fees, URLs, phone numbers, amounts, timelines or compensation.
   Only mention such details if they appear in the historical examples above.
4. Never say an action has been taken and never say the issue is resolved. You cannot
   perform actions on an account.
5. If the historical examples do not cover this issue, say what information you need.
6. Reply only with this JSON object and nothing else:

{{"reply": "<the customer-facing reply>", "needs_more_information": <true|false>, "evidence_used": [<ranks used, may be empty>]}}"""

TEMPLATE_BASELINE_REPLY = (
    "Thanks for reaching out, and sorry for the trouble. So we can look into this, "
    "could you share a few more details about what happened?"
)


def sanitise_evidence_text(text: str) -> str:
    """Strip @mentions so another customer's anonymised id cannot reach the prompt."""
    return re.sub(r"\s+", " ", MENTION_PATTERN.sub("", str(text))).strip()


def format_evidence(evidence: list[dict]) -> str:
    if not evidence:
        return "(no similar historical cases were found)"
    blocks = []
    for item in evidence:
        customer = sanitise_evidence_text(item["historical_customer_message"])[:MAX_EVIDENCE_CHARS]
        reply = sanitise_evidence_text(item["historical_brand_reply"])[:MAX_EVIDENCE_CHARS]
        blocks.append(f"[{item['rank']}] customer: {customer}\n    team replied: {reply}")
    return "\n".join(blocks)


def build_prompt(customer_message: str, intent: str, evidence: list[dict]) -> str:
    return PROMPT_TEMPLATE.format(
        customer_message=sanitise_evidence_text(customer_message),
        intent=intent,
        evidence=format_evidence(evidence),
    )


def parse_response(text: str) -> dict:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"model did not return valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("model returned JSON that is not an object")

    missing = {"reply", "needs_more_information"} - set(payload)
    if missing:
        raise ValueError(f"model response missing required fields: {sorted(missing)}")
    if not isinstance(payload["reply"], str):
        raise ValueError("'reply' must be a string")
    if not isinstance(payload["needs_more_information"], bool):
        raise ValueError("'needs_more_information' must be a boolean")

    used = payload.get("evidence_used", [])
    if not isinstance(used, list) or not all(isinstance(rank, int) for rank in used):
        raise ValueError("'evidence_used' must be a list of integers")

    return {
        "reply": payload["reply"].strip(),
        "needs_more_information": payload["needs_more_information"],
        "evidence_used": used,
    }


def _allowed_text(customer_message: str, evidence: list[dict]) -> str:
    parts = [str(customer_message)]
    for item in evidence:
        parts.append(str(item["historical_customer_message"]))
        parts.append(str(item["historical_brand_reply"]))
    return " ".join(parts)


def check_grounding(reply: str, customer_message: str, evidence: list[dict]) -> list[dict]:
    """Detect classes of unsupported claim. This is not a correctness check."""
    flags: list[dict] = []

    if not reply or not reply.strip():
        flags.append({"code": "G1_empty_reply", "detail": "reply is blank"})
        return flags

    allowed = _allowed_text(customer_message, evidence)
    allowed_urls = set(URL_PATTERN.findall(allowed))
    for url in set(URL_PATTERN.findall(reply)):
        if url not in allowed_urls:
            flags.append({"code": "G2_unsupported_url", "detail": url})

    for name, pattern in (
        ("currency", CURRENCY_PATTERN),
        ("percentage", PERCENT_PATTERN),
        ("phone", PHONE_PATTERN),
        ("long_number", LONG_DIGITS_PATTERN),
    ):
        for match in set(pattern.findall(reply)):
            if match.strip() not in allowed:
                flags.append({
                    "code": "G3_unsupported_number",
                    "detail": f"{name}: {match.strip()}",
                })

    for match in ACTION_CLAIM_PATTERN.finditer(reply):
        flags.append({"code": "G4_action_claim", "detail": match.group(0).strip()})

    match = RESOLUTION_CLAIM_PATTERN.search(reply)
    if match:
        flags.append({"code": "G5_resolution_claim", "detail": match.group(0)})

    for mention in set(ANON_MENTION_PATTERN.findall(reply)):
        flags.append({"code": "G6_leaked_identifier", "detail": mention})

    return flags


def _result(system: str, reply: str, customer_message: str, evidence: list[dict],
            needs_more: bool, used: list[int], extra: dict | None = None) -> dict:
    flags = check_grounding(reply, customer_message, evidence)
    result = {
        "system": system,
        "reply": reply,
        "needs_more_information": needs_more,
        "evidence_used": used,
        "grounding_flags": flags,
        "grounding_passed": not flags,
        "evidence_count": len(evidence),
        "top_similarity": evidence[0]["similarity"] if evidence else None,
    }
    result.update(extra or {})
    return result


def generate_baseline_template(customer_message: str, evidence: list[dict]) -> dict:
    """Baseline B: a fixed generic support response that ignores the evidence."""
    return _result("baseline_b_template", TEMPLATE_BASELINE_REPLY, customer_message, evidence,
                   needs_more=True, used=[], extra={"prompt_version": None, "model": None})


def generate_baseline_echo(customer_message: str, evidence: list[dict]) -> dict:
    """Baseline A: return the top-ranked historical brand reply verbatim, sanitised."""
    reply = sanitise_evidence_text(evidence[0]["historical_brand_reply"]) if evidence else ""
    return _result("baseline_a_echo_top1", reply, customer_message, evidence,
                   needs_more=not bool(reply), used=[1] if evidence else [],
                   extra={"prompt_version": None, "model": None})


def generate_llm(customer_message: str, intent: str, evidence: list[dict], *,
                 model: str | None = None, use_cache: bool = True,
                 complete_fn=None) -> dict:
    """Grounded generation. `complete_fn` is injectable so tests never need Ollama."""
    prompt = build_prompt(customer_message, intent, evidence)
    call = complete_fn or llm.complete
    response = call(prompt, model=model or config.GENERATION_MODEL, use_cache=use_cache)

    try:
        parsed = parse_response(response["text"])
    except ValueError as error:
        return _result(
            "llm_grounded", "", customer_message, evidence, needs_more=False, used=[],
            extra={
                "prompt_version": PROMPT_VERSION,
                "model": response.get("model"),
                "parse_error": str(error),
                "raw_text": response["text"][:500],
                "latency_seconds": response.get("latency_seconds"),
                "from_cache": response.get("from_cache"),
            },
        )

    return _result(
        "llm_grounded", parsed["reply"], customer_message, evidence,
        needs_more=parsed["needs_more_information"], used=parsed["evidence_used"],
        extra={
            "prompt_version": PROMPT_VERSION,
            "model": response.get("model"),
            "parse_error": None,
            "latency_seconds": response.get("latency_seconds"),
            "from_cache": response.get("from_cache"),
        },
    )
