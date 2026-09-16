from __future__ import annotations

from dataclasses import asdict, dataclass

from src import config
from src.taxonomy import WEAK_LABEL_RULES

POLICY_VERSION = "esc-v1"

# Only intents with weak-label rules can be predicted. OTHER, UNCLEAR and
# non_support_commentary never reach the classifier's output.
PREDICTABLE_INTENTS = frozenset(WEAK_LABEL_RULES)

# Deliberate system policy, not learned from the dataset. See docs/DECISION_LOG.md.
HUMAN_REVIEW_POLICY_INTENTS = frozenset({
    "staff_and_service_complaint",
    "general_dissatisfaction",
})

EMPTY_REPLY_FLAG = "G1_empty_reply"
CLAIM_FLAGS = frozenset({
    "G2_unsupported_url",
    "G3_unsupported_number",
    "G4_action_claim",
    "G5_resolution_claim",
    "G6_leaked_identifier",
    "G7_excessive_copying",
})
KNOWN_FLAGS = CLAIM_FLAGS | {EMPTY_REPLY_FLAG}

GENERATION_UNUSABLE = "generation_unusable"
GROUNDING_CHECK_FAILED = "grounding_check_failed"
INVALID_EVIDENCE_CITATION = "invalid_evidence_citation"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
EXPLICIT_HUMAN_REVIEW_POLICY = "explicit_human_review_policy"
LOW_INTENT_CONFIDENCE = "low_intent_confidence"
WEAK_GROUNDING_FOR_SUBSTANTIVE_REPLY = "weak_grounding_for_substantive_reply"

RULE_ORDER = (
    GENERATION_UNUSABLE,
    GROUNDING_CHECK_FAILED,
    INVALID_EVIDENCE_CITATION,
    INSUFFICIENT_EVIDENCE,
    EXPLICIT_HUMAN_REVIEW_POLICY,
    LOW_INTENT_CONFIDENCE,
    WEAK_GROUNDING_FOR_SUBSTANTIVE_REPLY,
)

AWAITING_CUSTOMER_DETAILS = "awaiting_customer_details"


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_unit_interval(name: str, value) -> None:
    if not _is_number(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a number in [0, 1], got {value!r}")


@dataclass(frozen=True)
class Thresholds:
    intent_confidence: float
    top1_similarity: float
    min_evidence_count: int

    def __post_init__(self):
        _require_unit_interval("intent_confidence threshold", self.intent_confidence)
        _require_unit_interval("top1_similarity threshold", self.top1_similarity)
        if not _is_int(self.min_evidence_count) or self.min_evidence_count < 0:
            raise ValueError("min_evidence_count must be a non-negative integer")


def default_thresholds() -> Thresholds:
    return Thresholds(
        intent_confidence=config.ESCALATION_MIN_INTENT_CONFIDENCE,
        top1_similarity=config.ESCALATION_MIN_RETRIEVAL_SIMILARITY,
        min_evidence_count=config.ESCALATION_MIN_EVIDENCE_COUNT,
    )


@dataclass(frozen=True)
class EscalationSignals:
    """Everything the decision may look at. There is deliberately no gold-label field."""

    predicted_intent: str
    intent_confidence: float
    evidence_count: int
    top1_similarity: float | None
    reply: str
    parse_error: str | None
    grounding_flag_codes: tuple[str, ...]
    needs_more_information: bool | None
    evidence_used: tuple[int, ...]
    evidence_used_coerced: bool = False
    evidence_intent_matches: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "grounding_flag_codes", tuple(self.grounding_flag_codes))
        object.__setattr__(self, "evidence_used", tuple(self.evidence_used))

        if self.predicted_intent not in PREDICTABLE_INTENTS:
            raise ValueError(f"{self.predicted_intent!r} is not an intent the classifier can predict")
        _require_unit_interval("intent_confidence", self.intent_confidence)

        if not _is_int(self.evidence_count) or self.evidence_count < 0:
            raise ValueError("evidence_count must be a non-negative integer")
        if self.evidence_count == 0:
            if self.top1_similarity is not None:
                raise ValueError("top1_similarity must be None when no evidence was retrieved")
        else:
            _require_unit_interval("top1_similarity", self.top1_similarity)

        if not isinstance(self.reply, str):
            raise ValueError("reply must be a string")
        if self.parse_error is not None and not isinstance(self.parse_error, str):
            raise ValueError("parse_error must be a string or None")

        unknown = set(self.grounding_flag_codes) - KNOWN_FLAGS
        if unknown:
            raise ValueError(f"unknown grounding flag codes: {sorted(unknown)}")

        if self.parsed:
            if not isinstance(self.needs_more_information, bool):
                raise ValueError("needs_more_information must be a boolean for a parsed reply")
            if not all(_is_int(rank) for rank in self.evidence_used):
                raise ValueError("evidence_used must contain integers")
        else:
            # Unparsed output has no model fields; carrying defaults would let rules read them.
            if self.needs_more_information is not None or self.evidence_used:
                raise ValueError("an unparsed reply cannot carry needs_more_information or evidence_used")

        if not isinstance(self.evidence_used_coerced, bool):
            raise ValueError("evidence_used_coerced must be a boolean")
        if self.evidence_intent_matches is not None and (
            not _is_int(self.evidence_intent_matches)
            or not 0 <= self.evidence_intent_matches <= self.evidence_count
        ):
            raise ValueError("evidence_intent_matches must be between 0 and evidence_count")

    @property
    def parsed(self) -> bool:
        return self.parse_error is None


def build_signals(predicted_intent: str, intent_confidence: float,
                  evidence: list[dict], generation: dict) -> EscalationSignals:
    """Assemble signals from a retrieval list and a `generate_llm` result."""
    parsed = generation.get("parse_error") is None
    return EscalationSignals(
        predicted_intent=predicted_intent,
        intent_confidence=float(intent_confidence),
        evidence_count=len(evidence),
        top1_similarity=float(evidence[0]["similarity"]) if evidence else None,
        reply=generation.get("reply") or "",
        parse_error=generation.get("parse_error"),
        grounding_flag_codes=tuple(f["code"] for f in generation.get("grounding_flags", [])),
        needs_more_information=generation["needs_more_information"] if parsed else None,
        evidence_used=tuple(generation.get("evidence_used", [])) if parsed else (),
        evidence_used_coerced=bool(generation.get("evidence_used_coerced", False)),
        evidence_intent_matches=sum(
            1 for item in evidence if item.get("historical_weak_intent") == predicted_intent
        ),
    )


def _fired_rules(s: EscalationSignals, t: Thresholds) -> dict[str, str]:
    """Every rule is evaluated; nothing short-circuits, so the audit lists all reasons."""
    fired: dict[str, str] = {}

    if not s.parsed:
        fired[GENERATION_UNUSABLE] = "The model output could not be parsed, so there is no reply to send."
    elif not s.reply.strip() or EMPTY_REPLY_FLAG in s.grounding_flag_codes:
        fired[GENERATION_UNUSABLE] = "The generated reply is empty."

    claims = sorted(set(s.grounding_flag_codes) & CLAIM_FLAGS)
    if claims:
        fired[GROUNDING_CHECK_FAILED] = f"Grounding checks flagged the reply: {', '.join(claims)}."

    if s.parsed:
        invalid = sorted({r for r in s.evidence_used if not 1 <= r <= s.evidence_count})
        if invalid:
            fired[INVALID_EVIDENCE_CITATION] = (
                f"The reply cites evidence rank(s) {', '.join(map(str, invalid))} "
                f"but only {s.evidence_count} item(s) were retrieved."
            )

    if s.evidence_count < t.min_evidence_count:
        fired[INSUFFICIENT_EVIDENCE] = (
            f"Only {s.evidence_count} evidence item(s) were retrieved; "
            f"at least {t.min_evidence_count} are required."
        )

    if s.predicted_intent in HUMAN_REVIEW_POLICY_INTENTS:
        fired[EXPLICIT_HUMAN_REVIEW_POLICY] = (
            f"Predicted intent '{s.predicted_intent}' is routed to a human by explicit system policy."
        )

    if s.intent_confidence < t.intent_confidence:
        fired[LOW_INTENT_CONFIDENCE] = (
            f"Intent confidence {s.intent_confidence:.4f} is below the threshold "
            f"{t.intent_confidence:.4f}."
        )

    if s.parsed and s.needs_more_information is False:
        if s.top1_similarity is None or s.top1_similarity < t.top1_similarity:
            shown = "unavailable" if s.top1_similarity is None else f"{s.top1_similarity:.4f}"
            fired[WEAK_GROUNDING_FOR_SUBSTANTIVE_REPLY] = (
                f"Top retrieval similarity {shown} is below the threshold "
                f"{t.top1_similarity:.4f} and the reply gives a substantive answer "
                f"rather than asking for details."
            )

    return fired


def decide(signals: EscalationSignals, thresholds: Thresholds | None = None) -> dict:
    thresholds = thresholds or default_thresholds()
    fired = _fired_rules(signals, thresholds)
    reason_codes = [code for code in RULE_ORDER if code in fired]

    informational = []
    if signals.parsed and signals.needs_more_information is True:
        informational.append(AWAITING_CUSTOMER_DETAILS)

    if reason_codes:
        reason = " ".join(fired[code] for code in reason_codes)
    else:
        reason = "No escalation rule fired."
        if informational:
            reason += " The reply asks the customer for more details."

    signal_record = asdict(signals)
    signal_record["grounding_flag_codes"] = list(signals.grounding_flag_codes)
    signal_record["evidence_used"] = list(signals.evidence_used)

    return {
        "decision": "escalate" if reason_codes else "auto_handle",
        "reason_codes": reason_codes,
        "informational_codes": informational,
        "reason": reason,
        "signals": signal_record,
        "thresholds": asdict(thresholds),
        "policy_version": POLICY_VERSION,
    }
