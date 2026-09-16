from __future__ import annotations

import re

TAXONOMY_VERSION = "v2"

INTENTS = [
    "flight_delay",
    "flight_cancellation_rebooking",
    "baggage",
    "seating_and_upgrade",
    "boarding_and_gate",
    "booking_fees_and_fare_rules",
    "staff_and_service_complaint",
    "loyalty_and_lounge",
    "praise_and_compliment",
    "general_dissatisfaction",
    "non_support_commentary",
]
ESCAPE_LABELS = ["OTHER", "UNCLEAR"]
ALL_LABELS = INTENTS + ESCAPE_LABELS

# Intents whose correct handling is a human, regardless of model confidence.
ALWAYS_ESCALATE = {"staff_and_service_complaint", "general_dissatisfaction", "OTHER", "UNCLEAR"}

# Weak-label patterns derived from the taxonomy v2 definitions, not from Phase 4
# clusters. Precision matters far more than recall here: an ambiguous message is
# left unlabelled rather than guessed at.
WEAK_LABEL_RULES: dict[str, list[str]] = {
    "flight_delay": [
        r"\bdelay(ed|s|ing)?\b", r"\btarmac\b", r"\bstill waiting\b", r"\bhours? late\b",
        r"\brunning late\b", r"\bsat on the plane\b",
    ],
    "flight_cancellation_rebooking": [
        r"\bcancell?(ed|ation|ing)\b", r"\brebook(ed|ing)?\b", r"\bbumped\b",
        r"\bmissed (my |our )?connection\b", r"\bstandby\b", r"\brerout(e|ed|ing)\b",
    ],
    "baggage": [
        r"\bbag(s|gage)?\b", r"\bluggage\b", r"\bsuitcase\b", r"\bcarry[- ]on\b",
        r"\bgate[- ]check\b", r"\boverhead bin\b",
    ],
    "seating_and_upgrade": [
        r"\bseat(s|ing)?\b", r"\bupgrade(d|s)?\b", r"\bfirst class\b", r"\bbusiness class\b",
        r"\blegroom\b", r"\bmain cabin extra\b",
    ],
    "boarding_and_gate": [
        r"\bboarding pass\b", r"\bboarding group\b", r"\bpriority boarding\b",
        r"\bgate agent\b", r"\bgate change\b", r"\bpre[- ]?board\b",
    ],
    "booking_fees_and_fare_rules": [
        r"\brefund(ed|s)?\b", r"\bbasic economy\b", r"\bfare(s)?\b", r"\bvoucher\b",
        r"\bchange fee\b", r"\bcharged twice\b", r"\breservation\b", r"\bbooking\b",
    ],
    "staff_and_service_complaint": [
        r"\bflight attendant\b", r"\bwas rude\b", r"\brude (to|staff|agent)\b",
        r"\bunprofessional\b", r"\bdisrespectful\b",
    ],
    "loyalty_and_lounge": [
        r"\baadvantage\b", r"\bexecutive platinum\b", r"\badmirals club\b",
        r"\bflagship lounge\b", r"\belite status\b", r"\bmiles (posted|missing|credited)\b",
    ],
    "praise_and_compliment": [
        r"\bthank(s| you)\b", r"\bshout ?out\b", r"\bkudos\b", r"\bexcellent service\b",
        r"\bgreat (crew|service|flight|job)\b", r"\bamazing (crew|service|staff)\b",
    ],
    "general_dissatisfaction": [
        r"\bworst airline\b", r"\bnever fly(ing)? (with )?(you|aa|american)\b",
        r"\bnever again\b", r"\bdisgrace\b", r"\bpathetic\b",
    ],
}

COMPILED_RULES = {
    intent: [re.compile(pattern, re.I) for pattern in patterns]
    for intent, patterns in WEAK_LABEL_RULES.items()
}

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


def normalise(text: str) -> str:
    stripped = MENTION_PATTERN.sub(" ", URL_PATTERN.sub(" ", str(text).lower()))
    return " ".join(TOKEN_PATTERN.findall(stripped))


def weak_label(text: str) -> tuple[str | None, dict[str, int]]:
    """Assign an intent only when exactly one intent matches most strongly."""
    scores = {
        intent: sum(1 for pattern in patterns if pattern.search(str(text)))
        for intent, patterns in COMPILED_RULES.items()
    }
    scores = {intent: score for intent, score in scores.items() if score > 0}
    if not scores:
        return None, {}

    best = max(scores.values())
    winners = [intent for intent, score in scores.items() if score == best]
    return (winners[0] if len(winners) == 1 else None), scores
