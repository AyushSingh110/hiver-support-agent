# Taxonomy v2 — AmericanAir support intents

> **Status: FROZEN for the golden set — not yet validated.**
>
> Frozen and unvalidated are both true and are not the same thing. *Frozen* means the
> label set and definitions will not change while the golden set is annotated, so
> labels cannot be tuned to the evaluation. *Not yet validated* means these boundaries
> were written **after** reading the 148-row pilot, so measuring them on that pilot
> would be circular. The golden set is their first honest test.
>
> **Changes from v1:** 1 intent added, 1 broadened, 7 definitions sharpened.
> **No intent was merged or removed.**

Label with **exactly one** `primary_intent`. Add `secondary_intent` only when a
message carries two genuinely separate requests.

---

## 1. `flight_delay`

**Definition.** The customer's flight is running late but is still operating as their
flight.

**Include:** delay reports, tarmac waits, "how long will this be", a delay putting a
connection at risk but not yet missed.
**Exclude:** the flight was cancelled → 2. The customer has been moved to a different
flight → 2.

**Closest confusing intent:** `flight_cancellation_rebooking`.
**Tie-breaker:** *Is the customer still on their original flight?* Yes → here.

---

## 2. `flight_cancellation_rebooking`

**Definition.** The flight is cancelled, the customer was bumped, or they need to be
moved to a different flight.

**Include:** cancellations, involuntary bumping, rebooking requests, a missed
connection that now requires a new flight, standby requests.
**Exclude:** a late flight still operating → 1. A voluntary itinerary change the
customer initiates for their own reasons → 6.

**Closest confusing intent:** `flight_delay`.
**Tie-breaker:** *Has the itinerary changed, or must it change?* Yes → here. A delay
that **caused** a missed connection belongs here, because the required action is
rebooking, not information about a delay.

---

## 3. `baggage`

**Definition.** Any message where a bag is the subject.

**Include:** lost, damaged or delayed bags, gate-check disputes, carry-on size rules,
overhead space, **and baggage fees**.
**Exclude:** a fee complaint with no bag involved → 6.

**Closest confusing intent:** `booking_fees_and_fare_rules`.
**Tie-breaker:** *Is a bag involved?* Yes → here, **even when the complaint is about
money**. This rule exists because v1 left the baggage-fee boundary undefined and it
was the most flagged ambiguity in the pilot.

---

## 4. `seating_and_upgrade`

**Definition.** Where the customer sits — seat assignment, paid seat selection,
upgrades, cabin class.

**Include:** seat assignment problems, paid-seat disputes, upgrade requests and
status, first/business class questions.
**Exclude:** boarding order or priority boarding → 5.

**Closest confusing intent:** `boarding_and_gate`.
**Tie-breaker:** *Where you sit* → here. *When or how you board* → 5.

---

## 5. `boarding_and_gate`

**Definition.** Boarding passes, boarding groups, priority boarding, gate agents,
gate changes.

**Include:** boarding pass problems, boarding group disputes, gate change confusion,
denied boarding at the gate.
**Exclude:** the seat itself → 4. Rudeness from a gate agent → 7.

**Closest confusing intent:** `seating_and_upgrade`.
**Tie-breaker:** as in 4. If the complaint is about a *person's conduct* rather than
the process, use 7.

---

## 6. `booking_fees_and_fare_rules`

**Definition.** Reservations, voluntary changes, fare rules, basic economy
restrictions, fees and refunds.

**Include:** booking errors, double charges, fare rule questions, basic economy
restrictions, refund requests, voluntary date changes.
**Exclude:** anything involving a bag → 3. Anything about the loyalty programme or a
loyalty account → 8.

**Closest confusing intents:** `baggage`, `loyalty_and_lounge`.
**Tie-breaker:** *No bag* **and** *not the loyalty programme* → here.

---

## 7. `staff_and_service_complaint`

**Definition.** The conduct of a specific, identifiable staff member.

**Include:** named or locatable staff behaving rudely or unprofessionally ("the flight
attendant on AA200", "the agent at gate F26").
**Exclude:** anonymous anger at the airline → 10.

**Closest confusing intent:** `general_dissatisfaction`.
**Tie-breaker:** *Is a specific person identifiable?* Yes → here. **Likely
escalation-relevant**: staff-conduct complaints usually need a human.

---

## 8. `loyalty_and_lounge` *(broadened in v2)*

**Definition.** The AAdvantage programme, elite status, lounge access, **and loyalty
account or profile administration**.

**Include:** miles not posting, status questions and disputes, Admirals Club and
Flagship Lounge access, **account/profile changes such as a name change, account
merge or login problem**.
**Exclude:** a fare or booking question that merely mentions status → 6.

**Closest confusing intent:** `booking_fees_and_fare_rules`.
**Tie-breaker:** *Is it about the programme or the account, rather than one specific
booking?* Yes → here.

**Why broadened.** v1 listed only "miles, elite status, lounge access", so a clear
request — *"Got married, need to change my last name on my AAdvantage"* — had no home
and was labelled `UNCLEAR`. The message fit the intent's spirit but not its letter.

---

## 9. `praise_and_compliment` *(sharpened in v2)*

**Definition.** Positive feedback that **evaluates AA's service, staff or
experience**.

**Include:** praise of named staff, praise of a flight or service quality, thanks for
a problem resolved.
**Exclude:** sharing a photo, view, or news with no judgement of service → 11.
Positive words attached to a live complaint → label the underlying issue.

**Closest confusing intent:** `non_support_commentary`.

**Tie-breakers, in this order:**

1. **Author identity (apply first).** If the message clearly appears to be authored by
   American Airlines staff, crew, or an organizational representative rather than a
   customer, classify it as `non_support_commentary`, **even when it expresses
   positive sentiment about American Airlines**. See the note under 11.
2. *Is AA's service, staff or experience being evaluated?* Yes → here. Merely
   observing or sharing → 11.

---

## 10. `general_dissatisfaction` *(sharpened in v2)*

**Definition.** Negative sentiment with **no identifiable actionable issue**.

**Include:** venting, "worst airline", generalised frustration with no specific
request or incident.
**Exclude:** venting that also names a real problem → that problem's intent. A
specific named staff member → 7. A real issue with no matching intent → `OTHER`.

**Closest confusing intents:** `OTHER`, `non_support_commentary`.
**Tie-breaker:** *Is there an actionable issue?* Yes → its intent, or `OTHER`. No, and
the sentiment is negative → here. No, and the sentiment is neutral or positive → 11.

---

## 11. `non_support_commentary` *(new in v2)*

**Definition.** Social or travel commentary with **no request and no evaluation of
AA's service**.

**Include:** travel photos and views, lounge or airport check-ins, sharing AA news or
route announcements, aviation enthusiasm, neutral observations about flying.
**Exclude:** any request or question → its intent. Any judgement of service, positive
→ 9 or negative → 10.

**Closest confusing intent:** `praise_and_compliment`.

**Tie-breakers, in this order:**

1. **Author identity (apply first).** If the message clearly appears to be authored by
   American Airlines staff, crew, or an organizational representative rather than a
   customer, classify it as `non_support_commentary`, **even when it expresses
   positive sentiment about American Airlines**.
2. *Is there a request or a service judgement?* Neither → here.

**How to apply the author-identity test.** The dataset provides **no author-role
metadata**, so this is a **textual heuristic** judged from the message itself.
Self-identifying phrasing is the signal: *"our employees"*, *"privilege to serve"*,
*"our team"*, first-person plural on AA's behalf. It will miss insiders who do not
self-identify, and it should not be used to guess at an author's employer from
enthusiasm alone.

**Why this test exists.** Two pilot messages praise AA warmly but are written from
inside the company (*"It was a privilege to serve…"*, *"the families of **our**
employees"*). They are not customer feedback, so treating them as praise would
pollute an intent meant to capture what customers say about service. Without this
rule the taxonomy could not reproduce its own final labels for those rows.

**Why added.** `OTHER` was conflating two things needing **opposite** handling:
genuine long-tail support issues that likely need a human, and social posts that
should never be escalated. Separating them is a workflow distinction, not a cosmetic
one. Supported by 9 clear random-stratum examples (~7.6%).

---

## `OTHER` *(sharpened in v2)*

**Definition.** A **genuine support issue** that none of intents 1–11 covers.

**Include:** real problems with no home — ground transport, airport security, broken
web forms, unreachable phone lines, passenger documentation, accessibility.
**Exclude:** no support issue at all → 11. Cannot tell what they want → `UNCLEAR`.

**Closest confusing intents:** `non_support_commentary`, `UNCLEAR`.
**Tie-breaker:** *Is there a support issue at all?* No → 11. Yes, but no intent fits →
here.

**Retained deliberately.** The pilot showed a real long tail: seven `OTHER` rows
covering seven unrelated issues. A catch-all invented to shrink this percentage would
hide the tail rather than handle it. **Add a note** naming the issue whenever you use
`OTHER`.

---

## `UNCLEAR` *(sharpened in v2)*

**Definition.** You cannot determine what the customer wants.

**Include:** messages too vague, fragmentary or context-dependent to interpret;
messages in a language you cannot read.
**Exclude:** you know what they want but no intent fits → `OTHER`. There is no
request at all but the message is perfectly clear → 11.

**Closest confusing intent:** `OTHER`.
**Tie-breaker:** *Can you tell what they want?* Yes but no slot → `OTHER`. No →
`UNCLEAR`.

**Note.** `UNCLEAR` describes a property of the **message**. `OTHER` describes a gap
in the **taxonomy**. Keeping them separate is what lets us tell "our categories are
incomplete" from "this message is uninterpretable".

**Retained deliberately despite low frequency.** After the v2 relabelling, `UNCLEAR`
stands at **2 of 120 random-stratum rows (1.7%)**, below the 2% threshold that would
normally flag a label as a merge-or-drop candidate. It is kept anyway, for three
reasons:

1. A 148-row pilot is **not sufficient evidence** to remove a safeguard.
2. The drop happened **because v2 worked** — messages that were clear but contained no
   request moved to `non_support_commentary`, which is exactly the intended effect.
   Falling frequency here is a success signal, not a redundancy signal.
3. A taxonomy with no way to say "I cannot tell" will instead say something wrong. The
   cost of keeping a rarely-used escape hatch is far lower than the cost of forcing
   uninterpretable messages into a confident label.

Revisit only if the golden set also shows near-zero use.

---

## Intents deliberately **not** added in v2

**`inflight_experience`** (WiFi, seat comfort, catering, entertainment). The pilot
produced **one** random-stratum example (0.8%), below the 2% threshold. The other
candidates already belong elsewhere. Such messages go to `OTHER` for now; revisit if
the golden set shows more.

**A merged `flight_disruption`.** `flight_delay` and `flight_cancellation_rebooking`
stay separate because they lead to materially different support actions — information
versus rebooking. The 67% non-high confidence on cancellation in the pilot was a
*definition* problem, addressed by the tie-breaker in 2.

---

## Boundary summary

| Pair | Question to ask |
| --- | --- |
| delay vs cancellation | Still on the original flight? |
| baggage vs booking fees | Is a bag involved? |
| seating vs boarding | Where you sit, or when you board? |
| loyalty vs booking fees | The programme/account, or one booking? |
| **praise vs commentary (first)** | **Is the author a customer, or AA staff/crew?** |
| praise vs commentary (second) | Is service being evaluated? |
| dissatisfaction vs OTHER | Is there an actionable issue? |
| OTHER vs UNCLEAR | Can you tell what they want? |
| OTHER vs commentary | Is there a support issue at all? |
