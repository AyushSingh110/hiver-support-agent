# Taxonomy v1 — CANDIDATE, frozen for the pilot batch only

> **This is not the final taxonomy.** It is a candidate, derived from Phase 4
> evidence, frozen so the pilot batch has a stable target to label against.
>
> It is expected to change. Pilot results decide what `taxonomy_v2` looks like.
>
> **Provenance:** authored by hand from recurring term evidence in 24,190 AmericanAir
> opening messages. It is **not** derived from the Phase 4 cluster assignments, which
> were measurably unreliable (16.56% explained variance, silhouette 0.04, ARI 0.37 —
> see `docs/DECISION_LOG.md` D25).

Label with **exactly one** `primary_intent` per message. Add `secondary_intent` only
when a message genuinely carries two.

---

## 1. `flight_delay`

The customer reports or asks about a flight running late, a tarmac wait, or a delay
threatening a connection.

**Yes:** *"Delayed 3 hours, will I make my DFW connection?"* · *"Sitting on the tarmac
an hour with no update."*
**No:** cancellations → `flight_cancellation_rebooking`. A missed flight through the
customer's own lateness → `booking_fees_and_fare_rules`.

---

## 2. `flight_cancellation_rebooking`

A flight is cancelled, the customer was bumped, or they need to be moved to another
flight.

**Yes:** *"AA1234 cancelled — options to get to ORD tonight?"* · *"Bumped and nobody
will rebook me."*
**No:** a delay that has not become a cancellation → `flight_delay`. A voluntary
change the customer initiates → `booking_fees_and_fare_rules`.

⚠️ **Watch this one.** Its Phase 4 cluster held 15.2% of the corpus but its
centroid-nearest messages were travel photos, not cancellations. The frequency
estimate is weak.

---

## 3. `baggage`

Anything about bags: lost, damaged, delayed, carry-on and gate-check disputes,
baggage fees, overhead space.

**Yes:** *"Bag never arrived at MIA."* · *"Why gate-check a hard-side bag when there's
room?"* · *"$50 to check my carry-on?"*
**No:** a fee complaint with no bag involved → `booking_fees_and_fare_rules`.

⚠️ **Known boundary problem:** a baggage *fee* complaint could go either here or to
intent 6. **For the pilot, put it here** and flag `is_ambiguous = y`. How often that
happens is exactly what decides the rule in v2.

---

## 4. `seating_and_upgrade`

Seat assignments, paid seat selection, upgrades, cabin class.

**Yes:** *"Paid for extra legroom, got moved to a middle seat."* · *"Why did my
upgrade clear then vanish?"*
**No:** boarding order or priority boarding → `boarding_and_gate`.

⚠️ **Open question:** this may be two intents (`seating` and `upgrade`). Label both
here for the pilot; the counts will tell us whether to split.

---

## 5. `boarding_and_gate`

Boarding passes, boarding groups, priority boarding, gate agents, gate changes.

**Yes:** *"Boarding pass won't load in the app."* · *"Why am I group 8 with status?"*
**No:** a delay announced at the gate → `flight_delay`. Rudeness from a gate agent →
`staff_and_service_complaint`.

---

## 6. `booking_fees_and_fare_rules`

Reservations, changes, fare rules, basic economy restrictions, fees, refund requests.

**Yes:** *"Basic economy won't let me pick a seat — correct?"* · *"Charged twice for
one booking."* · *"How do I get a refund?"*
**No:** baggage fees → `baggage` (see the rule above).

⚠️ **Least confident intent.** Refund language never formed its own cluster; it
appeared scattered inside other issues. This may need splitting into `refund_request`
and `fare_rules`, or merging elsewhere.

---

## 7. `staff_and_service_complaint`

Complaints about the conduct of specific staff — flight attendants, gate agents,
phone agents.

**Yes:** *"The flight attendant on AA200 was extremely rude to my mother."*
**No:** *"Worst airline ever"* with no specific person or incident →
`general_dissatisfaction`.

Likely **escalation-relevant**: staff-conduct complaints usually need a human.

---

## 8. `loyalty_and_lounge`

AAdvantage miles, elite status, Admirals Club lounge access.

**Yes:** *"Miles from my October trip never posted."* · *"Admirals Club access on a
partner ticket?"*

⚠️ **May be too rare to keep.** Estimated 1–2%. Its Phase 4 cluster merged Admirals
Club with *vegan creamer*, so the cluster evidence is impure. If the pilot yields
under 2%, this is a merge-or-drop candidate.

---

## 9. `praise_and_compliment`

Positive feedback with no request. Not a support issue, but the agent must recognise
it so it does not invent a problem — and it is an obvious auto-handle case.

**Yes:** *"Huge shout out to Alex at PHX — saved my connection."*
**No:** *"Thanks for finally sorting it, took 3 hours though"* → label the underlying
issue; praise is incidental.

⚠️ **Probably under-counted.** Praise vocabulary (`thank`, `love`, `happy`)
saturated the 38% dump cluster, so the true rate is likely well above its own
cluster's 2.8%.

---

## 10. `general_dissatisfaction`

Venting with no specific, actionable request.

**Yes:** *"I hate @AmericanAir"* · *"Worst airline in America, never again."*
**No:** *"Worst airline — bag missing 4 days"* → `baggage`. There is an actionable
issue underneath the venting.

The clearest **escalate-or-acknowledge** class: no grounded factual reply is possible.

---

## Escape hatches — keep these separate

| Label | Use when | What it signals |
| --- | --- | --- |
| `OTHER` | A real, clear intent that none of the 10 covers | **The taxonomy has a gap.** Please add a note naming the missing intent |
| `UNCLEAR` | You cannot tell what the customer wants | **The message is ambiguous**, not the taxonomy |

These mean different things and must never be collapsed. `OTHER` drives taxonomy
revision; `UNCLEAR` measures how much of the corpus is inherently unlabelable from
the opening message alone.

---

## Intents deliberately excluded from v1

- **`refund_request`** — folded into intent 6 pending evidence; no cluster support.
- **`website_app_technical`** — occasional but no cluster support. If you meet these,
  label `OTHER` and note it. That is how it would earn a place in v2.
- **Any language-based category** — Phase 4 surfaced a Spanish cluster (0.93%), but a
  language is not an intent. Label Spanish messages by their intent if you can read
  them, otherwise `UNCLEAR`.
