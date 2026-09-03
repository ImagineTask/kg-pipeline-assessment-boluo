"""Money and duration normalisation.

This happens here, in code, and never in the LLM: the model copies amounts and
deadlines verbatim so they can be checked character-for-character against the
source clause, and the conversion to a number is deterministic.
"""
from __future__ import annotations

import re

MULTIPLIER = {"k": 1_000, "m": 1_000_000, "bn": 1_000_000_000, "b": 1_000_000_000}
CURRENCY = {"£": "GBP", "$": "USD", "€": "EUR"}

MONEY = re.compile(
    r"(?P<sym>[£$€])\s*(?P<num>\d[\d,]*(?:\.\d+)?)\s*(?P<mult>bn|[kmb])?\b", re.I
)
PERCENT = re.compile(r"(?P<num>\d+(?:\.\d+)?)\s*(?:%|per\s*cent)", re.I)
DURATION = re.compile(
    r"(?P<num>\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve)\s*"
    r"(?:\((?:\d+)\)\s*)?"
    r"(?P<unit>Working\s+Day|Business\s+Day|day|week|month|year|hour)s?",
    re.I,
)
WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twelve": 12}


def parse_money(text: str | None) -> dict | None:
    """'£1,000,000' -> 1000000 GBP; '£1m' -> 1000000 GBP; '1%' -> a percentage."""
    if not text:
        return None
    m = MONEY.search(text)
    if m:
        value = float(m.group("num").replace(",", ""))
        mult = (m.group("mult") or "").lower()
        value *= MULTIPLIER.get(mult, 1)
        return {
            "amount_value": value,
            "amount_currency": CURRENCY.get(m.group("sym"), "GBP"),
            "amount_kind": "money",
        }
    p = PERCENT.search(text)
    if p:
        return {"amount_value": float(p.group("num")), "amount_currency": None,
                "amount_kind": "percentage"}
    return None


def parse_duration(text: str | None) -> dict | None:
    """'30 days' -> P30D. Working Days are kept flagged: five Working Days is a
    week and a bit of calendar time, and conflating the two makes every deadline
    comparison wrong."""
    if not text:
        return None
    m = DURATION.search(text)
    if not m:
        return None
    raw = m.group("num").lower()
    n = WORDS.get(raw, None)
    if n is None:
        try:
            n = int(raw)
        except ValueError:
            return None
    unit = re.sub(r"\s+", " ", m.group("unit").lower())
    iso = {
        "hour": f"PT{n}H", "day": f"P{n}D", "working day": f"P{n}D",
        "business day": f"P{n}D", "week": f"P{n}W", "month": f"P{n}M", "year": f"P{n}Y",
    }.get(unit)
    if iso is None:
        return None
    return {
        "duration_iso": iso,
        "duration_value": n,
        "duration_unit": unit,
        "working_days": unit in ("working day", "business day"),
    }
