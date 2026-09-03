"""Regexes and predicates shared by the stitcher and the chunker.

Kept in one place because stitching and chunking must agree on what a "new
structural unit" looks like: if they disagree, the stitcher joins across a
boundary the chunker then tries to split, and clauses come out mangled.
"""
from __future__ import annotations

import re

# --------------------------------------------------------------------------- #
# document-level headings
# --------------------------------------------------------------------------- #
# Some schedule titles are printed with the framework reference in front of them
# ("RM6116 Call-Off Schedule 24 (Supplier Furnished Terms)"), so the prefix is
# optional rather than absent.
DOC_HEADING = re.compile(
    r"^\s*(?:RM\d+\s+(?:Network Services\s+\d+\s+)?)?("
    r"Core Terms"
    r"|Framework Award Form"
    r"|(?:Framework|Joint|Call[- ]?Off)\s+Schedule\s+\d+"
    r")\b",
    re.IGNORECASE,
)

SUB_HEADING = re.compile(
    r"^\s*(Annex\s+[A-Z]?\d*[A-Z]?\d*|Part\s+[A-Z]\b|Appendix\s+\d+|Section\s+[A-Z0-9]+)\b",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------- #
# clause numbering
# --------------------------------------------------------------------------- #
CLAUSE_NUMBER = re.compile(r"^\s*(\d+(?:\.\d+){0,3})\.?\s+(?=\S)")
LETTERED_LIMB = re.compile(r"^\s*\(([a-z]{1,2}|[ivxlcdm]{1,6})\)\s+", re.IGNORECASE)
ROMAN_LIMB = re.compile(r"^\s*\((?:i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii)\)\s+")

TERMINAL_PUNCT = tuple(".:;?!”’\")")


def is_doc_heading(text: str) -> bool:
    return bool(DOC_HEADING.match(text.strip()))


def is_sub_heading(text: str) -> bool:
    return bool(SUB_HEADING.match(text.strip()))


def clause_number(text: str) -> str | None:
    m = CLAUSE_NUMBER.match(text)
    return m.group(1) if m else None


def is_all_caps_heading(text: str) -> bool:
    t = text.strip()
    if len(t) < 4 or len(t) > 120:
        return False
    letters = [c for c in t if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def starts_new_unit(text: str) -> bool:
    """True when this text begins a structurally new block that must not be
    joined onto whatever preceded it."""
    t = text.strip()
    if not t:
        return False
    return bool(
        clause_number(t)
        or is_doc_heading(t)
        or is_sub_heading(t)
        or is_all_caps_heading(t)
    )


def is_heading_like(text: str) -> bool:
    """True when this text is itself a heading, and so must not absorb the block
    that follows it even though it lacks terminal punctuation."""
    t = re.sub(r"\s+", " ", text).strip()
    if not t:
        return False
    if is_doc_heading(t) or is_sub_heading(t) or is_all_caps_heading(t):
        return True
    # "3. What needs to be delivered" - numbered clause heading, short, no full stop
    m = CLAUSE_NUMBER.match(t)
    if m and len(t) < 90 and not t.endswith("."):
        rest = t[m.end():]
        if rest[:1].isupper() and rest.count(" ") < 12:
            return True
    return False


def ends_open(text: str) -> bool:
    """Sentence is unfinished: no terminal punctuation at the end."""
    t = re.sub(r"\s+", " ", text).rstrip()
    return bool(t) and not t.endswith(TERMINAL_PUNCT)


def ends_hyphenated(text: str) -> bool:
    t = text.rstrip()
    return t.endswith("-") and len(t) > 1 and t[-2].isalpha()


def unbalanced_parens(text: str) -> bool:
    return text.count("(") > text.count(")")


# A list limb ends "...; and" / "...; or" and is complete; the conjunction is
# dropped before an ending is judged so the semicolon shows through.
TRAILING_CONJUNCTION = re.compile(r"[;,]\s*(?:and/or|and|or)\s*$", re.I)

# Endings that can only be mid-sentence. This is the shape clause 3.2.9 had
# ("...the Buyer needs to make use of the") before the stitcher was fixed.
TRUNCATED = re.compile(
    r"(,|-|\b(?:the|a|an|of|to|in|on|for|with|and|or|by|at|from|that|which|as|"
    r"is|are|be|been|shall|must|may|any|all|its|their|this|these|such|under|"
    r"between|including|pursuant"
    # A reference-introducing noun never ends a sentence: "...as described in
    # paragraphs" is a clause whose numbers were lost at the page break, and the
    # continuation begins with something that looks exactly like a clause number.
    # Seventeen clauses in this document were truncated this way, and only the
    # independent graph-quality check found them.
    r"|paragraphs?|clauses?|schedules?|annexe?s?|parts?|appendix|appendices"
    r"|sections?|regulations?|lots?"
    r")\s*)$",
    re.I,
)


def ending(text: str) -> str:
    return TRAILING_CONJUNCTION.sub(";", text.rstrip())


def is_truncated(text: str) -> bool:
    return bool(TRUNCATED.search(ending(text)))
