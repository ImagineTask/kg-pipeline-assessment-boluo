"""Stage 1.5.1 - definitions and defined-term linking. No model involved.

Joint Schedule 1 is the global dictionary. Individual schedules also define terms
locally, and a local definition overrides the global one *within its own
document* - so scope is carried on every record and resolved at lookup time.
"""
from __future__ import annotations

import re
from collections import Counter

from rapidfuzz import fuzz

from src.common import load_jsonl, path, report, write_jsonl

GLOBAL_DEFS_DOC = "joint_schedule_1"
MAX_TERM_WORDS = 9
MAX_TERM_CHARS = 70
OCR_MATCH_THRESHOLD = 90


def plausible_term(term: str) -> bool:
    """A defined term is a short, capitalised noun phrase.

    Without this the two-column reader picks up stray left-column fragments -
    a bare "for", a wrapped sentence - and they become gazetteer entries that
    match thousands of times.
    """
    if not (2 <= len(term) <= MAX_TERM_CHARS):
        return False
    if not term[0].isupper():
        return False
    if len(term.split()) > MAX_TERM_WORDS:
        return False
    return any(ch.isalpha() for ch in term)


def reconcile_term(term: str, page_text: str, corpus: str) -> str | None:
    """Repair an OCR error in a defined term using the parallel pdftotext text.

    Document AI reads "Occasion of Tax Non-Compliance" as "Occasion of Tax
    Jon-Compliance" on page 129. A character substitution like that is invisible
    to a character-count diff, but it silently breaks the gazetteer entry for one
    of the document's more important defined terms. The second extraction already
    exists, so the term is checked against it.
    """
    # The term as Document AI read it agrees with the second extraction, or the
    # document uses it elsewhere (its own definition chunk is the one hit that
    # does not count) - either way there is nothing to repair.
    if term in page_text or corpus.count(term) > 1:
        return None
    # Search the whole second extraction, not just the term's own page: with
    # `-layout` the term column wraps around the definition column, so the intact
    # phrase may not sit on one line where the term is defined - but it does
    # wherever the document *uses* the term.
    words = page_text.split()
    n = len(term.split())
    best, best_score = None, 0.0
    for size in {max(n - 1, 1), n, n + 1}:
        for i in range(len(words) - size + 1):
            candidate = " ".join(words[i:i + size])
            if abs(len(candidate) - len(term)) > 6:
                continue
            score = fuzz.ratio(candidate, term)
            if score > best_score:
                best, best_score = candidate, score
    if not best or best_score < OCR_MATCH_THRESHOLD:
        return None
    fixed = best.strip(' "\u201c\u201d\'.,;:!?)]}')
    # `pdftotext -layout` prints the term column and the definition column on the
    # same physical line, so a word window can straddle the two ("Call-Off Start
    # Date" -> "Call-Off Start the"). The decisive test is that a real defined
    # term is *used* elsewhere in the document; a straddled fragment never is.
    return fixed if fixed and fixed in corpus else None

# strip the leading quoted term from the chunk text to get the definition body
LEADING_TERM = re.compile(r'^\s*[“"\']?[^"”\']{0,90}[”"\']?\s*(?=\S)')
MEANS = re.compile(r"\b(means|shall mean|has the meaning|shall have the meaning)\b", re.I)


def definition_body(text: str, term: str) -> str:
    body = text.strip()
    if body.startswith('"'):
        end = body.find('"', 1)
        if end != -1:
            body = body[end + 1:]
    elif body.lower().startswith(term.lower()):
        body = body[len(term):]
    return body.strip(" \t\n:;-")


def build(clauses: list[dict], page_text: dict[int, str]) -> tuple[list[dict], list[dict], list[dict]]:
    defs: list[dict] = []
    seen: set[tuple[str, str]] = set()
    corrections: list[dict] = []
    rejected: list[dict] = []
    # every clause's text, used to confirm a repaired term is one the document
    # actually uses
    corpus = "\n".join(c["text"] for c in clauses)
    reference_text = "\n".join(page_text[p] for p in sorted(page_text))
    for c in clauses:
        if c["chunk_type"] != "definition" or not c["heading"]:
            continue
        term = re.sub(r"\s+", " ", c["heading"]).strip(' "“”\'’:;,')
        if not plausible_term(term):
            rejected.append({"term": term[:80], "clause_id": c["clause_id"]})
            continue
        fixed = reconcile_term(term, reference_text, corpus)
        if fixed and plausible_term(fixed):
            corrections.append({"from": term, "to": fixed, "page": c["page_start"]})
            term = fixed
        scope = "global" if c["doc_id"] == GLOBAL_DEFS_DOC else "document_local"
        key = (term.lower(), c["doc_id"])
        if key in seen:
            continue
        seen.add(key)
        defs.append(
            {
                "term": term,
                "heading_raw": re.sub(r"\s+", " ", c["heading"]).strip(' "“”\'’:;,'),
                "definition_text": definition_body(c["text"], term),
                "scope": scope,
                "defined_in": c["doc_id"],
                "clause_id": c["clause_id"],
                "page_start": c["page_start"],
                "page_end": c["page_end"],
                "hierarchy_path": c["hierarchy_path"],
            }
        )
    return defs, corrections, rejected


def gazetteer(defs: list[dict]) -> list[tuple[str, re.Pattern, list[dict]]]:
    """Terms sorted longest-first so 'Key Subcontractor' wins over 'Subcontractor'
    and 'Material Default' over 'Default'."""
    by_term: dict[str, list[dict]] = {}
    for d in defs:
        by_term.setdefault(d["term"], []).append(d)
    out = []
    for term in sorted(by_term, key=lambda t: (-len(t), t)):
        # Exact capitalised form on word boundaries, plus the regular plural and
        # possessive. The contract defines "Service Offer" and then uses "Service
        # Offers" throughout; without this, 73 defined terms go unlinked wherever
        # they appear in the plural.
        forms = [re.escape(term)]
        if not term.endswith("s"):
            forms.append(re.escape(term) + "s")
        if term.endswith("y") and len(term) > 2:
            forms.append(re.escape(term[:-1]) + "ies")
        alternation = "|".join(sorted(forms, key=len, reverse=True))
        pattern = re.compile(rf"(?<![\w-])(?:{alternation})(?:'s|\u2019s)?(?![\w-])")
        out.append((term, pattern, by_term[term]))
    return out


def link(clauses: list[dict], defs: list[dict]) -> list[dict]:
    """Emit USES_TERM edges. Longer terms are matched first and their spans are
    masked out, so a nested shorter term cannot also claim the same text."""
    gaz = gazetteer(defs)
    edges: list[dict] = []
    for c in clauses:
        text = c["text"]
        if not text:
            continue
        mask = bytearray(len(text))
        counts: Counter[str] = Counter()
        for term, pattern, entries in gaz:
            for m in pattern.finditer(text):
                if any(mask[m.start():m.end()]):
                    continue
                mask[m.start():m.end()] = b"\x01" * (m.end() - m.start())
                counts[term] += 1
        for term, n in counts.items():
            # a local definition in this clause's own document beats the global one
            entries = next(e for t, _, e in gaz if t == term)
            local = [e for e in entries if e["defined_in"] == c["doc_id"]]
            chosen = (local or [e for e in entries if e["scope"] == "global"] or entries)[0]
            edges.append(
                {
                    "type": "USES_TERM",
                    "source": c["clause_id"],
                    "target": chosen["term"],
                    "occurrences": n,
                    "definition_scope": chosen["scope"],
                    "defined_in": chosen["defined_in"],
                    "resolved": True,
                }
            )
    return edges


def main() -> None:
    clauses = load_jsonl(path("clauses"))
    page_text = {p["page"]: p["text"] for p in load_jsonl(path("pdftotext_pages"))}
    defs, corrections, rejected = build(clauses, page_text)
    write_jsonl(path("definitions"), defs)
    edges = link(clauses, defs)
    write_jsonl(path("term_edges"), edges)

    by_scope = Counter(d["scope"] for d in defs)
    overrides = sorted(
        {d["term"] for d in defs if d["scope"] == "document_local"}
        & {d["term"] for d in defs if d["scope"] == "global"}
    )
    report(
        "definitions_report.json",
        {
            "definitions": len(defs),
            "ocr_corrected_terms": corrections,
            "rejected_term_candidates": len(rejected),
            "rejected_sample": rejected[:25],
            "by_scope": dict(by_scope),
            "documents_with_local_definitions": len(
                {d["defined_in"] for d in defs if d["scope"] == "document_local"}
            ),
            "local_overrides_of_global_terms": len(overrides),
            "local_override_terms": overrides[:60],
            "use_edges": len(edges),
            "clauses_with_terms": len({e["source"] for e in edges}),
            "most_used_terms": Counter(
                {t: sum(e["occurrences"] for e in edges if e["target"] == t)
                 for t in {e["target"] for e in edges}}
            ).most_common(25),
        },
    )
    print(f"OCR-corrected terms: {len(corrections)}; rejected candidates: {len(rejected)}")
    print(
        f"{len(defs)} definitions ({by_scope['global']} global, "
        f"{by_scope['document_local']} document-local, {len(overrides)} local overrides) | "
        f"{len(edges)} USES_TERM edges over {len({e['source'] for e in edges})} clauses"
    )


if __name__ == "__main__":
    main()
