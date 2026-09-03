"""Stage 1.5.2 - cross-reference resolution. Entirely deterministic.

The rule that matters more than any regex:

    `Clause N`     always means Core Terms, wherever it is written.
    `Paragraph N`  means paragraph N of *the schedule the reference sits in*,
                   unless a schedule is named explicitly.

A resolver that ignores the containing document produces mostly wrong edges, and
they look plausible - which is worse than missing ones, because they silently
misroute the agent. Every reference that cannot be resolved is written to
unresolved.jsonl with a reason; none is dropped silently.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from src.common import SETTINGS, load_jsonl, path, report, write_jsonl

CFG = SETTINGS["crossrefs"]
CORE = "core_terms"

SCHEDULE = r"(?:Framework|Joint|Call[-\s]?Off)\s+Schedule\s+\d+"
NUMS = r"\d+(?:\.\d+){0,3}"
NUM_LIST = rf"{NUMS}(?:\s*(?:,|\band\b|\bto\b|\bor\b|&|-|–)\s*{NUMS})*"

REFERENCE = re.compile(
    rf"""
    (?P<self>\bthis\s+(?P<selfkind>Clause|Paragraph|Schedule|Annex|Part|Appendix)\b)
    # "Paragraphs 2 to 4 of Part C" - the qualifier decides which Part's numbering
    # applies, and ignoring it lands the edge on the same number in the wrong Part.
  | (?P<qualified>\b(?P<qkind>Paragraph|Clause)s?\s+(?P<qnums>{NUM_LIST})\s+of\s+
        (?-i:(?P<qsectype>Part|Annex|Appendix))\s+(?P<qseclabel>[A-Z]\d*|\d+[A-Z]?)\b)
  | (?P<sched_sub>(?P<sched1>{SCHEDULE})(?:\s*\([^)]{{0,80}}\))?\s*,?\s*
        (?P<subkind>Paragraph|Clause|Annex|Part|Appendix)s?\s+(?P<subnums>{NUM_LIST}|[A-Z]\d*))
  | (?P<sched_only>(?P<sched2>{SCHEDULE}))
  | (?P<numbered>\b(?P<kind>Clause|Paragraph)s?\s+(?P<nums>{NUM_LIST}))
  | (?P<named>\b(?-i:(?P<namedkind>Annex|Part|Appendix))\s+(?P<label>[A-Z]\d*|\d+[A-Z]?)\b)
    """,
    re.VERBOSE | re.IGNORECASE,
)

BARE_DIRECTION = re.compile(r"\b(above|below)\b", re.I)

# "Part 7 of the Finance Act 2004" is a statutory reference, not an internal one.
# The document carries around sixty of these; resolving them inside the schedule
# manufactures edges that point at unrelated paragraphs.
LEGISLATION = re.compile(
    r"\b(Act|Regulations?|Directive|Order|Statute|Schedule\s+\d+\s+of\s+the)\b\s*\d{0,4}", re.I
)


# --------------------------------------------------------------------------- #
# document ids
# --------------------------------------------------------------------------- #
def schedule_doc_id(text: str) -> str | None:
    m = re.match(r"\s*(Framework|Joint|Call[-\s]?Off)\s+Schedule\s+(\d+)", text, re.I)
    if not m:
        return None
    family = m.group(1).lower().replace("-", "_").replace(" ", "_")
    family = "call_off" if family.startswith("call") else family
    return f"{family}_schedule_{m.group(2)}"


# --------------------------------------------------------------------------- #
# number list parsing
# --------------------------------------------------------------------------- #
TOKEN = re.compile(rf"({NUMS})|(,|\band\b|\bto\b|\bor\b|&|-|–)", re.I)


def expand(num_text: str) -> tuple[list[str], list[str]]:
    """Expand a number list or range into individual targets.

    'Clauses 10.6.1 and 10.6.2' -> two targets.
    'Paragraphs 4.3 to 4.6'     -> four targets; ranges are expanded in full,
                                   because an unexpanded range is an edge that
                                   points at nothing.
    """
    tokens = [(m.group(1), (m.group(2) or "").lower()) for m in TOKEN.finditer(num_text)]
    numbers = [t[0] for t in tokens if t[0]]
    connectors = [t[1] for t in tokens if t[1]]
    notes: list[str] = []
    if not numbers:
        return [], notes

    out: list[str] = [numbers[0]]
    for i, nxt in enumerate(numbers[1:]):
        conn = connectors[i] if i < len(connectors) else ","
        if conn in ("to", "-", "–"):
            lo_parts, hi_parts = out[-1].split("."), nxt.split(".")
            if len(lo_parts) == len(hi_parts) and lo_parts[:-1] == hi_parts[:-1]:
                try:
                    lo, hi = int(lo_parts[-1]), int(hi_parts[-1])
                except ValueError:
                    out.append(nxt)
                    continue
                if 0 <= hi - lo <= CFG["max_range_expansion"]:
                    prefix = ".".join(lo_parts[:-1])
                    out.extend(
                        (f"{prefix}.{n}" if prefix else str(n)) for n in range(lo + 1, hi + 1)
                    )
                    notes.append(f"range {out[0]}..{nxt}")
                    continue
                notes.append(f"range too wide: {lo_parts[-1]}..{hi_parts[-1]}")
            out.append(nxt)
        else:
            out.append(nxt)
    return out, notes


# --------------------------------------------------------------------------- #
# index
# --------------------------------------------------------------------------- #
def _section_prefixes(section: str) -> list[str]:
    """'part_b.annex_1' also answers to 'part_b' and to 'annex_1'."""
    parts = section.split(".")
    out = [".".join(parts[:i]) for i in range(1, len(parts))]
    out += parts
    return [p for p in dict.fromkeys(out) if p and p != section]


class Index:
    def __init__(self, clauses: list[dict]):
        self.docs = {c["doc_id"] for c in clauses}
        self.by_doc_num: dict[tuple[str, str], list[dict]] = defaultdict(list)
        self.by_doc_sec_num: dict[tuple[str, str | None, str], dict] = {}
        self.section_first: dict[tuple[str, str], dict] = {}
        self.section_fallback: dict[tuple[str, str], dict] = {}
        self.doc_first: dict[str, dict] = {}
        for c in clauses:
            if c["is_split"] and not c["clause_id"].endswith("#p1"):
                continue
            self.doc_first.setdefault(c["doc_id"], c)
            if c["section"]:
                # A Part's guidance note is not the Part, so substantive clauses
                # anchor a section and preambles are only a fallback.
                target = (self.section_first if c["chunk_type"] != "preamble"
                          else self.section_fallback)
                target.setdefault((c["doc_id"], c["section"]), c)
                for prefix in _section_prefixes(c["section"]):
                    target.setdefault((c["doc_id"], prefix), c)
            if not c["number"]:
                continue
            self.by_doc_num[(c["doc_id"], c["number"])].append(c)
            self.by_doc_sec_num.setdefault((c["doc_id"], c["section"], c["number"]), c)

    def section(self, doc_id: str, section: str | None) -> dict | None:
        if not section:
            return None
        return (self.section_first.get((doc_id, section))
                or self.section_fallback.get((doc_id, section)))

    def resolve_number_in_section(self, doc_id: str, section: str, number: str
                                  ) -> tuple[str | None, str]:
        """Resolve a number inside an explicitly named Part or Annex."""
        exact = self.by_doc_sec_num.get((doc_id, section, number))
        if exact:
            return exact["clause_id"], "named_section_exact"
        for (doc, sec, num), clause in self.by_doc_sec_num.items():
            if doc == doc_id and num == number and sec and section in sec.split("."):
                return clause["clause_id"], "named_section_nested"
        return None, "no_such_number_in_named_section"

    def resolve_number(self, doc_id: str, number: str, source: dict) -> tuple[str | None, str]:
        """Resolve within a document, preferring the source clause's own section.

        Numbering restarts inside each Part and Annex, so 'Paragraph 3' inside
        Part B means Part B's paragraph 3, not Part A's.
        """
        hit = self.by_doc_sec_num.get((doc_id, source.get("section"), number))
        if hit:
            return hit["clause_id"], "same_section"
        candidates = self.by_doc_num.get((doc_id, number), [])
        if len(candidates) == 1:
            return candidates[0]["clause_id"], "unique_in_document"
        if not candidates:
            return None, "no_such_number"
        unsectioned = [c for c in candidates if not c["section"]]
        if len(unsectioned) == 1:
            return unsectioned[0]["clause_id"], "document_level"
        # Still ambiguous: take the nearest by page. A paragraph reference almost
        # always points inside the reader's own neighbourhood, and a flagged
        # best guess beats dropping the edge - the resolution is recorded on the
        # edge so it can be audited.
        nearest = min(candidates, key=lambda x: abs(x["page_start"] - source["page_start"]))
        return nearest["clause_id"], "nearest_by_page"


# --------------------------------------------------------------------------- #
def _enclosing_section(section: str | None, kind: str) -> str | None:
    """The component of the source's section path that `kind` names."""
    if not section:
        return None
    parts = section.split(".")
    for part in reversed(parts):
        if part.startswith(f"{kind}_"):
            return part
    return parts[0]


def scope_for(match: re.Match, source: dict) -> tuple[str, str, str] | None:
    """Return (ref_class, scope_doc_id, scope_rule) for a match, or None.

    `Annex`/`Part`/`Appendix` are matched case-sensitively: lower-case "part of"
    and "section 12" appear constantly in ordinary prose and in statutory
    references, and treating them as internal pointers manufactures hundreds of
    edges to nothing.
    """
    g = match.groupdict()
    if g["self"]:
        kind = g["selfkind"].lower()
        return ("self", source["doc_id"], f"self_{kind}")
    if g["qualified"]:
        return ("paragraph", source["doc_id"], "paragraph_within_named_section")
    if g["sched_sub"]:
        doc = schedule_doc_id(g["sched1"])
        kind = g["subkind"].lower()
        if kind == "clause":
            # "Framework Schedule 7, Clause 3" still means a Core Terms clause
            return ("clause", CORE, "clause_to_core_terms")
        return (kind, doc, "explicit_schedule")
    if g["sched_only"]:
        return ("schedule", schedule_doc_id(g["sched2"]), "schedule_to_document")
    if g["numbered"]:
        if g["kind"].lower() == "clause":
            return ("clause", CORE, "clause_to_core_terms")
        return ("paragraph", source["doc_id"], "paragraph_to_containing_schedule")
    if g["named"]:
        return (g["namedkind"].lower(), source["doc_id"], "named_within_containing_schedule")
    return None


def resolve(clauses: list[dict]) -> tuple[list[dict], list[dict], dict]:
    index = Index(clauses)
    edges: list[dict] = []
    unresolved: list[dict] = []
    class_counts: Counter[str] = Counter()
    seen: set[tuple] = set()

    for c in clauses:
        text = c["text"]
        if not text:
            continue
        for m in REFERENCE.finditer(text):
            scope = scope_for(m, c)
            if scope is None:
                continue
            ref_class, scope_doc, scope_rule = scope
            phrase = re.sub(r"\s+", " ", m.group(0)).strip()
            class_counts[ref_class] += 1

            def emit(target: str | None, reason: str, note: str | None = None) -> None:
                if target is None:
                    unresolved.append(
                        {
                            "source": c["clause_id"], "reference_text": phrase,
                            "ref_class": ref_class, "scope_doc": scope_doc,
                            "scope_rule": scope_rule, "reason": reason,
                            "page": c["page_start"],
                        }
                    )
                    return
                key = (c["clause_id"], target, phrase)
                if key in seen:
                    return
                seen.add(key)
                edges.append(
                    {
                        "type": "CROSS_REFERENCES", "source": c["clause_id"], "target": target,
                        "target_kind": "Document" if target in index.docs else "Clause",
                        "reference_text": phrase, "ref_class": ref_class,
                        "scope_rule": scope_rule, "resolution": reason,
                        "page": c["page_start"], "resolved": True,
                        **({"note": note} if note else {}),
                    }
                )

            if ref_class == "self":
                kind = m.group("selfkind").lower()
                if kind == "schedule":
                    emit(c["doc_id"], "self_schedule")
                elif kind in ("part", "annex", "appendix"):
                    # "this Part" means the whole Part, not the clause the phrase
                    # sits in. Parts have no node of their own, so it resolves to
                    # the Part's first substantive clause.
                    section = _enclosing_section(c.get("section"), kind)
                    target = index.section(c["doc_id"], section)
                    emit(target["clause_id"] if target else None,
                         f"self_{kind}_section" if target else "no_enclosing_section")
                else:
                    emit(c["clause_id"], f"self_{kind}")
                continue

            if scope_doc is None or scope_doc not in index.docs:
                emit(None, "unknown_document")
                continue

            if ref_class == "schedule":
                emit(scope_doc, "document")
                continue

            if ref_class in ("annex", "part", "appendix"):
                window = text[m.end(): m.end() + 60]
                if LEGISLATION.match(window.lstrip()) or re.match(r"^\s*of\s+the\s", window):
                    unresolved.append(
                        {
                            "source": c["clause_id"], "reference_text": phrase,
                            "ref_class": "legislation", "scope_doc": None,
                            "scope_rule": "external_legislation",
                            "reason": "external_legislation", "page": c["page_start"],
                        }
                    )
                    continue
                label = (m.group("label") or "").strip()
                if not label:
                    emit(None, "no_label")
                    continue
                sec = f"{m.group('namedkind').lower()}_{label.lower()}"
                hit = index.section(scope_doc, sec)
                emit(hit["clause_id"] if hit else None,
                     "section_first_clause" if hit else "no_such_section")
                continue

            if m.group("qualified"):
                raw = m.group("qnums")
                section = f"{m.group('qsectype').lower()}_{m.group('qseclabel').lower()}"
            else:
                raw = m.group("subnums") if m.group("sched_sub") else m.group("nums")
                section = None
            targets, notes = expand(raw)
            if not targets:
                emit(None, "no_numbers")
                continue
            for number in targets:
                if section:
                    target, reason = index.resolve_number_in_section(scope_doc, section, number)
                else:
                    target, reason = index.resolve_number(scope_doc, number, c)
                emit(target, reason, "; ".join(notes) or None)

        for m in BARE_DIRECTION.finditer(text):
            window = text[max(0, m.start() - 40): m.start()]
            if not re.search(rf"({SCHEDULE}|Clause|Paragraph|Annex|Part|Appendix)", window, re.I):
                unresolved.append(
                    {
                        "source": c["clause_id"], "reference_text": m.group(0),
                        "ref_class": "relative", "scope_doc": c["doc_id"],
                        "scope_rule": "relative_direction", "reason": "bare_direction_word",
                        "page": c["page_start"],
                    }
                )

    total = len(edges) + len(unresolved)
    # Bare "above"/"below" with no number, and statutory references, are logged as
    # unresolvable by design; the acceptance rate is measured over references that
    # are supposed to resolve to a node in this document.
    by_design = sum(1 for u in unresolved if u["reason"] in ("bare_direction_word", "external_legislation"))
    resolvable = total - by_design
    stats = {
        "references_detected": total,
        "resolved": len(edges),
        "unresolved": len(unresolved),
        "unresolvable_by_design": by_design,
        "resolution_rate_all": round(len(edges) / max(total, 1), 4),
        "resolution_rate": round(len(edges) / max(resolvable, 1), 4),
        "by_class": dict(class_counts),
        "by_scope_rule": dict(Counter(e["scope_rule"] for e in edges)),
        "unresolved_reasons": dict(Counter(u["reason"] for u in unresolved)),
    }
    return edges, unresolved, stats


def main() -> None:
    clauses = load_jsonl(path("clauses"))
    edges, unresolved, stats = resolve(clauses)
    write_jsonl(path("edges"), edges)
    write_jsonl(path("unresolved"), unresolved)

    # regression guard on the rule that matters most
    paragraph_to_core = [
        e for e in edges
        if e["ref_class"] == "paragraph" and e["target"].startswith(f"{CORE}.")
        and e["scope_rule"] != "explicit_schedule"
    ]
    non_core_sources = [e for e in paragraph_to_core if not e["source"].startswith(f"{CORE}.")]

    stats["scope_regression_paragraph_to_core_terms"] = len(non_core_sources)
    stats["acceptance_resolution_ge_95pct"] = stats["resolution_rate"] >= 0.95
    stats["unresolved_sample"] = unresolved[:40]
    report("crossref_report.json", stats)

    print(
        f"{stats['references_detected']} references | resolved {stats['resolved']} "
        f"({stats['resolution_rate']:.1%}) | unresolved {stats['unresolved']}"
    )
    print(f"  scope regression (Paragraph N -> Core Terms from a schedule): {len(non_core_sources)}")
    print(f"  by class: {stats['by_class']}")
    print(f"  unresolved reasons: {stats['unresolved_reasons']}")


if __name__ == "__main__":
    main()
