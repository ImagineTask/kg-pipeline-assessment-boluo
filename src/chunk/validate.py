"""Stage 1.4 - deterministic validation (gate 1).

Runs before any LLM call. Hard checks fail the build; soft checks warn. The
numbering-continuity check is the primary page-stitching alarm: a jump from 10.2
to 10.4 almost always means 10.3 was lost at a page break.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict

from src.common import ROOT, SETTINGS, load_jsonl, path, report
from src.textutils import CLAUSE_NUMBER, ending as _ending, is_truncated

CFG = SETTINGS["chunk"]
BOILERPLATE_RESIDUE = re.compile(r"Crown Copyright|Version:\s*3\.0\.11", re.I)
TERMINAL = (".", ":", ";", "?", "!", '"', "”", ")", "]")


SPOT_CHECKS = [
    ("core_terms.2.5", "lettered limbs (a)-(d) held inside the parent clause"),
    ("core_terms.3.2.9", "sentence split across a page break"),
    ("core_terms.3.2.10", "two-digit sub-number immediately after a page break"),
    ("joint_schedule_1", "definition-style chunking, not numeric"),
    ("framework_schedule_6", "two-line wrapped heading"),
    ("framework_schedule_3", "multi-page table in Annex 1"),
    ("call_off_schedule_2", "nested Parts with Annexes"),
]


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def numbering_gaps(clauses: list[dict]) -> list[dict]:
    """Report missing siblings in each numbering sequence."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for c in clauses:
        if not c["number"] or c["is_split"]:
            continue
        parts = c["number"].split(".")
        groups[(c["doc_id"], c["section"], ".".join(parts[:-1]))].append(c)

    gaps: list[dict] = []
    for (doc_id, section, prefix), members in groups.items():
        seen = {}
        for c in members:
            try:
                seen[int(c["number"].split(".")[-1])] = c
            except ValueError:
                continue
        if not seen:
            continue
        lo, hi = min(seen), max(seen)
        if hi > 200 or hi - lo > 60:
            continue          # a stray number, not a numbering sequence
        for n in range(lo, hi + 1):
            if n in seen:
                continue
            after = seen.get(n - 1) or seen.get(n + 1)
            gaps.append(
                {
                    "doc_id": doc_id,
                    "section": section,
                    "missing": f"{prefix}.{n}" if prefix else str(n),
                    "near_page": after["page_start"] if after else None,
                    "neighbour": after["clause_id"] if after else None,
                }
            )
    return gaps


def accepted_truncations() -> dict[str, str]:
    """Reviewed exceptions to the sentence-integrity check.

    The gate stays at zero unreviewed truncations. An entry here records that a
    human compared the chunk against the source page and found no lost text.
    """
    f = ROOT / "config" / "accepted_truncations.json"
    return json.loads(f.read_text())["accepted"] if f.exists() else {}


def main() -> int:
    accepted = accepted_truncations()
    stream = load_jsonl(path("document_stream"))
    clauses = load_jsonl(path("clauses"))
    docs = {c["doc_id"] for c in clauses}
    ids = {c["clause_id"] for c in clauses}

    stream_chars = sum(len(_norm(b["text"])) for b in stream)
    chunk_chars = sum(len(_norm(c["text"])) + len(_norm(c["heading"] or "")) for c in clauses)
    coverage = chunk_chars / max(stream_chars, 1)

    duplicates = [cid for cid in ids if sum(1 for c in clauses if c["clause_id"] == cid) > 1]
    orphans = [
        c["clause_id"] for c in clauses
        if c["parent_id"] and c["parent_id"] not in ids and c["parent_id"] not in docs
    ]
    empty = [c["clause_id"] for c in clauses if not c["text"].strip() and not (c["heading"] or "").strip()]
    residue = [c["clause_id"] for c in clauses if BOILERPLATE_RESIDUE.search(c["text"])]

    # sentence integrity - the direct test for a missed cross-page join.
    # Heading-only chunks and table chunks legitimately carry no terminal mark.
    prose = [c for c in clauses if c["chunk_type"] in ("clause", "definition") and c["text"].strip()]
    # A missed cross-page join truncates a *sentence*, so the hard check targets
    # prose-length chunks. Short unterminated fragments are headings, form-field
    # labels and template placeholders; they are reported as a soft signal.
    open_ended = [c for c in prose if not _ending(c["text"]).endswith(TERMINAL)]
    unterminated = [
        {"clause_id": c["clause_id"], "page": c["page_start"], "chars": c["char_count"],
         "tail": c["text"][-70:]}
        for c in open_ended if is_truncated(c["text"])
    ]
    short_open = [c["clause_id"] for c in open_ended if not is_truncated(c["text"])]

    starts_mid_sentence = [
        c["clause_id"] for c in prose
        if re.match(r"^[a-z]", c["text"].strip()) and not c["number"] and c["chunk_type"] != "definition"
    ]

    gaps = numbering_gaps(clauses)

    short = [c["clause_id"] for c in clauses if 0 < c["char_count"] < CFG["soft_min_chars"]]
    long_ = [c["clause_id"] for c in clauses if c["char_count"] > CFG["soft_max_chars"]]
    multi_number = [
        c["clause_id"] for c in clauses
        if sum(1 for line in c["text"].splitlines() if CLAUSE_NUMBER.match(line)) > 1
    ]
    spanning = sum(1 for c in clauses if c["spans_pages"])

    by_id = {c["clause_id"]: c for c in clauses}
    spot = []
    for key, why in SPOT_CHECKS:
        hit = by_id.get(key)
        if hit is None:
            members = [c for c in clauses if c["doc_id"] == key]
            spot.append({"target": key, "why": why, "found": bool(members), "chunks": len(members)})
        else:
            spot.append(
                {
                    "target": key, "why": why, "found": True,
                    "pages": [hit["page_start"], hit["page_end"]],
                    "text": hit["text"][:220],
                }
            )

    unreviewed = [u for u in unterminated if u["clause_id"] not in accepted]
    stale = [k for k in accepted if k not in ids]

    hard = {
        "coverage_ge_99pct": coverage >= 0.99,
        "no_duplicate_clause_ids": not duplicates,
        "no_orphan_parents": not orphans,
        "no_empty_chunks": not empty,
        "no_boilerplate_residue": not residue,
        "sentence_integrity": not unreviewed,
        "no_chunk_starts_mid_sentence": not starts_mid_sentence,
    }
    soft = {
        "chunks_under_min_chars": len(short),
        "chunks_over_max_chars": len(long_),
        "chunks_with_multiple_clause_numbers": len(multi_number),
        "short_unterminated_fragments": len(short_open),
        "pct_spanning_pages": round(100 * spanning / max(len(clauses), 1), 1),
        "numbering_gaps": len(gaps),
    }

    result = {
        "chunks": len(clauses),
        "documents": len(docs),
        "coverage": round(coverage, 4),
        "hard_checks": hard,
        "hard_checks_passed": all(hard.values()),
        "soft_checks": soft,
        "duplicates": duplicates[:20],
        "orphans": orphans[:20],
        "empty": empty[:20],
        "boilerplate_residue": residue[:20],
        "unterminated_sample": unterminated[:30],
        "unterminated_count": len(unterminated),
        "unterminated_unreviewed": unreviewed[:30],
        "accepted_truncations": len(accepted),
        "stale_accepted_entries": stale,
        "starts_mid_sentence_sample": starts_mid_sentence[:20],
        "numbering_gaps_sample": gaps[:40],
        "spot_checks": spot,
    }
    report("validation_gate1.json", result)

    print(f"coverage {coverage:.2%} | chunks {len(clauses)} | docs {len(docs)}")
    for k, v in hard.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  soft: {soft}")
    print(f"  truncations: {len(unterminated)} total, {len(accepted)} reviewed+accepted, "
          f"{len(unreviewed)} unreviewed" + (f"; STALE: {stale}" if stale else ""))
    return 0 if all(hard.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
