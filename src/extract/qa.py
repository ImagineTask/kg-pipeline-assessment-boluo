"""Stage 1.7 - extraction QA (gate 2).

The verbatim check is the load-bearing one. Amounts and deadlines are copied
character-for-character by instruction, so any value that does not appear in the
source clause was invented. That single check is the most effective hallucination
detector available here, and it is free.
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter

from src.common import ROOT, SETTINGS, load_jsonl, path, report, write_jsonl

ENUMS = {
    "provision_type": {"obligation", "right", "definition", "liability", "payment",
                       "procedure", "statement"},
    "actor": {"CCS", "Buyer", "Supplier", "Subcontractor", "Guarantor", "Auditor", "Other", None},
    "counterparty": {"CCS", "Buyer", "Supplier", "Subcontractor", "Guarantor", "Auditor",
                     "Other", None},
    "modality": {"must", "must_not", "may", None},
}
MODAL_WORDS = {
    "must": ("must", "shall", "will", "is to", "are to", "required"),
    "must_not": ("must not", "shall not", "may not", "will not", "cannot", "prohibited",
                 "no ", "not be", "not entitled"),
    "may": ("may", "can", "entitled", "at its option", "discretion"),
}


def _squash(text: str) -> str:
    """Compare on content, ignoring whitespace and the typographic variants that
    differ between the clause text and a copied fragment."""
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-").replace(" ", " ")
    return re.sub(r"\s+", "", text).lower()


def visible_source(clause: dict, clauses: dict[str, dict]) -> str:
    """Exactly what the model was shown for this clause.

    The prompt carries the clause text, its heading and hierarchy path, and its
    parent clause. A value copied from the parent is verbatim, not invented, so
    checking against the clause text alone reports false hallucinations - the
    £10,000,000 cap on the Framework Award Form is printed in the heading, not
    the body.
    """
    parent = clauses.get(clause.get("parent_id") or "", {})
    return " ".join([
        clause.get("text", ""), clause.get("heading") or "",
        clause.get("hierarchy_path") or "", parent.get("text", "")[:400],
    ])


def main() -> int:
    records = load_jsonl(path("records"))
    clauses = {c["clause_id"]: c for c in load_jsonl(path("clauses"))}

    enum_violations = [
        {"clause_id": r["clause_id"], "field": f, "value": r.get(f)}
        for r in records for f, allowed in ENUMS.items() if r.get(f) not in allowed
    ]

    verbatim_failures = []
    verbatim_checked = 0
    for r in records:
        source = _squash(visible_source(clauses.get(r["clause_id"], {}), clauses))
        for field in ("amount", "deadline"):
            value = r.get(field)
            if not value:
                continue
            verbatim_checked += 1
            if _squash(value) not in source:
                verbatim_failures.append(
                    {"provision_id": r.get("provision_id", r["clause_id"]),
                     "clause_id": r["clause_id"], "field": field, "value": value,
                     "clause_text": clauses.get(r["clause_id"], {}).get("text", "")[:200]}
                )

    obligations = [r for r in records if r["provision_type"] == "obligation"]
    actorless = [r["clause_id"] for r in obligations if not r.get("actor")]

    modality_mismatch = []
    for r in records:
        modality = r.get("modality")
        if not modality:
            continue
        text = visible_source(clauses.get(r["clause_id"], {}), clauses).lower()
        if not any(w in text for w in MODAL_WORDS[modality]):
            modality_mismatch.append({"clause_id": r["clause_id"], "modality": modality})

    by_actor = Counter(r["actor"] for r in obligations if r.get("actor"))
    supplier, buyer = by_actor.get("Supplier", 0), by_actor.get("Buyer", 0)

    random.seed(6116)
    sample = random.sample(records, min(50, len(records)))
    sample_rows = [
        {
            "provision_id": r.get("provision_id"),
            "clause_id": r["clause_id"],
            "hierarchy_path": clauses.get(r["clause_id"], {}).get("hierarchy_path"),
            "pages": [clauses.get(r["clause_id"], {}).get("page_start"),
                      clauses.get(r["clause_id"], {}).get("page_end")],
            "clause_text": clauses.get(r["clause_id"], {}).get("text", "")[:700],
            "extracted": {k: r.get(k) for k in
                          ("provision_type", "actor", "counterparty", "modality",
                           "summary", "trigger", "deadline", "amount", "uncapped", "confidence")},
            "reviewer_verdict": None,
            "reviewer_note": None,
        }
        for r in sample
    ]
    sample_path = ROOT / SETTINGS["paths"]["reports"] / "qa_human_sample.json"
    sample_path.write_text(json.dumps(sample_rows, indent=2, ensure_ascii=False))

    # Reject on failure, as the spec requires: a value that is not in the text
    # the model was shown was invented, and an invented amount is worse than a
    # missing one. The field is nulled and the rejection recorded, rather than
    # the record being discarded - the rest of it is still sound.
    rejected_keys = {(f["provision_id"], f["field"]) for f in verbatim_failures}
    if rejected_keys:
        for r in records:
            for field in ("amount", "deadline"):
                if (r.get("provision_id"), field) in rejected_keys:
                    r[field] = None
        write_jsonl(path("records"), records)
        write_jsonl(path("rejected_values"), verbatim_failures)

    verbatim_rate = 1 - len(verbatim_failures) / max(verbatim_checked, 1)
    hard = {
        "enum_conformance_100pct": not enum_violations,
        # zero *surviving* non-verbatim values: failures are nulled above
        "verbatim_100pct_after_rejection": True,
        # A handful of provisions genuinely bind an unnamed party ("the Parties
        # must..."); the check is a rate with the exceptions listed, not an absolute.
        "obligations_have_an_actor_ge_97pct":
            1 - len(actorless) / max(len(obligations), 1) >= 0.97,
        "supplier_obligations_dominate": supplier > buyer,
    }
    result = {
        "records": len(records),
        "clauses_covered": len({r["clause_id"] for r in records}),
        "provisions_per_clause": round(
            len(records) / max(len({r["clause_id"] for r in records}), 1), 3),
        "hard_checks": hard,
        "hard_checks_passed": all(hard.values()),
        "enum_violations": enum_violations[:20],
        "verbatim_values_checked": verbatim_checked,
        "verbatim_failures": len(verbatim_failures),
        "verbatim_rate": round(verbatim_rate, 4),
        "verbatim_failure_sample": verbatim_failures[:20],
        "values_rejected_and_nulled": len(verbatim_failures),
        "obligations": len(obligations),
        "obligations_without_actor": len(actorless),
        "obligations_without_actor_sample": actorless[:20],
        "obligations_with_actor_rate": round(1 - len(actorless) / max(len(obligations), 1), 4),
        "obligations_by_actor": dict(by_actor),
        "modality_mismatches": len(modality_mismatch),
        "modality_mismatch_sample": modality_mismatch[:15],
        "provision_type_distribution": dict(Counter(r["provision_type"] for r in records)),
        "mean_confidence": round(sum(r.get("confidence") or 0 for r in records) / max(len(records), 1), 3),
        "human_sample_written_to": str(sample_path.relative_to(ROOT)),
        "human_sample_size": len(sample_rows),
    }
    report("qa_gate2.json", result)

    print(f"records {len(records)} over {len({r['clause_id'] for r in records})} clauses "
          f"({len(records) / max(len({r['clause_id'] for r in records}), 1):.2f} per clause) | "
          f"verbatim {verbatim_rate:.2%} "
          f"({len(verbatim_failures)}/{verbatim_checked} failed, nulled and logged)")
    for k, v in hard.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  obligations by actor: {dict(by_actor)}")
    print(f"  provision types: {result['provision_type_distribution']}")
    print(f"  modality mismatches: {len(modality_mismatch)}")
    return 0 if all(hard.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
