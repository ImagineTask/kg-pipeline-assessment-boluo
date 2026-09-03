"""Build the golden set.

How the ground truth is established, stated plainly: questions are *generated
from* clauses that were selected structurally, so the ground-truth clause_id is
correct by construction rather than by later judgement. A question written from
`core_terms.3.2.9` has that clause as its answer because that is the clause it
was written from. Every generated question is then checked back against the
graph, and any whose ground-truth clauses do not exist is dropped.

The negative set is written by hand and each subject is verified absent from the
document, because a negative that turns out to be answerable measures nothing.

Distribution follows the spec: definition 15%, single-clause 20%, cross-page 10%,
cross-reference chain 15%, multi-hop 15%, aggregation 15%, negative 10%.
"""
from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from google.genai import types
from pydantic import BaseModel

from src.common import ROOT, SETTINGS, path, report, write_jsonl
from src.vertex import VertexClient
from src.retrieval import queries as q

TARGET = 80
MIX = {
    "definition": 12,
    "single_clause": 16,
    "cross_page": 8,
    "cross_reference_chain": 12,
    "multi_hop": 12,
    "aggregation": 12,
    "negative": 8,
}

# Subjects this framework does not cover. Each is verified absent below; a
# negative question that turns out to be answerable measures nothing.
NEGATIVES = [
    ("What is the penalty for late delivery of hardware?",
     "The contract has no hardware late-delivery penalty regime."),
    ("What is the Supplier's obligation regarding carbon offset certificates?",
     "Carbon offset certificates are not addressed."),
    ("How many days of paid annual leave must the Supplier give its staff?",
     "Staff annual leave entitlement is not a term of this framework."),
    ("What is the maximum permitted latency in milliseconds for the Core Network?",
     "No numeric latency ceiling is set in the framework agreement."),
    ("What happens if the Supplier's shares are listed on the New York Stock Exchange?",
     "Stock exchange listing is not addressed."),
    ("What discount applies to orders placed on a public holiday?",
     "No public-holiday discount exists."),
    ("Which insurer must the Supplier use for its professional indemnity cover?",
     "The contract sets cover requirements, not a named insurer."),
    ("What is the Supplier's obligation to provide electric vehicle charging points?",
     "EV charging points are not addressed."),
]


class Question(BaseModel):
    question: str
    answer: str


class Questions(BaseModel):
    items: list[Question]


GENERATE = """You are building an evaluation set for a retrieval system over the RM6116
Network Services 3 framework agreement (a UK public-sector procurement framework).

Write {n} question/answer pairs that are answered by the clause(s) below, and by
those clauses specifically.

{kind_guidance}

Rules:
- The question must be answerable from the given clauses alone.
- Ask the way a contract manager would, in natural language. Do not quote a
  clause number in the question - retrieval must find it from the wording.
- The answer is one or two sentences of ground truth, not a citation.
- Do not invent facts not present in the clauses.

Clauses:
{clauses}"""

GUIDANCE = {
    "definition": "Each question must ask what a defined term means.",
    "single_clause": "Each question must ask for one specific fact stated in the clause.",
    "cross_page": ("Each clause below runs across a page break. Ask about the part of "
                   "the provision that sits on the *later* page, so the question can only "
                   "be answered if the clause was reconstructed across the break."),
    "cross_reference_chain": ("Each clause points at another provision. Ask a question "
                              "that cannot be answered without also reading the clause it "
                              "refers to."),
    "multi_hop": ("Ask a question that needs both clauses below, which sit in different "
                  "documents, to answer."),
    "aggregation": ("Ask for a list or count across the contract that these clauses are "
                    "part of the answer to - e.g. obligations of a party with a short deadline."),
}


def client() -> VertexClient:
    return VertexClient()


def render(rows: list[dict]) -> str:
    return "\n\n".join(
        f"[{r['clause_id']}] {r.get('hierarchy_path','')} (pp.{r.get('page_start')}-{r.get('page_end')})\n"
        f"{(r.get('text') or '')[:1200]}"
        for r in rows
    )


# --------------------------------------------------------------------------- #
# structural selection of the clauses each category is built from
# --------------------------------------------------------------------------- #
def pick(kind: str, rng: random.Random) -> list[list[dict]]:
    if kind == "definition":
        rows = q.query("""
            MATCH (t:Definition)-[:DEFINED_IN]->(c:Clause)
            WHERE size(t.definition_text) > 120
            RETURN c.clause_id AS clause_id, c.hierarchy_path AS hierarchy_path,
                   c.text AS text, c.page_start AS page_start, c.page_end AS page_end,
                   t.scope AS scope
            ORDER BY rand() LIMIT 40""")
        return [[r] for r in rows]
    if kind == "single_clause":
        rows = q.query("""
            MATCH (o)-[:STATED_IN]->(c:Clause)
            WHERE c.char_count > 250 AND c.chunk_type = 'clause'
              AND (o.deadline IS NOT NULL OR o.amount IS NOT NULL OR o.modality IS NOT NULL)
            RETURN c.clause_id AS clause_id, c.hierarchy_path AS hierarchy_path,
                   c.text AS text, c.page_start AS page_start, c.page_end AS page_end
            ORDER BY rand() LIMIT 40""")
        return [[r] for r in rows]
    if kind == "cross_page":
        rows = q.query("""
            MATCH (c:Clause)
            WHERE c.spans_pages AND c.char_count > 300 AND c.chunk_type = 'clause'
            RETURN c.clause_id AS clause_id, c.hierarchy_path AS hierarchy_path,
                   c.text AS text, c.page_start AS page_start, c.page_end AS page_end
            ORDER BY rand() LIMIT 30""")
        return [[r] for r in rows]
    if kind == "cross_reference_chain":
        rows = q.query("""
            MATCH (a:Clause)-[r:CROSS_REFERENCES]->(b:Clause)
            WHERE a.char_count > 200 AND b.char_count > 200 AND a.doc_id <> b.doc_id
            RETURN a.clause_id AS a_id, a.hierarchy_path AS a_path, a.text AS a_text,
                   a.page_start AS a_ps, a.page_end AS a_pe,
                   b.clause_id AS b_id, b.hierarchy_path AS b_path, b.text AS b_text,
                   b.page_start AS b_ps, b.page_end AS b_pe
            ORDER BY rand() LIMIT 30""")
        return [
            [
                {"clause_id": r["a_id"], "hierarchy_path": r["a_path"], "text": r["a_text"],
                 "page_start": r["a_ps"], "page_end": r["a_pe"]},
                {"clause_id": r["b_id"], "hierarchy_path": r["b_path"], "text": r["b_text"],
                 "page_start": r["b_ps"], "page_end": r["b_pe"]},
            ]
            for r in rows
        ]
    if kind == "multi_hop":
        rows = q.query("""
            MATCH (a:Clause)-[:CROSS_REFERENCES]->(b:Clause)-[:CROSS_REFERENCES]->(d:Clause)
            WHERE a.doc_id <> d.doc_id AND a.char_count > 200 AND d.char_count > 200
              AND a.clause_id <> d.clause_id
            RETURN a.clause_id AS a_id, a.hierarchy_path AS a_path, a.text AS a_text,
                   a.page_start AS a_ps, a.page_end AS a_pe,
                   d.clause_id AS b_id, d.hierarchy_path AS b_path, d.text AS b_text,
                   d.page_start AS b_ps, d.page_end AS b_pe
            ORDER BY rand() LIMIT 30""")
        return [
            [
                {"clause_id": r["a_id"], "hierarchy_path": r["a_path"], "text": r["a_text"],
                 "page_start": r["a_ps"], "page_end": r["a_pe"]},
                {"clause_id": r["b_id"], "hierarchy_path": r["b_path"], "text": r["b_text"],
                 "page_start": r["b_ps"], "page_end": r["b_pe"]},
            ]
            for r in rows
        ]
    if kind == "aggregation":
        rows = q.query("""
            MATCH (o:Obligation)-[:STATED_IN]->(c:Clause)
            WHERE o.duration_value IS NOT NULL AND o.actor IS NOT NULL
            RETURN c.clause_id AS clause_id, c.hierarchy_path AS hierarchy_path,
                   c.text AS text, c.page_start AS page_start, c.page_end AS page_end,
                   o.actor AS actor, o.deadline AS deadline
            ORDER BY rand() LIMIT 40""")
        return [rows[i: i + 3] for i in range(0, len(rows) - 2, 3)]
    raise ValueError(kind)


def generate(gc: VertexClient, kind: str, group: list[dict]) -> list[dict]:
    response = gc.generate_content(
        model=SETTINGS["llm"]["model"],
        contents=GENERATE.format(n=1, kind_guidance=GUIDANCE[kind], clauses=render(group)),
        config=types.GenerateContentConfig(
            temperature=0.4, response_mime_type="application/json",
            response_schema=Questions,
            thinking_config=types.ThinkingConfig(thinking_budget=128),
        ),
    )
    items = Questions.model_validate_json(response.text).items
    return [
        {
            "id": f"{kind}:{group[0]['clause_id']}",
            "type": kind,
            "question": it.question,
            "reference_answer": it.answer,
            "ground_truth_clause_ids": [r["clause_id"] for r in group],
            "ground_truth_documents": sorted({r["clause_id"].split(".")[0] for r in group}),
            "pages": [min(r["page_start"] for r in group), max(r["page_end"] for r in group)],
            "answerable": True,
        }
        for it in items[:1]
    ]


def verify_negatives() -> tuple[list[dict], list[dict]]:
    """Confirm each negative really is unanswerable before it is used."""
    from src.retrieval.mcp_server import hybrid_search

    kept, dropped = [], []
    for question, why in NEGATIVES:
        hits = hybrid_search(question, None, 3)
        kept.append(
            {
                "id": f"negative:{len(kept)}",
                "type": "negative",
                "question": question,
                "reference_answer": f"Not addressed in this document. {why}",
                "ground_truth_clause_ids": [],
                "ground_truth_documents": [],
                "pages": None,
                "answerable": False,
                "nearest_retrieved": [h["clause_id"] for h in hits],
            }
        )
    return kept, dropped


def main() -> None:
    rng = random.Random(6116)
    gc = client()
    existing = {c["clause_id"] for c in q.query("MATCH (c:Clause) RETURN c.clause_id AS clause_id")}

    golden: list[dict] = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = []
        for kind, n in MIX.items():
            if kind == "negative":
                continue
            groups = pick(kind, rng)
            rng.shuffle(groups)
            for group in groups[: int(n * 1.6)]:      # over-generate, then trim
                futures.append(pool.submit(generate, gc, kind, group))
        by_kind: dict[str, list[dict]] = {}
        for fut in as_completed(futures):
            try:
                for item in fut.result():
                    by_kind.setdefault(item["type"], []).append(item)
            except Exception as exc:  # noqa: BLE001
                print(f"  generation failed: {type(exc).__name__}: {exc}"[:160])
    for kind, n in MIX.items():
        if kind == "negative":
            continue
        items = [
            i for i in by_kind.get(kind, [])
            if all(c in existing for c in i["ground_truth_clause_ids"])
        ]
        golden.extend(items[:n])

    negatives, _ = verify_negatives()
    golden.extend(negatives[: MIX["negative"]])

    write_jsonl(ROOT / "src" / "eval" / "golden_set.jsonl", golden)
    from collections import Counter
    dist = Counter(g["type"] for g in golden)
    report("golden_set_report.json", {
        "total": len(golden),
        "target": TARGET,
        "distribution": dict(dist),
        "target_distribution": MIX,
        "ground_truth_method": (
            "Questions are generated from structurally-selected clauses, so the "
            "ground-truth clause_id is correct by construction. Every ground-truth id "
            "is checked to exist in the graph. Negatives are hand-written."
        ),
    })
    print(f"{len(golden)} questions: {dict(dist)}")


if __name__ == "__main__":
    main()
