"""Build the aggregation golden set, where ground truth is the *complete* answer.

The main golden set cannot measure aggregation. Its questions are generated from
clauses, so "all Supplier obligations with a deadline under 5 Working Days" gets
three arbitrary clauses as ground truth when thirty satisfy it — and recall
against three members of a thirty-member set measures nothing.

Here each question is defined by a Cypher predicate, so ground truth is every
clause that satisfies it, computed from the graph rather than sampled from it.
That makes set precision and recall both meaningful, and it does so without
needing community detection or a global-search stage: these are structured
filters and traversals the existing tools already perform.

Predicates are kept to answer sets the tools can actually return - the MCP layer
caps results at 25 rows for `get_obligations` and 10 elsewhere, so a 60-clause
answer is unreachable by construction and would measure the cap, not the system.
"""
from __future__ import annotations

import json

from src.common import ROOT, report, write_jsonl
from src.retrieval import queries as q

MIN_SET, MAX_SET = 3, 25

# question, the predicate that defines its complete answer, and the tool a
# competent agent should reach for
CANDIDATES = [
    ("Which provisions exclude the Supplier's liability from the cap altogether?",
     "MATCH (l:Liability_Cap)-[:STATED_IN]->(c:Clause) WHERE l.uncapped RETURN DISTINCT c.clause_id AS id",
     "get_liability_position"),
    ("List the payment terms set out in the Core Terms.",
     "MATCH (f:Financial_Term)-[:STATED_IN]->(c:Clause) WHERE c.doc_id='core_terms' RETURN DISTINCT c.clause_id AS id",
     "get_obligations"),
    ("Which clauses across the agreement cite Clause 10.4.1?",
     "MATCH (a:Clause)-[:CROSS_REFERENCES]->(:Clause {clause_id:'core_terms.10.4.1'}) RETURN DISTINCT a.clause_id AS id",
     "trace_references(direction=in)"),
    ("Which clauses refer to Clause 34, the dispute resolution provisions?",
     "MATCH (a:Clause)-[:CROSS_REFERENCES]->(b:Clause) WHERE b.clause_id STARTS WITH 'core_terms.34' RETURN DISTINCT a.clause_id AS id",
     "trace_references(direction=in)"),
    ("Which obligations must the Guarantor perform?",
     "MATCH (o:Obligation)-[:STATED_IN]->(c:Clause) WHERE o.actor='Guarantor' RETURN DISTINCT c.clause_id AS id",
     "get_obligations"),
    ("What must the Supplier do within 5 Working Days or less?",
     "MATCH (o:Obligation)-[:STATED_IN]->(c:Clause) WHERE o.actor='Supplier' AND o.working_days AND o.duration_value<=5 RETURN DISTINCT c.clause_id AS id",
     "get_obligations(max_duration_days=5)"),
    ("Which obligations fall on the Buyer under Call-Off Schedule 2 (Staff Transfer)?",
     "MATCH (o:Obligation)-[:STATED_IN]->(c:Clause) WHERE o.actor='Buyer' AND c.doc_id='call_off_schedule_2' RETURN DISTINCT c.clause_id AS id",
     "get_obligations(doc_filter)"),
    ("Which clauses in Joint Schedule 7 impose obligations on the Supplier?",
     "MATCH (o:Obligation)-[:STATED_IN]->(c:Clause) WHERE o.actor='Supplier' AND c.doc_id='joint_schedule_7' RETURN DISTINCT c.clause_id AS id",
     "get_obligations(doc_filter)"),
    ("Which provisions give the Buyer a right or remedy under the Core Terms?",
     "MATCH (r:Remedy)-[:STATED_IN]->(c:Clause) WHERE r.actor='Buyer' AND c.doc_id='core_terms' RETURN DISTINCT c.clause_id AS id",
     "get_termination_rights / search"),
    ("Which clauses in the Core Terms use the defined term 'Deliverables'?",
     "MATCH (c:Clause)-[:USES_TERM]->(t:Definition {term:'Deliverables'}) WHERE c.doc_id='core_terms' RETURN DISTINCT c.clause_id AS id",
     "lookup_definition + expand"),
    ("Which clauses does Core Terms 10.6.1 point to?",
     "MATCH (:Clause {clause_id:'core_terms.10.6.1'})-[:CROSS_REFERENCES]->(b:Clause) RETURN DISTINCT b.clause_id AS id",
     "trace_references"),
    ("Which obligations in the Core Terms carry a deadline expressed in days?",
     "MATCH (o:Obligation)-[:STATED_IN]->(c:Clause) WHERE c.doc_id='core_terms' AND o.duration_iso IS NOT NULL RETURN DISTINCT c.clause_id AS id",
     "get_obligations"),
    ("Which clauses in Call-Off Schedule 9 (Security) impose obligations with a deadline?",
     "MATCH (o:Obligation)-[:STATED_IN]->(c:Clause) WHERE c.doc_id='call_off_schedule_9' AND o.duration_iso IS NOT NULL RETURN DISTINCT c.clause_id AS id",
     "get_obligations(doc_filter)"),
    ("Which provisions state a liability cap with a monetary amount?",
     "MATCH (l:Liability_Cap)-[:STATED_IN]->(c:Clause) WHERE l.cap_amount IS NOT NULL RETURN DISTINCT c.clause_id AS id",
     "get_liability_position"),
]


def main() -> None:
    rows, skipped = [], []
    for i, (question, cypher, expected_tool) in enumerate(CANDIDATES):
        truth = sorted({r["id"] for r in q.query(cypher)})
        entry = {"id": f"agg:{i}", "type": "aggregation_complete", "question": question,
                 "ground_truth_clause_ids": truth, "answer_set_size": len(truth),
                 "predicate": cypher, "expected_tool": expected_tool, "answerable": True,
                 "reference_answer": f"The complete answer is {len(truth)} clauses, "
                                     f"defined by the predicate rather than sampled."}
        (rows if MIN_SET <= len(truth) <= MAX_SET else skipped).append(entry)

    write_jsonl(ROOT / "src" / "eval" / "golden_set_aggregation.jsonl", rows)
    report("aggregation_set_report.json", {
        "questions": len(rows),
        "skipped_out_of_range": [{"question": s["question"], "size": s["answer_set_size"]}
                                 for s in skipped],
        "answer_set_sizes": {r["question"][:60]: r["answer_set_size"] for r in rows},
        "mean_answer_set_size": round(sum(r["answer_set_size"] for r in rows) / max(len(rows), 1), 1),
        "note": ("Ground truth is every clause satisfying the predicate, so precision and "
                 "recall are set measures, not top-k. Sets outside "
                 f"{MIN_SET}-{MAX_SET} are skipped: the MCP result caps make them "
                 "unreachable by construction."),
    })
    print(f"{len(rows)} aggregation questions "
          f"(mean answer set {sum(r['answer_set_size'] for r in rows) / max(len(rows), 1):.1f} clauses); "
          f"{len(skipped)} skipped as out of range")
    for r in rows:
        print(f"  {r['answer_set_size']:>3} clauses | {r['question'][:70]}")


if __name__ == "__main__":
    main()
