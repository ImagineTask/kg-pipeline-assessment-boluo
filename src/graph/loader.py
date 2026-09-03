"""Stage 2.2 - materialise the graph in Neo4j.

Load order is Document -> Clause -> Definition -> USES_TERM -> CROSS_REFERENCES
-> semantic nodes, so every relationship lands on nodes that already exist.
Everything is MERGE-on-key with SET for properties, so a reload is idempotent,
and every write is a parameterised UNWIND over a batch - Cypher is never built by
string concatenation.
"""
from __future__ import annotations

import json
from collections import Counter

from neo4j import GraphDatabase

from src.common import ROOT, SETTINGS, env, load_jsonl, path, report
from src.graph.normalise import parse_duration, parse_money

BATCH = 1000


def driver():
    return GraphDatabase.driver(
        env("NEO4J_URI"), auth=(env("NEO4J_USERNAME"), env("NEO4J_PASSWORD"))
    )


def run_batched(session, cypher: str, rows: list[dict]) -> int:
    for i in range(0, len(rows), BATCH):
        session.run(cypher, batch=rows[i: i + BATCH])
    return len(rows)


def apply_ddl(session) -> None:
    text = (ROOT / "src" / "graph" / "ddl.cypher").read_text()
    for stmt in [s.strip() for s in text.split(";") if s.strip() and not s.strip().startswith("//")]:
        session.run(stmt)


# --------------------------------------------------------------------------- #
def load_documents(session, docs: list[dict]) -> int:
    return run_batched(session, """
        UNWIND $batch AS row
        MERGE (d:Document {doc_id: row.doc_id})
        SET d.title = row.title, d.doc_type = row.doc_type,
            d.page_start = row.page_start, d.page_end = row.page_end,
            d.is_optional = row.is_optional, d.jurisdiction_variant = row.jurisdiction_variant
    """, docs)


def load_clauses(session, clauses: list[dict]) -> int:
    n = run_batched(session, """
        UNWIND $batch AS row
        MERGE (c:Clause {clause_id: row.clause_id})
        SET c.doc_id = row.doc_id, c.doc_type = row.doc_type, c.number = row.number,
            c.depth = row.depth, c.heading = row.heading, c.text = row.text,
            c.hierarchy_path = row.hierarchy_path, c.section = row.section,
            c.page_start = row.page_start, c.page_end = row.page_end,
            c.spans_pages = row.spans_pages, c.chunk_type = row.chunk_type,
            c.is_split = row.is_split, c.char_count = row.char_count
    """, clauses)
    run_batched(session, """
        UNWIND $batch AS row
        MATCH (c:Clause {clause_id: row.clause_id})
        MATCH (p:Clause {clause_id: row.parent_id})
        MERGE (c)-[:PART_OF]->(p)
    """, [c for c in clauses if c["parent_id"]])
    run_batched(session, """
        UNWIND $batch AS row
        MATCH (c:Clause {clause_id: row.clause_id})
        MATCH (d:Document {doc_id: row.doc_id})
        MERGE (c)-[:IN_DOCUMENT]->(d)
    """, clauses)
    # a clause whose parent is the document itself hangs off the document
    run_batched(session, """
        UNWIND $batch AS row
        MATCH (c:Clause {clause_id: row.clause_id})
        MATCH (d:Document {doc_id: row.parent_id})
        MERGE (c)-[:PART_OF]->(d)
    """, [c for c in clauses if c["parent_id"] and "." not in (c["parent_id"] or "x.")])
    return n


def load_definitions(session, defs: list[dict]) -> int:
    rows = [
        {
            **d,
            # the term alone is not unique: schedules define terms locally that
            # override Joint Schedule 1 inside their own document
            "key": f"{d['defined_in']}:{d['term'].lower()}",
            "term_lower": d["term"].lower(),
        }
        for d in defs
    ]
    n = run_batched(session, """
        UNWIND $batch AS row
        MERGE (t:Definition {key: row.key})
        SET t.term = row.term, t.term_lower = row.term_lower,
            t.definition_text = row.definition_text, t.scope = row.scope,
            t.defined_in = row.defined_in, t.clause_id = row.clause_id,
            t.hierarchy_path = row.hierarchy_path,
            t.page_start = row.page_start, t.page_end = row.page_end
    """, rows)
    run_batched(session, """
        UNWIND $batch AS row
        MATCH (t:Definition {key: row.key})
        MATCH (c:Clause {clause_id: row.clause_id})
        MERGE (t)-[:DEFINED_IN]->(c)
    """, rows)
    # Where a term was repaired against the second extraction, carry the repair
    # onto the clause so citations read correctly. The clause_id keeps its
    # original slug: it is an opaque key that downstream artefacts already use.
    repaired = [r for r in rows if r.get("heading_raw") and r["heading_raw"] != r["term"]]
    run_batched(session, """
        UNWIND $batch AS row
        MATCH (c:Clause {clause_id: row.clause_id})
        SET c.heading = row.term,
            c.heading_as_printed = row.heading_raw,
            c.hierarchy_path = replace(c.hierarchy_path, row.heading_raw, row.term),
            c.text = replace(c.text, row.heading_raw, row.term)
    """, repaired)
    return n


def load_term_edges(session, edges: list[dict]) -> int:
    session.run("MATCH ()-[r:USES_TERM]->() DELETE r")
    rows = [{**e, "key": f"{e['defined_in']}:{e['target'].lower()}"} for e in edges]
    return run_batched(session, """
        UNWIND $batch AS row
        MATCH (c:Clause {clause_id: row.source})
        MATCH (t:Definition {key: row.key})
        MERGE (c)-[r:USES_TERM]->(t)
        SET r.occurrences = row.occurrences, r.definition_scope = row.definition_scope
    """, rows)


def load_crossrefs(session, edges: list[dict]) -> int:
    # Cross-reference edges are wholly derived from edges.jsonl. MERGE alone is
    # not idempotent across a *changed* resolver: edges the previous run created
    # and this one did not would survive, and the count check would drift.
    session.run("MATCH ()-[r:CROSS_REFERENCES]->() DELETE r")
    session.run("MATCH ()-[r:REFERENCES_DOCUMENT]->() DELETE r")
    to_clause = [e for e in edges if e["target_kind"] == "Clause"]
    to_doc = [e for e in edges if e["target_kind"] == "Document"]
    run_batched(session, """
        UNWIND $batch AS row
        MATCH (a:Clause {clause_id: row.source})
        MATCH (b:Clause {clause_id: row.target})
        MERGE (a)-[r:CROSS_REFERENCES {reference_text: row.reference_text}]->(b)
        SET r.ref_class = row.ref_class, r.scope_rule = row.scope_rule,
            r.resolution = row.resolution, r.page = row.page
    """, to_clause)
    run_batched(session, """
        UNWIND $batch AS row
        MATCH (a:Clause {clause_id: row.source})
        MATCH (d:Document {doc_id: row.target})
        MERGE (a)-[r:REFERENCES_DOCUMENT {reference_text: row.reference_text}]->(d)
        SET r.ref_class = row.ref_class, r.scope_rule = row.scope_rule, r.page = row.page
    """, to_doc)
    return len(to_clause) + len(to_doc)


# --------------------------------------------------------------------------- #
# semantic layer - the graph_materialisation rules from the ontology
# --------------------------------------------------------------------------- #
def semantic_rows(records: list[dict], clauses: dict[str, dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {k: [] for k in
                                  ("obligation", "remedy", "liability", "financial", "other", "actor", "event")}
    actors: set[str] = set()
    events: set[str] = set()
    for r in records:
        cid = r["clause_id"]
        if cid not in clauses:
            continue
        base = {
            # the node key is the provision; the clause remains the citation
            "id": r.get("provision_id", cid), "clause_id": cid, "summary": r["summary"],
            "provision_index": r.get("provision_index", 1),
            "actor": r.get("actor"), "counterparty": r.get("counterparty"),
            "modality": r.get("modality"), "trigger": r.get("trigger"),
            "deadline": r.get("deadline"), "amount": r.get("amount"),
            "confidence": r.get("confidence"),
            "provision_type": r["provision_type"],
            **(parse_money(r.get("amount")) or {}),
            **(parse_duration(r.get("deadline")) or {}),
        }
        kind = {
            "obligation": "obligation", "right": "remedy",
            "liability": "liability", "payment": "financial",
        }.get(r["provision_type"], "other")
        if kind == "liability":
            base["uncapped"] = bool(r.get("uncapped"))
        out[kind].append(base)
        for who in (r.get("actor"), r.get("counterparty")):
            if who:
                actors.add(who)
        if r.get("trigger"):
            events.add(r["trigger"][:200])
    out["actor"] = [{"name": a} for a in sorted(actors)]
    out["event"] = [{"name": e} for e in sorted(events)]
    return out


SEMANTIC_CYPHER = {
    "obligation": ("""
        UNWIND $batch AS row
        MERGE (n:Obligation {id: row.id})
        SET n += row
        WITH n, row MATCH (c:Clause {clause_id: row.clause_id}) MERGE (n)-[:STATED_IN]->(c)
        WITH n, row WHERE row.actor IS NOT NULL
        MATCH (a:Actor {name: row.actor}) MERGE (a)-[:BOUND_BY]->(n)
        WITH n, row WHERE row.counterparty IS NOT NULL
        MATCH (b:Actor {name: row.counterparty}) MERGE (n)-[:OWED_TO]->(b)
    """),
    "remedy": ("""
        UNWIND $batch AS row
        MERGE (n:Remedy {id: row.id})
        SET n += row
        WITH n, row MATCH (c:Clause {clause_id: row.clause_id}) MERGE (n)-[:STATED_IN]->(c)
        WITH n, row WHERE row.actor IS NOT NULL
        MATCH (a:Actor {name: row.actor}) MERGE (a)-[:HAS_REMEDY]->(n)
        WITH n, row WHERE row.trigger IS NOT NULL
        MATCH (e:Defined_Event {name: left(row.trigger, 200)}) MERGE (n)-[:TRIGGERED_BY]->(e)
    """),
    "liability": ("""
        UNWIND $batch AS row
        MERGE (n:Liability_Cap {id: row.id})
        SET n += row, n.cap_amount = row.amount_value
        WITH n, row MATCH (c:Clause {clause_id: row.clause_id}) MERGE (n)-[:STATED_IN]->(c)
        WITH n, row WHERE row.uncapped
        MATCH (c2:Clause {clause_id: row.clause_id}) MERGE (n)-[:EXCLUDED_FROM_CAP]->(c2)
    """),
    "financial": ("""
        UNWIND $batch AS row
        MERGE (n:Financial_Term {id: row.id})
        SET n += row, n.payment_window = row.duration_iso
        WITH n, row MATCH (c:Clause {clause_id: row.clause_id}) MERGE (n)-[:STATED_IN]->(c)
        WITH n, row WHERE row.actor IS NOT NULL
        MATCH (a:Actor {name: row.actor}) MERGE (a)-[:PAYS]->(n)
    """),
    "other": ("""
        UNWIND $batch AS row
        MERGE (n:Provision {id: row.id})
        SET n += row
        WITH n, row MATCH (c:Clause {clause_id: row.clause_id}) MERGE (n)-[:STATED_IN]->(c)
    """),
}


SEMANTIC_LABELS = ["Obligation", "Remedy", "Liability_Cap", "Financial_Term", "Provision"]


def load_semantic(session, rows: dict[str, list[dict]]) -> dict[str, int]:
    # The semantic layer is wholly derived from records.jsonl, so it is rebuilt
    # rather than merged into. Merging is not enough: the node key is the
    # provision id, but the *label* comes from provision_type, so a provision
    # reclassified between runs (obligation -> right) leaves its old node behind
    # under the old label with an id that still looks current. That produced 224
    # duplicate nodes and no check caught it, because every id was still valid.
    session.run(f"""
        MATCH (n) WHERE any(l IN labels(n) WHERE l IN {SEMANTIC_LABELS})
        DETACH DELETE n""")
    run_batched(session, "UNWIND $batch AS row MERGE (:Actor {name: row.name})", rows["actor"])
    run_batched(session, "UNWIND $batch AS row MERGE (:Defined_Event {name: row.name})", rows["event"])
    counts = {}
    for kind, cypher in SEMANTIC_CYPHER.items():
        counts[kind] = run_batched(session, cypher, rows[kind])
    return counts


def prune(session, clauses: list[dict], defs: list[dict], records: list[dict]) -> dict:
    """Remove nodes that no longer exist in the source files.

    MERGE makes a reload idempotent for what is *present*; it says nothing about
    what has been removed. Re-chunking that merges two clauses leaves the old id
    behind as a node with stale text, still reachable by search and still cited.
    Deletion is scoped to the derived labels - the source files are the authority
    for every one of them.
    """
    counts = {}
    counts["clauses"] = session.run("""
        MATCH (c:Clause) WHERE NOT c.clause_id IN $keep
        DETACH DELETE c RETURN count(*) AS n
    """, keep=[c["clause_id"] for c in clauses]).single()["n"]
    counts["definitions"] = session.run("""
        MATCH (t:Definition) WHERE NOT t.key IN $keep
        DETACH DELETE t RETURN count(*) AS n
    """, keep=[f"{d['defined_in']}:{d['term'].lower()}" for d in defs]).single()["n"]
    # semantic nodes are rebuilt wholesale in load_semantic, so nothing to prune
    counts["events"] = session.run("""
        MATCH (e:Defined_Event) WHERE NOT (e)<-[:TRIGGERED_BY]-()
        DELETE e RETURN count(*) AS n""").single()["n"]
    return counts


# --------------------------------------------------------------------------- #
def main() -> None:
    docs = json.loads((ROOT / SETTINGS["paths"]["reports"] / "documents.json").read_text())
    for d in docs:
        d["is_optional"] = d["doc_id"].startswith("call_off_schedule")
        d["jurisdiction_variant"] = d["doc_id"] in (
            "call_off_schedule_17", "call_off_schedule_19", "call_off_schedule_21"
        )
    clauses = load_jsonl(path("clauses"))
    defs = load_jsonl(path("definitions"))
    term_edges = load_jsonl(path("term_edges"))
    xrefs = load_jsonl(path("edges"))
    records = load_jsonl(path("records")) if path("records").exists() else []
    by_id = {c["clause_id"]: c for c in clauses}

    with driver() as drv, drv.session() as session:
        apply_ddl(session)
        counts = {
            "documents": load_documents(session, docs),
            "clauses": load_clauses(session, clauses),
            "definitions": load_definitions(session, defs),
            "uses_term": load_term_edges(session, term_edges),
            "cross_references": load_crossrefs(session, xrefs),
        }
        counts.update(load_semantic(session, semantic_rows(records, by_id)))
        counts["pruned"] = prune(session, clauses, defs, records)
        counts["semantic_nodes"] = session.run(f"""
            MATCH (n) WHERE any(l IN labels(n) WHERE l IN {SEMANTIC_LABELS})
            RETURN count(n) AS n""").single()["n"]

        checks = {
            "clause_nodes": session.run("MATCH (c:Clause) RETURN count(c) AS n").single()["n"],
            "document_nodes": session.run("MATCH (d:Document) RETURN count(d) AS n").single()["n"],
            "definition_nodes": session.run("MATCH (t:Definition) RETURN count(t) AS n").single()["n"],
            "cross_reference_edges": session.run(
                "MATCH ()-[r:CROSS_REFERENCES]->() RETURN count(r) AS n").single()["n"],
            "reference_document_edges": session.run(
                "MATCH ()-[r:REFERENCES_DOCUMENT]->() RETURN count(r) AS n").single()["n"],
            "uses_term_edges": session.run(
                "MATCH ()-[r:USES_TERM]->() RETURN count(r) AS n").single()["n"],
            "orphan_semantic_nodes": session.run("""
                MATCH (n) WHERE any(l IN labels(n) WHERE l IN
                  ['Obligation','Remedy','Liability_Cap','Financial_Term','Provision'])
                  AND NOT (n)-[:STATED_IN]->(:Clause)
                RETURN count(n) AS n""").single()["n"],
            "clauses_without_document": session.run(
                "MATCH (c:Clause) WHERE NOT (c)-[:IN_DOCUMENT]->(:Document) RETURN count(c) AS n"
            ).single()["n"],
        }

    expected_xref = Counter(e["target_kind"] for e in xrefs)
    result = {
        "loaded": counts,
        "graph_counts": checks,
        "source_counts": {
            "clauses": len(clauses), "documents": len(docs), "definitions": len(defs),
            "cross_references_to_clause": expected_xref["Clause"],
            "cross_references_to_document": expected_xref["Document"],
            "records": len(records),
        },
        "acceptance": {
            "clause_count_matches": checks["clause_nodes"] == len(clauses),
            "crossref_count_matches": checks["cross_reference_edges"] == expected_xref["Clause"],
            "zero_orphan_semantic_nodes": checks["orphan_semantic_nodes"] == 0,
            "zero_clauses_without_document": checks["clauses_without_document"] == 0,
            "semantic_nodes_match_records": counts["semantic_nodes"] == len(records),
        },
    }
    report("graph_load_report.json", result)
    print(json.dumps(result["graph_counts"], indent=2))
    print("acceptance:", result["acceptance"])


if __name__ == "__main__":
    main()
