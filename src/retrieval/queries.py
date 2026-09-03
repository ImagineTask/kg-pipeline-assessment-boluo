"""Parameterised graph queries.

Narrow, named queries only - there is deliberately no "run this Cypher" entry
point. A general Cypher tool returns unbounded results, is a security hole, and
gets abused by an agent that does not know the schema.

Every result carries `clause_id` and `hierarchy_path`. An answer to a contract
question without a citation is unusable.
"""
from __future__ import annotations

import functools
from typing import Any

from neo4j import GraphDatabase

from src.common import SETTINGS, env

CFG = SETTINGS["retrieval"]
MAX_ROWS = CFG["max_rows"]
MAX_CHARS = CFG["max_chars"]
DEPTH_CAP = SETTINGS["crossrefs"]["traversal_depth_cap"]


@functools.lru_cache(maxsize=1)
def _driver():
    return GraphDatabase.driver(
        env("NEO4J_URI"), auth=(env("NEO4J_USERNAME"), env("NEO4J_PASSWORD"))
    )


def query(cypher: str, **params) -> list[dict]:
    with _driver().session() as session:
        return [dict(r) for r in session.run(cypher, **params)]


def cap(rows: list[dict], max_rows: int = MAX_ROWS, max_chars: int = MAX_CHARS) -> list[dict]:
    """Bound every result set. A broad query must not be able to fill the agent's
    context with a whole schedule."""
    out, used = [], 0
    for row in rows[:max_rows]:
        row = dict(row)
        text = row.get("text") or ""
        if used + len(text) > max_chars:
            row["text"] = text[: max(0, max_chars - used)] + " […truncated]"
            row["truncated"] = True
        used += len(row.get("text") or "")
        out.append(row)
        if used >= max_chars:
            break
    return out


CLAUSE_FIELDS = """
    c.clause_id AS clause_id, c.hierarchy_path AS hierarchy_path,
    c.doc_id AS doc_id, c.number AS number, c.heading AS heading, c.text AS text,
    c.page_start AS page_start, c.page_end AS page_end, c.chunk_type AS chunk_type
"""


# --------------------------------------------------------------------------- #
def vector_search(embedding: list[float], top_k: int, doc_filter: list[str] | None) -> list[dict]:
    return query(f"""
        CALL db.index.vector.queryNodes('clause_embedding', $k, $embedding)
        YIELD node AS c, score
        WHERE $docs IS NULL OR c.doc_id IN $docs
        RETURN {CLAUSE_FIELDS}, score
        ORDER BY score DESC LIMIT $top_k
    """, embedding=embedding, k=max(top_k * 4, 25), top_k=top_k, docs=doc_filter)


def fulltext_search(text: str, top_k: int, doc_filter: list[str] | None) -> list[dict]:
    return query(f"""
        CALL db.index.fulltext.queryNodes('clause_fulltext', $q) YIELD node AS c, score
        WHERE $docs IS NULL OR c.doc_id IN $docs
        RETURN {CLAUSE_FIELDS}, score
        ORDER BY score DESC LIMIT $top_k
    """, q=text, top_k=top_k, docs=doc_filter)


def reciprocal_rank_fusion(*rankings: list[dict], k: int = 60,
                           weights: list[float] | None = None) -> list[dict]:
    """Merge rankings without needing their scores to be comparable.

    Weighted, because the two retrievers are not equally good here: measured on
    the golden set, unweighted fusion of vector and full-text scores *below*
    vector alone (0.736 vs 0.764 recall@10). Full-text still earns its place -
    it is what catches an exact defined term - but at half the weight.
    """
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = {}
    rows: dict[str, dict] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, row in enumerate(ranking, start=1):
            cid = row["clause_id"]
            scores[cid] = scores.get(cid, 0.0) + weight / (k + rank)
            rows.setdefault(cid, row)
    ordered = sorted(scores, key=lambda c: -scores[c])
    return [{**rows[c], "rrf_score": round(scores[c], 5)} for c in ordered]


# --------------------------------------------------------------------------- #
def expand_context(clause_id: str) -> dict[str, Any]:
    parents = query(f"""
        MATCH (c:Clause {{clause_id: $id}})-[:PART_OF*1..3]->(p:Clause)
        RETURN {CLAUSE_FIELDS.replace('c.', 'p.')}
    """, id=clause_id)
    siblings = query(f"""
        MATCH (c:Clause {{clause_id: $id}})-[:PART_OF]->(p)<-[:PART_OF]-(s:Clause)
        WHERE s.clause_id <> $id
        RETURN {CLAUSE_FIELDS.replace('c.', 's.')}
        ORDER BY s.number LIMIT 12
    """, id=clause_id)
    children = query(f"""
        MATCH (ch:Clause)-[:PART_OF]->(c:Clause {{clause_id: $id}})
        RETURN {CLAUSE_FIELDS.replace('c.', 'ch.')}
        ORDER BY ch.number LIMIT 12
    """, id=clause_id)
    terms = query("""
        MATCH (c:Clause {clause_id: $id})-[u:USES_TERM]->(t:Definition)
        RETURN t.term AS term, t.scope AS scope, t.defined_in AS defined_in,
               t.definition_text AS definition_text, u.occurrences AS occurrences
        ORDER BY u.occurrences DESC LIMIT 15
    """, id=clause_id)
    outbound = query("""
        MATCH (c:Clause {clause_id: $id})-[r:CROSS_REFERENCES]->(t:Clause)
        RETURN t.clause_id AS clause_id, t.hierarchy_path AS hierarchy_path,
               r.reference_text AS reference_text, r.ref_class AS ref_class,
               left(t.text, 400) AS text
        LIMIT 15
    """, id=clause_id)
    docs = query("""
        MATCH (c:Clause {clause_id: $id})-[r:REFERENCES_DOCUMENT]->(d:Document)
        RETURN d.doc_id AS doc_id, d.title AS title, r.reference_text AS reference_text
        LIMIT 10
    """, id=clause_id)
    return {
        "clause_id": clause_id,
        "parents": cap(parents, 4),
        "siblings": cap(siblings),
        "children": cap(children),
        "defined_terms": terms,
        "refers_to_clauses": cap(outbound),
        "refers_to_documents": docs,
    }


def lookup_definition(term: str, scope_doc: str | None = None) -> dict[str, Any]:
    """A local definition overrides Joint Schedule 1 inside its own document, so
    the answer depends on where the question is being asked from."""
    rows = query("""
        MATCH (t:Definition)
        WHERE toLower(t.term) = toLower($term)
           OR toLower(t.term) STARTS WITH toLower($term)
        RETURN t.term AS term, t.definition_text AS definition_text, t.scope AS scope,
               t.defined_in AS defined_in, t.clause_id AS clause_id,
               t.hierarchy_path AS hierarchy_path,
               t.page_start AS page_start, t.page_end AS page_end
        ORDER BY size(t.term) LIMIT 25
    """, term=term)
    local = [r for r in rows if scope_doc and r["defined_in"] == scope_doc]
    glob = [r for r in rows if r["scope"] == "global"]
    other_local = [r for r in rows if r["scope"] == "document_local" and r not in local]
    return {
        "term": term,
        "applicable": (local or glob or rows)[:1],
        "override_applies": bool(local),
        "global": glob[:3],
        "local_definitions_elsewhere": other_local[:8],
        "all_matches": len(rows),
    }


def get_obligations(actor: str | None, doc_filter: list[str] | None = None,
                    max_duration_days: int | None = None) -> list[dict]:
    return cap(query("""
        MATCH (o:Obligation)-[:STATED_IN]->(c:Clause)
        WHERE ($actor IS NULL OR o.actor = $actor)
          AND ($docs IS NULL OR c.doc_id IN $docs)
          AND ($maxdays IS NULL OR (o.duration_value IS NOT NULL AND o.duration_value <= $maxdays
               AND o.duration_unit IN ['day','working day','business day']))
        RETURN c.clause_id AS clause_id, c.hierarchy_path AS hierarchy_path,
               c.page_start AS page_start, c.page_end AS page_end,
               o.actor AS actor, o.counterparty AS counterparty, o.modality AS modality,
               o.summary AS summary, o.deadline AS deadline, o.duration_iso AS duration_iso,
               o.working_days AS working_days, left(c.text, 500) AS text
        ORDER BY coalesce(o.duration_value, 9999), c.clause_id
    """, actor=actor, docs=doc_filter, maxdays=max_duration_days), max_rows=25)


def get_termination_rights(actor: str | None = None, trigger: str | None = None) -> list[dict]:
    return cap(query("""
        MATCH (n)-[:STATED_IN]->(c:Clause)
        WHERE any(l IN labels(n) WHERE l IN ['Remedy','Obligation','Provision'])
          AND (toLower(n.summary) CONTAINS 'terminat' OR toLower(c.text) CONTAINS 'terminat')
          AND ($actor IS NULL OR n.actor = $actor)
          AND ($trigger IS NULL OR toLower(coalesce(n.trigger,'') + ' ' + c.text) CONTAINS toLower($trigger))
        RETURN c.clause_id AS clause_id, c.hierarchy_path AS hierarchy_path,
               c.page_start AS page_start, c.page_end AS page_end,
               n.actor AS actor, n.counterparty AS counterparty, n.trigger AS trigger,
               n.deadline AS notice_period, n.summary AS summary, left(c.text, 500) AS text
        ORDER BY c.clause_id
    """, actor=actor, trigger=trigger), max_rows=20)


def get_liability_position(instrument: str | None = None) -> dict[str, Any]:
    """Never return a cap without its carve-outs: a cap quoted alone is a wrong
    answer, because the exclusions are what decide the exposure."""
    caps = query("""
        MATCH (l:Liability_Cap)-[:STATED_IN]->(c:Clause)
        WHERE ($doc IS NULL OR c.doc_id = $doc)
        RETURN c.clause_id AS clause_id, c.hierarchy_path AS hierarchy_path,
               c.page_start AS page_start, c.page_end AS page_end,
               l.cap_amount AS cap_amount, l.amount AS amount_verbatim,
               l.uncapped AS uncapped, l.summary AS summary, left(c.text, 600) AS text
        ORDER BY l.uncapped DESC, l.cap_amount DESC
    """, doc=instrument)
    return {
        "caps": cap([r for r in caps if not r["uncapped"]], 10),
        "carve_outs_and_uncapped": cap([r for r in caps if r["uncapped"]], 15),
        "note": "A cap is only meaningful together with its carve-outs; both lists are returned.",
    }


def trace_references(clause_id: str, depth: int = 2, direction: str = "out") -> list[dict]:
    """Follow the reference chain.

    Cross-references in this document form cycles - Core Terms points at
    schedules that point back at Core Terms - so the path is required to be
    acyclic and the depth is hard-capped. Without both, this does not terminate.
    """
    depth = max(1, min(int(depth), DEPTH_CAP))
    pattern = {
        "out": "(a:Clause {clause_id: $id})-[:CROSS_REFERENCES*1..%d]->(b:Clause)",
        "in": "(b:Clause)-[:CROSS_REFERENCES*1..%d]->(a:Clause {clause_id: $id})",
        "both": "(a:Clause {clause_id: $id})-[:CROSS_REFERENCES*1..%d]-(b:Clause)",
    }[direction] % depth
    return cap(query(f"""
        MATCH p = {pattern}
        WHERE b.clause_id <> $id
          AND size(apoc.coll.toSet(nodes(p))) = size(nodes(p))   // acyclic paths only
        WITH b, min(length(p)) AS hops,
             [n IN nodes(p) | n.clause_id] AS path
        RETURN b.clause_id AS clause_id, b.hierarchy_path AS hierarchy_path,
               b.page_start AS page_start, b.page_end AS page_end,
               hops, path, left(b.text, 400) AS text
        ORDER BY hops, clause_id
    """, id=clause_id), max_rows=20)


def search_clauses_text_only(text: str, top_k: int = 5, doc_filter: list[str] | None = None):
    """Fulltext-only search, for use before embeddings exist."""
    return cap(fulltext_search(text, top_k, doc_filter), max_rows=top_k)


def graph_stats() -> dict:
    return query("""
        MATCH (c:Clause) WITH count(c) AS clauses
        MATCH (d:Document) WITH clauses, count(d) AS documents
        MATCH (t:Definition) WITH clauses, documents, count(t) AS definitions
        OPTIONAL MATCH ()-[r:CROSS_REFERENCES]->()
        WITH clauses, documents, definitions, count(r) AS cross_references
        OPTIONAL MATCH (o:Obligation)
        RETURN clauses, documents, definitions, cross_references, count(o) AS obligations
    """)[0]
