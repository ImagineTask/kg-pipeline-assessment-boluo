"""Stage 3.1 - MCP server over the RM6116 graph.

Narrow, parameterised tools only. There is no raw-Cypher tool: it would return
unbounded results, hand the model a way to reach anything in the database, and
is impossible to evaluate. Every tool caps its result size and returns
`clause_id` and `hierarchy_path` on every row so answers can be cited.
"""
from __future__ import annotations

import json
from typing import Any

from mcp.server.mcpserver import MCPServer

from src.common import SETTINGS
from src.graph.embeddings import Embedder
from src.retrieval import queries as q

CFG = SETTINGS["retrieval"]
mcp = MCPServer("rm6116-graphrag")
_embedder: Embedder | None = None


def embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def hybrid_search(query_text: str, doc_filter: list[str] | None, top_k: int) -> list[dict]:
    """Vector and fulltext, merged with reciprocal rank fusion.

    Neither alone is enough here: vector search confuses the near-identical
    boilerplate provisions that recur across schedules, and fulltext misses
    anything phrased differently from the contract's own wording.

    `doc_filter` is a *boost*, not a gate. Applied as a hard filter it is worse
    than useless: a caller that guesses the wrong schedule makes the answer
    unreachable, and in evaluation that cost more recall than the disambiguation
    ever won back. The filtered ranking is fused with the unfiltered one, so a
    good guess promotes the right document and a bad one merely adds noise.
    """
    unfiltered = _rank(query_text, None, top_k * 3)
    if not doc_filter:
        return q.cap(q.reciprocal_rank_fusion(*unfiltered, weights=[2.0, 1.0]), max_rows=top_k)
    filtered = _rank(query_text, doc_filter, top_k * 2)
    return q.cap(
        q.reciprocal_rank_fusion(*filtered, *unfiltered, weights=[1.0, 0.5, 2.0, 1.0]),
        max_rows=top_k,
    )


def _rank(query_text: str, doc_filter: list[str] | None, n: int) -> list[list[dict]]:
    fulltext = q.fulltext_search(_escape_lucene(query_text), n, doc_filter)
    try:
        vector = q.vector_search(embedder().embed([query_text], "RETRIEVAL_QUERY")[0],
                                 n, doc_filter)
    except Exception:                      # no vector index yet - fulltext still works
        vector = []
    return [vector, fulltext]


def _escape_lucene(text: str) -> str:
    for ch in r'+-&|!(){}[]^"~*?:\/':
        text = text.replace(ch, " ")
    return " ".join(text.split()) or "*"


# --------------------------------------------------------------------------- #
@mcp.tool()
def search_clauses(query: str, doc_filter: list[str] | None = None, top_k: int = 5) -> str:
    """Find clauses matching a question or phrase.

    Combines vector similarity and full-text search. `doc_filter` restricts to
    given doc_ids (e.g. ["core_terms", "joint_schedule_7"]) - use it when the
    question names a schedule, because standard provisions recur near-verbatim
    across schedules.
    """
    rows = hybrid_search(query, doc_filter, min(top_k, CFG["max_rows"]))
    return json.dumps({"query": query, "results": rows}, ensure_ascii=False)


@mcp.tool()
def expand_context(clause_id: str) -> str:
    """Return a clause's parents, siblings, children, defined terms used, and the
    clauses and documents it points at. Use this before answering from a single
    clause - provisions in this contract are rarely self-contained."""
    return json.dumps(q.expand_context(clause_id), ensure_ascii=False)


@mcp.tool()
def lookup_definition(term: str, scope_doc: str | None = None) -> str:
    """Look up a defined term.

    Joint Schedule 1 holds the global definitions, but schedules define terms
    locally that override it *within their own document*. Pass `scope_doc` when
    the question concerns a particular schedule, and the local definition wins
    where one exists.
    """
    return json.dumps(q.lookup_definition(term, scope_doc), ensure_ascii=False)


@mcp.tool()
def get_obligations(actor: str | None = None, doc_filter: list[str] | None = None,
                    max_duration_days: int | None = None) -> str:
    """List obligations, optionally filtered by who is bound (Supplier, Buyer,
    CCS, Subcontractor, Guarantor, Auditor), by document, and by a deadline
    shorter than `max_duration_days` (counted in days or Working Days)."""
    return json.dumps({"obligations": q.get_obligations(actor, doc_filter, max_duration_days)},
                      ensure_ascii=False)


@mcp.tool()
def get_termination_rights(actor: str | None = None, trigger: str | None = None) -> str:
    """Who may terminate, on what trigger, with what notice, and against what."""
    return json.dumps({"termination_rights": q.get_termination_rights(actor, trigger)},
                      ensure_ascii=False)


@mcp.tool()
def get_liability_position(instrument: str | None = None) -> str:
    """Liability caps together with their carve-outs and uncapped heads of loss.
    A cap quoted without its exclusions is a wrong answer, so both are returned."""
    return json.dumps(q.get_liability_position(instrument), ensure_ascii=False)


@mcp.tool()
def trace_references(clause_id: str, depth: int = 2, direction: str = "out") -> str:
    """Follow the cross-reference chain from a clause.

    `direction="out"` is what this clause points at; `direction="in"` is what
    points at it, which reveals which obligations depend on the clause you are
    reading. Depth is capped at 3 and paths are required to be acyclic - the
    references in this contract form cycles.
    """
    return json.dumps(
        {"clause_id": clause_id, "direction": direction, "depth": depth,
         "chain": q.trace_references(clause_id, depth, direction)},
        ensure_ascii=False,
    )


@mcp.tool()
def list_documents() -> str:
    """The 48 constituent documents with their page ranges - useful for building
    a `doc_filter`."""
    return json.dumps({"documents": q.query("""
        MATCH (d:Document) RETURN d.doc_id AS doc_id, d.title AS title,
        d.doc_type AS doc_type, d.page_start AS page_start, d.page_end AS page_end,
        d.jurisdiction_variant AS jurisdiction_variant
        ORDER BY d.page_start
    """)}, ensure_ascii=False)


TOOLS: dict[str, Any] = {
    "search_clauses": search_clauses,
    "expand_context": expand_context,
    "lookup_definition": lookup_definition,
    "get_obligations": get_obligations,
    "get_termination_rights": get_termination_rights,
    "get_liability_position": get_liability_position,
    "trace_references": trace_references,
    "list_documents": list_documents,
}


if __name__ == "__main__":
    mcp.run()
