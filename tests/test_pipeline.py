"""Regression tests over the built artefacts.

These run against the real pipeline output rather than fixtures, because the
failures they guard against are all data-shaped: a stitcher that stops joining,
a resolver that forgets scope. Run the pipeline first.
"""
from __future__ import annotations

import json

import pytest

from src.common import load_jsonl, path
from src.extract.crossrefs import expand
from src.ingest.boilerplate import shape
from src.textutils import is_truncated


@pytest.fixture(scope="session")
def clauses() -> dict[str, dict]:
    return {c["clause_id"]: c for c in load_jsonl(path("clauses"))}


@pytest.fixture(scope="session")
def edges() -> list[dict]:
    return load_jsonl(path("edges"))


# --------------------------------------------------------------------------- #
# page-boundary reconstruction
# --------------------------------------------------------------------------- #
def test_clause_3_2_9_reconstructs_across_its_page_break(clauses):
    """The spec's designated stitching test. Clause 3.2.9 ends on page 3 with
    '...the Buyer needs to make use of the' and continues on page 4 with
    'Goods.' after the stripped header."""
    c = clauses["core_terms.3.2.9"]
    assert c["spans_pages"] is True
    assert (c["page_start"], c["page_end"]) == (3, 4)
    assert c["text"].rstrip().endswith("make use of the Goods.")


def test_clause_3_2_10_survives_the_page_break(clauses):
    """A two-digit sub-number immediately after a page break must not be read as
    a continuation of 3.2.9."""
    c = clauses["core_terms.3.2.10"]
    assert c["number"] == "3.2.10"
    assert c["text"].startswith("The Supplier must indemnify the Buyer")


def test_lettered_limbs_stay_inside_their_parent_clause(clauses):
    c = clauses["core_terms.2.5"]
    for limb in ("(a)", "(b)", "(c)", "(d)"):
        assert limb in c["text"], f"limb {limb} missing from core_terms.2.5"


def test_clause_10_6_1_keeps_the_continuation_that_starts_with_a_number(clauses):
    """The clause continues onto a line beginning '20.2 or a Contract expires...'
    - a cross-reference, not a new clause. Truncation beats structure."""
    assert "a Contract expires" in clauses["core_terms.10.6.1"]["text"]


def test_no_boilerplate_survives_into_chunks(clauses):
    offenders = [k for k, c in clauses.items() if "Crown Copyright" in c["text"]]
    assert not offenders, offenders[:5]


# --------------------------------------------------------------------------- #
# document segmentation
# --------------------------------------------------------------------------- #
def test_all_forty_eight_documents_are_found(clauses):
    expected = (
        {"core_terms", "framework_award_form"}
        | {f"framework_schedule_{i}" for i in range(1, 10)}
        | {f"joint_schedule_{i}" for i in range(1, 13)}
        | {f"call_off_schedule_{i}" for i in range(1, 26)}
    )
    assert {c["doc_id"] for c in clauses.values()} == expected


def test_wrapped_two_line_heading_is_recovered():
    docs = {d["doc_id"]: d for d in json.loads((path("clauses").parent.parent / "reports" / "documents.json").read_text())}
    assert "Order Form Template and Call-Off Schedules" in docs["framework_schedule_6"]["title"]


# --------------------------------------------------------------------------- #
# cross-reference scope
# --------------------------------------------------------------------------- #
def test_clause_references_always_resolve_to_core_terms(edges):
    wrong = [
        e for e in edges
        if e["ref_class"] == "clause" and not e["target"].startswith("core_terms")
    ]
    assert not wrong, wrong[:3]


def test_paragraph_references_never_leak_into_core_terms(edges):
    """The scope-rule regression test. `Paragraph N` written inside a schedule
    must resolve inside that schedule, never into Core Terms."""
    leaked = [
        e for e in edges
        if e["ref_class"] == "paragraph"
        and e["target"].startswith("core_terms.")
        and not e["source"].startswith("core_terms.")
    ]
    assert not leaked, leaked[:3]


def test_same_clause_reference_resolves_identically_from_every_schedule(edges):
    hits = [e for e in edges if e["reference_text"] == "Clause 10.4.1"]
    assert len(hits) >= 10
    assert {e["target"] for e in hits} == {"core_terms.10.4.1"}
    assert len({e["source"].split(".")[0] for e in hits}) > 1


def test_ranges_expand_in_full():
    assert expand("4.3 to 4.6")[0] == ["4.3", "4.4", "4.5", "4.6"]
    assert expand("27 to 32")[0] == ["27", "28", "29", "30", "31", "32"]
    assert expand("10.6.1 and 10.6.2")[0] == ["10.6.1", "10.6.2"]


def test_every_edge_points_at_something_that_exists(clauses, edges):
    docs = {c["doc_id"] for c in clauses.values()}
    dangling = [
        e for e in edges
        if e["target"] not in clauses and e["target"] not in docs
    ]
    assert not dangling, dangling[:3]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def test_shape_collapses_digits_so_page_numbers_share_a_key():
    assert shape("Version: 3.0.11") == shape("Version: 9.9.99")
    # digits collapse position-for-position, so page numbers of the same width
    # share a key; differing widths form separate patterns, which the per-cluster
    # density check then accepts independently (pages 1-9 and 10-22 of Core Terms)
    assert shape("7") == shape("4") == "#"
    assert shape("12") == shape("22") == "##"


def test_truncation_detector_catches_the_3_2_9_shape():
    assert is_truncated("the Buyer needs to make use of the")
    assert not is_truncated("make use of the Goods.")
    assert not is_truncated("a valid invoice; and")   # a complete list limb


# --------------------------------------------------------------------------- #
# graph and retrieval - require a loaded Neo4j
# --------------------------------------------------------------------------- #
neo4j = pytest.importorskip("neo4j")


@pytest.fixture(scope="session")
def graph():
    from src.retrieval import queries as gq
    try:
        gq.graph_stats()
    except Exception as exc:                     # noqa: BLE001
        pytest.skip(f"Neo4j not available: {exc}")
    return gq


def test_graph_holds_every_clause_and_document(graph, clauses):
    stats = graph.graph_stats()
    assert stats["clauses"] == len(clauses)
    assert stats["documents"] == 48


def test_no_semantic_node_is_orphaned(graph):
    orphans = graph.query("""
        MATCH (n) WHERE any(l IN labels(n) WHERE l IN
          ['Obligation','Remedy','Liability_Cap','Financial_Term','Provision'])
          AND NOT (n)-[:STATED_IN]->(:Clause)
        RETURN count(n) AS n""")[0]["n"]
    assert orphans == 0


def test_reference_traversal_terminates_on_a_cycle(graph):
    """Core Terms 15.1 and 15.2 point at each other. Without the acyclic-path
    filter and the depth cap this query does not come back."""
    cycles = graph.query("""
        MATCH p=(a:Clause)-[:CROSS_REFERENCES*2..3]->(a)
        RETURN [n IN nodes(p) | n.clause_id] AS cycle LIMIT 1""")
    assert cycles, "expected at least one reference cycle in this document"
    start = cycles[0]["cycle"][0]
    for depth in (1, 2, 3):
        assert isinstance(graph.trace_references(start, depth, "both"), list)


def test_traversal_depth_is_hard_capped(graph):
    from src.common import SETTINGS
    cap = SETTINGS["crossrefs"]["traversal_depth_cap"]
    deep = graph.trace_references("core_terms.10.6.1", 99, "out")
    capped = graph.trace_references("core_terms.10.6.1", cap, "out")
    assert len(deep) == len(capped)


def test_local_definitions_override_the_global_one(graph):
    """A schedule that defines a term locally must win inside its own document."""
    overridden = graph.query("""
        MATCH (local:Definition {scope:'document_local'}), (g:Definition {scope:'global'})
        WHERE local.term_lower = g.term_lower
        RETURN local.term AS term, local.defined_in AS doc LIMIT 1""")
    assert overridden, "expected at least one local override of a global term"
    term, doc = overridden[0]["term"], overridden[0]["doc"]
    result = graph.lookup_definition(term, doc)
    assert result["override_applies"] is True
    assert result["applicable"][0]["defined_in"] == doc


def test_every_result_row_carries_a_citation(graph):
    for row in graph.search_clauses_text_only("termination", 5):
        assert row["clause_id"] and row["hierarchy_path"] is not None
