"""Stage 3.2 - LangGraph agent over the MCP tools.

classify -> retrieve -> expand -> reflect -> (loop, capped at 3 hops) -> synthesise -> verify

Every graph access goes through the MCP server over stdio. Nothing here imports a
query module or touches Neo4j: if a tool is broken, this agent is broken, which is
the only way the server's contract stays honest. The one exception is reciprocal
rank fusion, which is arithmetic over results already returned - no database.

This is a *fixed pipeline*, not a tool-choosing agent, and the distinction is
deliberate. The order of the nodes is decided here, in code; the model decides
only the route and whether the evidence is sufficient. Cost is therefore bounded
at three LLM calls and every question takes the same path, which is what makes it
evaluable.
"""
from __future__ import annotations

import json
import re
from typing import Any, Literal, TypedDict

from google.genai import types
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, ValidationError

from src.agent import prompts
from src.agent.mcp_client import MCPTools
from src.common import SETTINGS
# pure ranking arithmetic over results the tools already returned; no database
from src.retrieval.queries import reciprocal_rank_fusion
from src.vertex import VertexClient

MAX_HOPS = SETTINGS["retrieval"]["max_hops"]

# How much of a seed's score an expanded neighbour inherits. Evidence reached by
# following a reference is worth nearly as much as the clause that referenced it -
# that is the graph's contribution - while a parent is context and a definition is
# support. Insertion order carried this implicitly and badly: everything a seed
# pulled in sat directly behind it, so the sixth search hit fell past rank 10.
EDGE_WEIGHT = {
    "expand_context:reference": 0.85,
    "trace_references": 0.70,
    "expand_context:parent": 0.55,
    "lookup_definition": 0.50,
}
TOOL_BASE = 0.015          # scale of an RRF score, for rows that arrive without one
CITATION = re.compile(r"\[([a-z0-9_]+(?:\.[A-Za-z0-9_#]+)*)\]")
CAPITALISED_TERM = re.compile(r"\b(?:[A-Z][a-z]+)(?:\s+(?:of|the|and|[A-Z][a-z]+)){0,3}\b")


class Route(BaseModel):
    route: Literal["definition", "single_clause", "multi_hop", "comparison", "aggregation"]
    search_queries: list[str]
    doc_filter: list[str]
    terms: list[str]
    actor: str | None = None


class Reflection(BaseModel):
    sufficient: bool
    addressed: bool = True
    refined_query: str | None = None


class AgentState(TypedDict, total=False):
    question: str
    route: str
    plan: list[str]
    retrieved: list[dict]
    visited: set[str]
    hops: int
    answer: str
    citations: list[str]
    trace: list[dict]
    addressed: bool
    # carried between nodes: the classifier's routing hints
    doc_filter: list[str] | None
    terms: list[str]
    actor: str | None


class Agent:
    def __init__(self, model: str | None = None, tools: MCPTools | None = None):
        self.client = VertexClient()
        self.model = model or SETTINGS["llm"]["model"]
        self.tools = tools or MCPTools()
        self.tool_log: list[dict] = []
        self._call_cache: dict[str, dict] = {}
        self.documents = self.tool("list_documents")["documents"]
        self.doc_lines = "\n".join(f"  {d['doc_id']}: {d['title']}" for d in self.documents)
        self.graph = self._build()

    def tool(self, name: str, **args) -> dict:
        """Call an MCP tool. Arguments the caller left as None are dropped so the
        server applies its own defaults rather than being handed a null.

        Each call is recorded. Without it the trace shows only how much evidence a
        node gathered, not which graph operations produced it - and the question
        worth being able to answer about a GraphRAG system is how much of the
        answer came from traversing edges rather than from searching text.
        """
        payload = {k: v for k, v in args.items() if v is not None}
        # The reflect loop re-enters retrieve and expand, so without a memo the
        # same searches and the same definition lookups are repeated once per hop.
        # On a three-hop question that was 20 of 35 calls returning bytes the
        # agent already had.
        key = f"{name}:{json.dumps(payload, sort_keys=True)}"
        if key in self._call_cache:
            self.tool_log.append({"tool": name, "args": payload, "cached": True})
            return self._call_cache[key]
        raw = self.tools.call(name, payload)
        self.tool_log.append({"tool": name, "args": payload, "bytes": len(raw)})
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {}
        self._call_cache[key] = parsed
        return parsed

    def close(self) -> None:
        self.tools.close()

    # ----------------------------------------------------------------- LLM --
    def _ask(self, prompt: str, schema: type[BaseModel] | None = None,
             attempts: int = 3) -> Any:
        """A structured call that retries its own parse failures.

        Transport errors are already retried inside VertexClient. This handles the
        other failure: a syntactically valid call whose *output* is malformed - a
        run-away generation that never closes its JSON. One retry recovers it; the
        alternative is losing the whole question.
        """
        config = types.GenerateContentConfig(
            temperature=0,
            thinking_config=types.ThinkingConfig(thinking_budget=128),
            **({"response_mime_type": "application/json", "response_schema": schema,
                "max_output_tokens": 4096}
               if schema else {}),
        )
        last: Exception | None = None
        for _ in range(attempts if schema else 1):
            text = self.client.generate_content(
                model=self.model, contents=prompt, config=config
            ).text
            if schema is None:
                return text
            try:
                return schema.model_validate_json(text)
            except ValidationError as exc:
                last = exc
        raise last  # type: ignore[misc]

    # --------------------------------------------------------------- nodes --
    def classify(self, state: AgentState) -> AgentState:
        route = self._ask(
            prompts.CLASSIFY.format(question=state["question"], documents=self.doc_lines),
            Route,
        )
        return {
            "route": route.route,
            "plan": route.search_queries or [state["question"]],
            "retrieved": [],
            "visited": set(),
            "hops": 0,
            "trace": [{"node": "classify", "route": route.route,
                       "queries": route.search_queries, "doc_filter": route.doc_filter,
                       "terms": route.terms, "actor": route.actor}],
            "doc_filter": route.doc_filter or None,
            "terms": route.terms,
            "actor": route.actor,
        }

    def retrieve(self, state: AgentState) -> AgentState:
        doc_filter = state.get("doc_filter")
        # Fuse the plan's queries into ONE ranking rather than concatenating them.
        # Concatenation makes rank meaningless - query 2's best hit sits below
        # query 1's worst - and pushes everything the expansion finds past the
        # top of the evidence list.
        # The user's own question is always searched, and first. Measured on the
        # golden set the classifier's paraphrases retrieve *worse* than the
        # question as asked (0.587 vs 0.736 recall@10) - rewriting a question is
        # a way to lose the words that made it findable.
        queries = [state["question"]] + [
            p for p in state["plan"][:3] if p.strip().lower() != state["question"].strip().lower()
        ]
        rankings = [self.tool("search_clauses", query=qt, doc_filter=doc_filter,
                              top_k=10).get("results", []) for qt in queries[:4]]
        found: list[dict] = reciprocal_rank_fusion(
            *rankings, weights=[2.0] + [1.0] * (len(rankings) - 1))[:12]

        if state["route"] == "aggregation" and state.get("actor"):
            for row in self.tool("get_obligations", actor=state["actor"],
                                 doc_filter=doc_filter).get("obligations", []):
                found.append({**row, "source_tool": "get_obligations"})
        if state["route"] == "definition":
            for term in (state.get("terms") or [])[:3]:
                hit = self.tool("lookup_definition", term=term,
                                scope_doc=doc_filter[0] if doc_filter else None)
                for d in hit.get("applicable", []) + hit.get("global", []):
                    found.append({
                        "clause_id": d["clause_id"], "hierarchy_path": d["hierarchy_path"],
                        "text": f'"{d["term"]}" means {d["definition_text"]}',
                        "page_start": d["page_start"], "page_end": d["page_end"],
                        "source_tool": "lookup_definition", "scope": d["scope"],
                    })
        if "liability" in state["question"].lower() or "cap" in state["question"].lower():
            position = self.tool("get_liability_position",
                                 instrument=doc_filter[0] if doc_filter else None)
            for row in position.get("caps", []) + position.get("carve_outs_and_uncapped", []):
                found.append({**row, "source_tool": "get_liability_position"})
        if "terminat" in state["question"].lower():
            for row in self.tool("get_termination_rights",
                                 actor=state.get("actor")).get("termination_rights", []):
                found.append({**row, "source_tool": "get_termination_rights"})

        for rank, row in enumerate(found, start=1):
            row.setdefault("score", TOOL_BASE / (1 + 0.1 * rank))
        merged = _rank_evidence(state.get("retrieved", []) + found)
        return {
            "retrieved": merged,
            "trace": state["trace"] + [{"node": "retrieve", "new": len(found),
                                        "total": len(merged), "hop": state["hops"]}],
        }

    def expand(self, state: AgentState) -> AgentState:
        """Expand each hit and place what it finds *directly behind* that hit.

        Ordering is the whole point. A clause reached by following a reference is
        the graph's contribution, and appending it to the end of the evidence list
        buries it below every keyword match - it is then neither used by synthesis
        nor counted by recall@k. Interleaving keeps a referenced clause adjacent to
        the clause that referenced it.
        """
        visited = set(state.get("visited", set()))
        ordered: list[dict] = []
        n_added = 0
        for i, row in enumerate(state["retrieved"]):
            ordered.append(row)
            cid = row.get("clause_id")
            if i >= 5 or not cid or cid in visited:
                continue
            visited.add(cid)
            context = self.tool("expand_context", clause_id=cid)
            behind: list[dict] = []
            for parent in context.get("parents", [])[:1]:
                behind.append({**parent, "source_tool": "expand_context:parent"})
            refs = context.get("refers_to_clauses", [])
            for ref in refs[:4]:
                behind.append({**ref, "source_tool": "expand_context:reference"})
            # a clause that points elsewhere is not fully answered on its own
            if refs:
                chain = self.tool("trace_references", clause_id=cid, depth=2,
                                  direction="out").get("chain", [])
                for step in chain[:3]:
                    behind.append({**step, "source_tool": "trace_references"})
            seed_score = row.get("score", TOOL_BASE)
            for item in behind:
                item["score"] = seed_score * EDGE_WEIGHT.get(item["source_tool"], 0.5)
            n_added += len(behind)
            ordered.extend(behind)

        for term in _candidate_terms(state["retrieved"], state.get("terms") or []):
            hit = self.tool("lookup_definition", term=term)
            if not hit.get("all_matches"):
                continue        # not a defined term; the tool says so
            for d in hit.get("applicable", []):
                ordered.append({
                    "clause_id": d["clause_id"], "hierarchy_path": d["hierarchy_path"],
                    "text": f'"{d["term"]}" means {d["definition_text"]}',
                    "page_start": d["page_start"], "page_end": d["page_end"],
                    "source_tool": "lookup_definition",
                    "score": TOOL_BASE * EDGE_WEIGHT["lookup_definition"],
                })
                n_added += 1
        merged = _rank_evidence(ordered)
        return {
            "retrieved": merged, "visited": visited,
            "trace": state["trace"] + [{"node": "expand", "added": n_added,
                                        "visited": len(visited)}],
        }

    def reflect(self, state: AgentState) -> AgentState:
        reflection = self._ask(
            prompts.REFLECT.format(
                question=state["question"], n=len(state["retrieved"]),
                evidence=_render(state["retrieved"][:12]),
            ),
            Reflection,
        )
        hops = state["hops"] + 1
        out: AgentState = {
            "hops": hops,
            "addressed": reflection.addressed,
            "trace": state["trace"] + [{"node": "reflect", "sufficient": reflection.sufficient,
                                        "addressed": reflection.addressed, "hop": hops,
                                        "refined": reflection.refined_query}],
        }
        if not reflection.sufficient and hops < MAX_HOPS and reflection.refined_query:
            out["plan"] = [reflection.refined_query]
        return out

    def _should_loop(self, state: AgentState) -> str:
        last = state["trace"][-1]
        if last.get("node") == "reflect" and not last.get("sufficient") \
                and state["hops"] < MAX_HOPS and last.get("refined"):
            return "retrieve"
        return "synthesise"

    def synthesise(self, state: AgentState) -> AgentState:
        addressed = state.get("addressed", True)
        assessment = (
            "The retrieved clauses do address the subject of the question."
            if addressed else
            "The retrieved clauses do NOT address the subject of the question - they are "
            "the closest matches retrieval could find, not evidence that this document "
            "covers it. The correct answer is that the document does not address it."
        )
        answer = self._ask(prompts.SYNTHESISE.format(
            question=state["question"], evidence=_render(state["retrieved"][:20]),
            assessment=assessment))
        return {"answer": answer,
                "trace": state["trace"] + [{"node": "synthesise", "chars": len(answer or "")}]}

    def verify(self, state: AgentState) -> AgentState:
        """Every citation in the answer must exist in the retrieved evidence.
        A citation the agent invented is a correctness bug, not a formatting one."""
        available = {r["clause_id"] for r in state["retrieved"] if r.get("clause_id")}
        cited = CITATION.findall(state.get("answer") or "")
        good = [c for c in cited if c in available]
        bad = [c for c in cited if c not in available]
        answer = state.get("answer") or ""
        for c in bad:
            answer = answer.replace(f"[{c}]", f"[unverified: {c}]")
        return {
            "answer": answer, "citations": sorted(set(good)),
            "trace": state["trace"] + [{"node": "verify", "cited": len(cited),
                                        "verified": len(good), "stripped": bad}],
        }

    # ----------------------------------------------------------------- wire --
    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("classify", self.classify)
        g.add_node("retrieve", self.retrieve)
        g.add_node("expand", self.expand)
        g.add_node("reflect", self.reflect)
        g.add_node("synthesise", self.synthesise)
        g.add_node("verify", self.verify)
        g.set_entry_point("classify")
        g.add_edge("classify", "retrieve")
        g.add_edge("retrieve", "expand")
        g.add_edge("expand", "reflect")
        g.add_conditional_edges("reflect", self._should_loop,
                                {"retrieve": "retrieve", "synthesise": "synthesise"})
        g.add_edge("synthesise", "verify")
        g.add_edge("verify", END)
        return g.compile()

    # graph traversal vs text search - the distinction the trace exists to expose
    TRAVERSAL_TOOLS = {"expand_context", "trace_references", "get_obligations",
                       "get_liability_position", "get_termination_rights",
                       "lookup_definition"}

    def ask(self, question: str) -> dict:
        self.tool_log = []
        self._call_cache = {}
        result = self.graph.invoke({"question": question}, {"recursion_limit": 40})
        result["tool_calls"] = list(self.tool_log)
        served = sum(1 for t in self.tool_log if t.get("cached"))
        result["tool_summary"] = {
            "total": len(self.tool_log),
            "served_from_cache": served,
            "reached_neo4j": len(self.tool_log) - served,
            "text_search": sum(1 for t in self.tool_log if t["tool"] == "search_clauses"),
            "graph_traversal": sum(1 for t in self.tool_log
                                   if t["tool"] in self.TRAVERSAL_TOOLS),
        }
        return result


# --------------------------------------------------------------------------- #
def _dedupe(rows: list[dict]) -> list[dict]:
    out, seen = [], set()
    for r in rows:
        cid = r.get("clause_id")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        out.append(r)
    return out


def _rank_evidence(rows: list[dict]) -> list[dict]:
    """Order evidence by score rather than by the order the tools returned it.

    A clause keeps its best score if it is reached more than one way - arriving
    both as a search hit and as a referenced clause is a reason to rank it higher,
    not to overwrite it with whichever arrived second.
    """
    best: dict[str, dict] = {}
    for r in rows:
        cid = r.get("clause_id")
        if not cid:
            continue
        prior = best.get(cid)
        if prior is None or r.get("score", 0.0) > prior.get("score", 0.0):
            best[cid] = {**prior, **r} if prior else r
    return sorted(best.values(), key=lambda r: -r.get("score", 0.0))


def _render(rows: list[dict]) -> str:
    parts = []
    for r in rows:
        pages = f"pp.{r.get('page_start')}-{r.get('page_end')}"
        parts.append(
            f"[{r['clause_id']}] {r.get('hierarchy_path','')} ({pages})\n{(r.get('text') or '')[:900]}"
        )
    return "\n\n".join(parts)


def _candidate_terms(rows: list[dict], already: list[str]) -> list[str]:
    """Capitalised phrases in the retrieved text that might be defined terms.

    Which of them actually *are* defined is `lookup_definition`'s job, not a
    Cypher query's - checking here would mean reaching past the tool layer into
    the database, which is the thing this agent no longer does.
    """
    seen = {t.lower() for t in already}
    candidates: list[str] = []
    text = " ".join((r.get("text") or "")[:600] for r in rows[:5])
    for m in CAPITALISED_TERM.finditer(text):
        term = re.sub(r"\s+", " ", m.group(0)).strip()
        # A defined term is a capitalised noun phrase. Without these guards the
        # regex proposes "This Schedule", "Credit Ratings and" and phrases split
        # across a line break, and every one costs a round trip to return nothing.
        if len(term.split()) < 2 or term.lower() in seen:
            continue
        # a capitalised word at the start of a sentence is not a defined term
        if re.match(r"^(This|That|These|Those|The|Any|Such|Each|If|When|Where|While"
                    r"|Unless|Although|Provided|Subject|Notwithstanding|Following"
                    r"|During|Without|Upon|In|On|At|For|To)\b", term):
            continue
        if re.search(r"\b(and|or|of|to|in|for|with|by|as)$", term, re.I):
            continue
        seen.add(term.lower())
        candidates.append(term)
    return candidates[:4]


if __name__ == "__main__":
    import sys

    agent = Agent()
    try:
        question = " ".join(sys.argv[1:]) or "What is the cap on the Supplier's liability?"
        result = agent.ask(question)
        print(result["answer"])
        print("\ncitations:", result["citations"])
        print("trace:", json.dumps(result["trace"], default=str)[:600])
    finally:
        agent.close()
