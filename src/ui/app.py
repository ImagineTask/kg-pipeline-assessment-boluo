"""A chat interface over the RM6116 agent.

Shows the answer, but also what produced it: the clauses cited, the MCP tools
called, and how many reflect-loop hops it took. For contract work the provenance
matters as much as the prose - an answer you cannot trace back to a clause is not
usable, so the citations are expandable to their source text and page range
rather than being decoration.
"""
from __future__ import annotations

import time

import streamlit as st

from src.agent.graph import Agent
from src.agent.mcp_client import MCPTools

QUESTION_BUDGET = 25          # per browser session, since the endpoint is public

EXAMPLES = [
    "What is the cap on the Supplier's liability, and what falls outside it?",
    "What is an Occasion of Tax Non-Compliance?",
    "What happens to Call-Off Contracts if the Framework Contract is terminated?",
    "What conditions apply to the Clause 34 rights referenced in Joint Schedule 7?",
    "What must the Supplier do within 5 Working Days?",
]

st.set_page_config(page_title="RM6116 GraphRAG", page_icon="§", layout="wide")


@st.cache_resource(show_spinner="Connecting to the graph…")
def get_agent() -> Agent:
    return Agent(tools=MCPTools())


@st.cache_data(ttl=600, show_spinner=False)
def graph_summary() -> dict:
    from src.retrieval import queries as q
    return q.graph_stats()


def render_evidence(result: dict) -> None:
    cited = set(result.get("citations") or [])
    evidence = {r["clause_id"]: r for r in result["retrieved"] if r.get("clause_id")}

    if cited:
        st.markdown("**Cited clauses**")
        for cid in sorted(cited):
            row = evidence.get(cid, {})
            pages = f"pp. {row.get('page_start')}–{row.get('page_end')}" if row.get("page_start") else ""
            with st.expander(f"`{cid}` · {row.get('hierarchy_path', '')[:90]}  {pages}"):
                st.write(row.get("text") or "(text not retained)")

    summary = result.get("tool_summary", {})
    with st.expander(
        f"How it got there — {summary.get('total', 0)} tool calls "
        f"({summary.get('graph_traversal', 0)} graph traversals, "
        f"{summary.get('text_search', 0)} searches), {result.get('hops', 0)} reflect hops"
    ):
        for call in result.get("tool_calls", []):
            mark = "cached" if call.get("cached") else "→ Neo4j"
            args = {k: v for k, v in call.get("args", {}).items() if k != "top_k"}
            st.code(f"{call['tool']}({args})  [{mark}]", language=None)
        st.caption(
            f"{len(evidence)} clauses gathered as evidence; {len(cited)} cited. "
            "Every citation is checked against the evidence before the answer is shown — "
            "anything the model could not support is marked `[unverified]`."
        )


def main() -> None:
    st.title("RM6116 Network Services 3 — ask the framework agreement")
    stats = graph_summary()
    st.caption(
        f"{stats['clauses']:,} clauses · {stats['documents']} documents · "
        f"{stats['definitions']} defined terms · {stats['cross_references']:,} resolved "
        f"cross-references · {stats['obligations']:,} obligations. "
        "Answers cite the clauses they come from; where the contract does not address "
        "a question, it says so."
    )

    st.session_state.setdefault("asked", 0)
    st.session_state.setdefault("history", [])

    with st.sidebar:
        st.subheader("Try one of these")
        for example in EXAMPLES:
            if st.button(example, use_container_width=True):
                st.session_state["queued"] = example
        st.divider()
        st.caption(
            f"{QUESTION_BUDGET - st.session_state['asked']} of {QUESTION_BUDGET} "
            "questions left this session. The cap exists because the endpoint is public "
            "and every question costs model calls."
        )
        st.caption("Source: RM6116, Crown Copyright 2018, published by the Crown "
                   "Commercial Service. 475 pages, 48 constituent documents.")

    for entry in st.session_state["history"]:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if entry.get("result"):
                render_evidence(entry["result"])

    question = st.chat_input("e.g. what notice must the Buyer give to terminate?")
    if not question:
        question = st.session_state.pop("queued", None)
    if not question:
        return

    if st.session_state["asked"] >= QUESTION_BUDGET:
        st.warning("Session question limit reached. Reload the page to start a new one.")
        return

    st.session_state["history"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        started = time.time()
        with st.spinner("Searching the graph, following references…"):
            try:
                result = get_agent().ask(question)
            except Exception as exc:  # noqa: BLE001
                st.error(f"The agent failed on that one: {type(exc).__name__}. "
                         "Try rephrasing, or ask something else.")
                st.caption(str(exc)[:300])
                return
        st.session_state["asked"] += 1
        answer = result.get("answer") or "_No answer was produced._"
        st.markdown(answer)
        render_evidence(result)
        st.caption(f"{time.time() - started:.1f}s")
        st.session_state["history"].append(
            {"role": "assistant", "content": answer, "result": result})


if __name__ == "__main__":
    main()
