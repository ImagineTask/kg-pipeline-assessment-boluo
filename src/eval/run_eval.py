"""Run the agent over the golden set and score it.

One job: ask the agent every question in `golden_set.jsonl`, then measure

  retrieval - did the evidence it gathered contain the clauses that answer the
              question (recall@k, MRR)
  citations - does every citation resolve to a real clause and appear in the
              evidence the agent actually saw (structural, computed in code)
  answers   - is the answer faithful, supported by its citations, and does it
              answer the question; and on the negative set, does it abstain
              (judged by a model)

Retrieval and citation checks are arithmetic. Only the answer judgements need a
model, and they are marked as estimates wherever they are reported.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from google.genai import types
from pydantic import BaseModel

from src.agent.graph import CITATION, Agent
from src.agent.mcp_client import MCPTools
from src.common import ROOT, SETTINGS, load_jsonl, report, write_jsonl
from src.retrieval import queries as q
from src.vertex import VertexClient

GOLDEN = ROOT / "src" / "eval" / "golden_set.jsonl"
GOLDEN_AGG = ROOT / "src" / "eval" / "golden_set_aggregation.jsonl"
K = 10


# --------------------------------------------------------------------------- #
# metrics computed in code
# --------------------------------------------------------------------------- #
def retrieval_metrics(retrieved: list[str], truth: list[str]) -> dict:
    if not truth:
        return {}
    gt = set(truth)
    first = next((i + 1 for i, c in enumerate(retrieved) if c in gt), None)
    return {
        "recall_at_10": len({c for c in retrieved[:K] if c in gt}) / len(gt),
        "recall_at_20": len({c for c in retrieved[:20] if c in gt}) / len(gt),
        "precision_at_1": len([c for c in retrieved[:1] if c in gt]),
        "mrr": 1.0 / first if first else 0.0,
    }


def set_metrics(retrieved: list[str], truth: list[str]) -> dict:
    """Set precision and recall, for questions whose ground truth is the complete
    answer rather than a sample of it.

    Top-k is the wrong shape here: "every uncapped liability provision" has five
    correct answers and a system returning exactly those five is perfect, while
    recall@10 would call it 0.5 for the five slots it left empty.
    """
    gt, got = set(truth), set(retrieved)
    if not gt:
        return {}
    hit = len(gt & got)
    precision = hit / len(got) if got else 0.0
    recall = hit / len(gt)
    return {
        "set_precision": precision,
        "set_recall": recall,
        "set_f1": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "answer_set_size": len(gt),
        "returned": len(got),
        "found": hit,
    }


def citation_metrics(citations: list[str], retrieved: list[str], valid: set[str]) -> dict:
    """A citation must name a real clause and one the agent actually saw.
    Anything else is an invented citation, which is a correctness bug."""
    if not citations:
        return {"citation_resolve_rate": None, "citation_in_evidence_rate": None}
    seen = set(retrieved)
    return {
        "citation_resolve_rate": len([c for c in citations if c in valid]) / len(citations),
        "citation_in_evidence_rate": len([c for c in citations if c in seen]) / len(citations),
    }


# --------------------------------------------------------------------------- #
# the one thing that needs a model
# --------------------------------------------------------------------------- #
class Judgement(BaseModel):
    faithful: bool
    abstained: bool
    citations_support_claims: bool
    answers_the_question: bool
    reason: str


JUDGE = """You are grading an answer about a UK framework agreement.

Question: {question}
Reference answer: {reference}
Expected to be answerable: {answerable}

Answer under test:
{answer}

Evidence the agent had:
{evidence}

Grade strictly:
- faithful: every factual claim in the answer is supported by the evidence.
- abstained: the answer's substantive position is that this document does not
  address the question. Adding context about what it *does* cover still counts as
  abstaining; supplying a substantive answer, however hedged, does not.
- citations_support_claims: each bracketed clause_id is cited for a claim that
  clause actually supports.
- answers_the_question: the answer conveys the substance of the reference answer.
  For an unanswerable question this is true only when the agent abstained."""


def judge(client: VertexClient, item: dict, result: dict) -> dict:
    evidence = "\n\n".join(
        f"[{r['clause_id']}] {(r.get('text') or '')[:500]}" for r in result["retrieved"][:12]
    )
    try:
        response = client.generate_content(
            model=SETTINGS["llm"]["model"],
            contents=JUDGE.format(
                question=item["question"], reference=item["reference_answer"],
                answerable=item["answerable"], answer=(result["answer"] or "")[:6000],
                evidence=evidence),
            config=types.GenerateContentConfig(
                temperature=0, response_mime_type="application/json",
                response_schema=Judgement, max_output_tokens=2048,
                thinking_config=types.ThinkingConfig(thinking_budget=128)),
        )
        return Judgement.model_validate_json(response.text).model_dump()
    except Exception as exc:  # noqa: BLE001
        return {"faithful": None, "abstained": None, "citations_support_claims": None,
                "answers_the_question": None, "reason": f"judge failed: {exc}"[:200]}


# --------------------------------------------------------------------------- #
def evaluate_one(agent: Agent, client: VertexClient, item: dict, valid: set[str]) -> dict:
    try:
        result = agent.ask(item["question"])
    except Exception as exc:  # noqa: BLE001
        return {"id": item["id"], "type": item["type"],
                "error": f"{type(exc).__name__}: {exc}"[:300]}
    retrieved = [r["clause_id"] for r in result["retrieved"] if r.get("clause_id")]
    citations = result.get("citations") or []
    scorer = set_metrics if item["type"] == "aggregation_complete" else retrieval_metrics
    return {
        "id": item["id"], "type": item["type"], "question": item["question"],
        "answer": result["answer"], "retrieved": retrieved[:40], "citations": citations,
        "ground_truth": item["ground_truth_clause_ids"],
        **scorer(retrieved, item["ground_truth_clause_ids"]),
        **citation_metrics(citations, retrieved, valid),
        "judge": judge(client, item, result),
    }


def aggregate(rows: list[dict]) -> dict:
    def mean(key: str, subset: list[dict]) -> float | None:
        vals = [r[key] for r in subset if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else None

    def judged(key: str, subset: list[dict]) -> float | None:
        vals = [r["judge"][key] for r in subset
                if r.get("judge") and isinstance(r["judge"].get(key), bool)]
        return round(sum(vals) / len(vals), 4) if vals else None

    answerable = [r for r in rows if r.get("ground_truth")]
    negatives = [r for r in rows if r.get("type") == "negative"]
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)

    return {
        "questions": len(rows),
        "errors": sum(1 for r in rows if r.get("error")),
        "set_retrieval": {
            "precision": mean("set_precision", rows),
            "recall": mean("set_recall", rows),
            "f1": mean("set_f1", rows),
            "mean_answer_set_size": mean("answer_set_size", rows),
            "mean_returned": mean("returned", rows),
        } if any("set_recall" in r for r in rows) else None,
        "retrieval": {
            "recall_at_10": mean("recall_at_10", answerable),
            "recall_at_20": mean("recall_at_20", answerable),
            "precision_at_1": mean("precision_at_1", answerable),
            "mrr": mean("mrr", answerable),
        },
        "citations": {
            "resolve_to_a_real_clause": mean("citation_resolve_rate", rows),
            "present_in_the_evidence": mean("citation_in_evidence_rate", rows),
            "support_their_claim_judged": judged("citations_support_claims", rows),
        },
        "answers": {
            "answers_the_question_judged": judged("answers_the_question", rows),
            "faithfulness_judged": judged("faithful", rows),
            "abstention_on_negatives": judged("abstained", negatives),
            "false_abstention_on_answerable": judged("abstained", answerable),
        },
        "by_type": {
            t: {"n": len(sub), "recall_at_10": mean("recall_at_10", sub),
                "answers_the_question": judged("answers_the_question", sub)}
            for t, sub in by_type.items()
        },
        "targets": {"recall_at_10": 0.90, "citation_accuracy": 1.0,
                    "abstention_on_negatives": 0.90},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--set", choices=("main", "aggregation"), default="main",
                    help="main = the 80-question set; aggregation = complete-answer questions")
    args = ap.parse_args()

    golden = load_jsonl(GOLDEN_AGG if args.set == "aggregation" else GOLDEN)[: args.limit]
    valid = {r["clause_id"] for r in q.query("MATCH (c:Clause) RETURN c.clause_id AS clause_id")}
    client = VertexClient()
    tools = MCPTools()
    agent = Agent(tools=tools)

    print(f"asking the agent {len(golden)} questions", flush=True)
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate_one, agent, client, item, valid) for item in golden]
        for done, fut in enumerate(as_completed(futures), start=1):
            rows.append(fut.result())
            if done % 10 == 0:
                print(f"  {done}/{len(golden)}", flush=True)
    tools.close()

    suffix = "_aggregation" if args.set == "aggregation" else ""
    write_jsonl(ROOT / SETTINGS["paths"]["reports"] / f"eval_runs{suffix}.jsonl", rows)
    summary = aggregate(rows)
    summary = {k: v for k, v in summary.items() if v is not None}
    report(f"eval_report{suffix}.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
