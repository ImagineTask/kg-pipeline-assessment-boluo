"""Stage 2.3 - clause and definition embeddings, plus the vector indexes.

Two decisions worth stating:

1. What is embedded is `hierarchy_path + "\\n" + text`, not text alone. Standard
   provisions recur near-verbatim across schedules; on text alone a vector search
   returns an arbitrary copy. The path carries the schedule name and separates
   them.
2. `embedding_model` and `embedding_version` are stored on each node, so a stale
   index is detectable rather than silently wrong.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from google.genai import types
from neo4j import GraphDatabase

from src.common import SETTINGS, env, report
from src.vertex import VertexClient

CFG = SETTINGS["embeddings"]
VERSION = "v1"


def driver():
    return GraphDatabase.driver(
        env("NEO4J_URI"), auth=(env("NEO4J_USERNAME"), env("NEO4J_PASSWORD"))
    )


class Embedder:
    def __init__(self):
        self.client = VertexClient()
        self.model = CFG["model"]

    def embed(self, texts: list[str], task: str) -> list[list[float]]:
        response = self.client.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=task, output_dimensionality=CFG["dimensions"]
            ),
        )
        return [e.values for e in response.embeddings]


VECTOR_INDEXES = [
    f"""CREATE VECTOR INDEX clause_embedding IF NOT EXISTS
        FOR (c:Clause) ON (c.embedding)
        OPTIONS {{indexConfig: {{
          `vector.dimensions`: {CFG['dimensions']},
          `vector.similarity_function`: 'cosine'
        }}}}""",
    f"""CREATE VECTOR INDEX definition_embedding IF NOT EXISTS
        FOR (d:Definition) ON (d.embedding)
        OPTIONS {{indexConfig: {{
          `vector.dimensions`: {CFG['dimensions']},
          `vector.similarity_function`: 'cosine'
        }}}}""",
]

FETCH = {
    "clause": """
        MATCH (c:Clause)
        WHERE c.embedding IS NULL AND c.text <> ''
        RETURN c.clause_id AS id, coalesce(c.hierarchy_path,'') + '\\n' + c.text AS text
        LIMIT $limit""",
    "definition": """
        MATCH (d:Definition)
        WHERE d.embedding IS NULL
        RETURN d.key AS id, d.term + ': ' + coalesce(d.definition_text,'') AS text
        LIMIT $limit""",
}
WRITE = {
    "clause": """
        UNWIND $batch AS row
        MATCH (c:Clause {clause_id: row.id})
        CALL db.create.setNodeVectorProperty(c, 'embedding', row.embedding)
        SET c.embedding_model = $model, c.embedding_version = $version""",
    "definition": """
        UNWIND $batch AS row
        MATCH (d:Definition {key: row.id})
        CALL db.create.setNodeVectorProperty(d, 'embedding', row.embedding)
        SET d.embedding_model = $model, d.embedding_version = $version""",
}


def embed_label(session, embedder: Embedder, label: str, task: str) -> int:
    total = 0
    while True:
        rows = [dict(r) for r in session.run(FETCH[label], limit=2000)]
        if not rows:
            break
        size = CFG["batch_size"]
        chunks = [rows[i: i + size] for i in range(0, len(rows), size)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(embedder.embed, [r["text"][:8000] for r in ch], task): ch
                for ch in chunks
            }
            for fut in as_completed(futures):
                ch = futures[fut]
                vectors = fut.result()
                session.run(
                    WRITE[label],
                    batch=[{"id": r["id"], "embedding": v} for r, v in zip(ch, vectors)],
                    model=embedder.model, version=VERSION,
                )
                total += len(ch)
        print(f"  {label}: {total} embedded", flush=True)
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true", help="clear existing embeddings first")
    args = ap.parse_args()

    embedder = Embedder()
    with driver() as drv, drv.session() as session:
        if args.recreate:
            session.run("MATCH (n) WHERE n.embedding IS NOT NULL REMOVE n.embedding")
        for stmt in VECTOR_INDEXES:
            session.run(stmt)
        counts = {
            "clauses": embed_label(session, embedder, "clause", "RETRIEVAL_DOCUMENT"),
            "definitions": embed_label(session, embedder, "definition", "RETRIEVAL_DOCUMENT"),
        }
        missing = session.run(
            "MATCH (c:Clause) WHERE c.embedding IS NULL AND c.text <> '' RETURN count(c) AS n"
        ).single()["n"]

    report("embeddings_report.json", {
        "model": embedder.model, "dimensions": CFG["dimensions"],
        "version": VERSION, "embedded": counts, "clauses_still_missing": missing,
    })
    print(f"embedded {counts}; still missing: {missing}")


if __name__ == "__main__":
    main()
