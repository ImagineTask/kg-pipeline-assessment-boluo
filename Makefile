# RM6116 GraphRAG pipeline. Each target is a stage from BUILD_SPEC.md.
# Stages are gated: `make gate1` and `make gate2` fail the build on a hard check.

PY := PYTHONPATH=. .venv/bin/python

.PHONY: all setup ingest stitch chunk gate1 definitions crossrefs extract gate2 graph embed eval graph-eval ask mcp test clean-graph

all: ingest stitch chunk gate1 definitions crossrefs extract gate2 graph embed

setup:
	uv venv --python 3.12 .venv
	uv pip install --python .venv/bin/python -e ".[dev]"

# --- stage 1: data processing ---------------------------------------------
ingest:                       ## Document AI batch OCR + pdftotext cross-check
	$(PY) -m src.ingest.docai_batch
	$(PY) -m src.ingest.pdftotext_check

stitch:                       ## boilerplate strip + page-boundary reconstruction
	$(PY) -m src.ingest.boilerplate
	$(PY) -m src.ingest.stitch

chunk:                        ## structure-aware hierarchical chunking
	$(PY) -m src.chunk.structural

gate1:                        ## deterministic validation - fails the build
	$(PY) -m src.chunk.validate

definitions:                  ## definitions + defined-term linking
	$(PY) -m src.extract.definitions

crossrefs:                    ## scope-aware cross-reference resolution
	$(PY) -m src.extract.crossrefs

extract:                      ## LLM extraction against the ontology (resumable)
	$(PY) -m src.extract.llm_extract $(ARGS)

gate2:                        ## extraction QA - fails the build
	$(PY) -m src.extract.qa

# --- stage 2: graph --------------------------------------------------------
graph:                        ## DDL + load Neo4j
	$(PY) -m src.graph.loader

embed:                        ## clause + definition embeddings and vector indexes
	$(PY) -m src.graph.embeddings

# --- stage 3/4: agent and evaluation --------------------------------------
mcp:                          ## run the MCP server on stdio
	$(PY) -m src.retrieval.mcp_server

ask:                          ## ask the agent; make ask Q="what is the liability cap?"
	$(PY) -m src.agent.graph "$(Q)"

golden:                       ## build both golden sets
	$(PY) -m src.eval.build_golden_set
	$(PY) -m src.eval.build_aggregation_set

eval:                         ## score the agent on the 80-question golden set
	$(PY) -m src.eval.run_eval $(ARGS)

eval-agg:                     ## score set precision/recall on complete-answer questions
	$(PY) -m src.eval.run_eval --set aggregation $(ARGS)

graph-eval:                   ## measure the built graph: coverage, recall, precision
	$(PY) -m src.eval.eval_graph_quality $(ARGS)

test:
	$(PY) -m pytest tests/ -q

clean-graph:
	$(PY) -c "from src.graph.loader import driver; \
	  d=driver(); s=d.session(); s.run('MATCH (n) DETACH DELETE n'); print('graph cleared')"

neo4j-up:
	docker run -d --name rm6116-neo4j -p 7474:7474 -p 7687:7687 \
	  -e NEO4J_AUTH=neo4j/$${NEO4J_PASSWORD:?set NEO4J_PASSWORD in your .env} \
	  -e NEO4J_PLUGINS='["apoc"]' -e NEO4J_server_memory_heap_max__size=2G neo4j:5
