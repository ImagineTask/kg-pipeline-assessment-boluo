# Cloud Run image: Streamlit UI, the agent, and the MCP server it spawns.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml ./
# install dependencies only, so a code change does not re-resolve the tree
RUN pip install --no-cache-dir \
      "google-cloud-documentai>=2.29" "google-cloud-storage>=2.16" "google-genai>=1.0" \
      "neo4j>=5.20" "langgraph>=0.2" "pydantic>=2.7" "mcp>=1.2" "tenacity>=8.3" \
      "rapidfuzz>=3.9" "pyyaml>=6" "python-dotenv>=1.0" "numpy>=1.26" "streamlit>=1.38"

COPY config/ ./config/
COPY src/ ./src/

# the agent spawns the MCP server as a subprocess with this interpreter
ENV MCP_PYTHON=/usr/local/bin/python

EXPOSE 8080
CMD exec streamlit run src/ui/app.py \
      --server.port=${PORT:-8080} --server.address=0.0.0.0 \
      --server.headless=true --browser.gatherUsageStats=false
