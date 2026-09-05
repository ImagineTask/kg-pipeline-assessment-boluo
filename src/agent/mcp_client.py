"""A real MCP client, and the bridge from MCP tool schemas to Gemini declarations.

The pipeline agent in `graph.py` imports `src.retrieval.queries` directly and
never touches MCP - which makes the "MCP server -> agent" edge in the architecture
diagram a claim the code did not support. This module closes that: the server is
spawned as a subprocess, the tool list is discovered over stdio, and every call
the model makes goes through the protocol. If the server is broken, the agent is
broken, which is the point of having one.

The session is driven from a background thread so that a synchronous agent loop -
and the thread pool the evaluation runs it in - does not have to become async.
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
from typing import Any

from google.genai import types
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from src.common import ROOT

# The server inherits the full environment. A stripped one loses HOME, and with
# it the gcloud credentials the embedding call inside `search_clauses` needs.
# In the container there is no .venv; MCP_PYTHON names the interpreter to use.
SERVER = StdioServerParameters(
    command=os.environ.get("MCP_PYTHON") or str(ROOT / ".venv" / "bin" / "python"),
    args=["-m", "src.retrieval.mcp_server"],
    env={**os.environ, "PYTHONPATH": str(ROOT)},
    cwd=str(ROOT),
)


def _to_declaration(tool: Any) -> types.FunctionDeclaration:
    """MCP advertises JSON Schema; Gemini takes it directly."""
    # mcp 2.x renamed the field; accept either so the bridge is not pinned to one
    raw = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None)
    schema = dict(raw or {"type": "object", "properties": {}})
    schema.pop("$schema", None)
    schema.pop("additionalProperties", None)
    return types.FunctionDeclaration(
        name=tool.name,
        description=(tool.description or "").strip(),
        parameters_json_schema=schema,
    )


class MCPTools:
    """Synchronous facade over an MCP stdio session."""

    def __init__(self):
        self._requests: queue.Queue = queue.Queue()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self.declarations: list[types.FunctionDeclaration] = []
        self.tool_names: list[str] = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=90)
        if self._error:
            raise RuntimeError(f"MCP server failed to start: {self._error}")

    # ------------------------------------------------------------------ loop --
    def _run(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as exc:  # noqa: BLE001
            inner = getattr(exc, "exceptions", None)
            self._error = inner[0] if inner else exc
            self._ready.set()

    async def _serve(self) -> None:
        async with stdio_client(SERVER) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                self.declarations = [_to_declaration(t) for t in listed.tools]
                self.tool_names = [t.name for t in listed.tools]
                self._ready.set()
                while True:
                    item = await asyncio.get_running_loop().run_in_executor(
                        None, self._requests.get)
                    if item is None:
                        return
                    name, args, box = item
                    try:
                        result = await session.call_tool(name, args or {})
                        text = "".join(
                            getattr(c, "text", "") for c in (result.content or []))
                        box.put(text or "{}")
                    except Exception as exc:  # noqa: BLE001
                        box.put(json.dumps({"error": f"{type(exc).__name__}: {exc}"[:300]}))

    # ------------------------------------------------------------------ api ---
    def call(self, name: str, args: dict) -> str:
        box: queue.Queue = queue.Queue()
        self._requests.put((name, args, box))
        try:
            return box.get(timeout=120)
        except queue.Empty:
            return json.dumps({"error": "tool call timed out"})

    def close(self) -> None:
        self._requests.put(None)

    @property
    def gemini_tool(self) -> types.Tool:
        return types.Tool(function_declarations=self.declarations)
