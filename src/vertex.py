"""A Vertex AI client that survives credential expiry.

Long stages here run for tens of minutes - the extraction pass, the embedding
pass, an evaluation over both systems. An access token minted from the gcloud CLI
expires inside that window, and the underlying client holds onto the credential it
was built with, so a run that started fine dies part-way through with
401 UNAUTHENTICATED. The client is rebuilt on an auth failure and the call retried
once, which is cheap and makes every long stage restartable in place.
"""
from __future__ import annotations

import threading
import time
from typing import Any

from google import genai

from src.common import env
from src.gcp_auth import credentials

AUTH_MARKERS = ("UNAUTHENTICATED", "401", "invalid_grant", "Reauthentication")
RATE_MARKERS = ("RESOURCE_EXHAUSTED", "429", "503", "UNAVAILABLE", "500 INTERNAL")


def new_client() -> genai.Client:
    project = env("VERTEX_PROJECT_ID", env("GCP_PROJECT_ID"))
    return genai.Client(
        vertexai=True, project=project, location=env("VERTEX_LOCATION", "global"),
        credentials=credentials(project),
    )


class VertexClient:
    """Thread-safe wrapper: shared client, rebuilt under a lock on auth failure."""

    def __init__(self):
        self._client = new_client()
        self._lock = threading.Lock()

    @property
    def raw(self) -> genai.Client:
        return self._client

    def _rebuild(self) -> None:
        with self._lock:
            self._client = new_client()

    def _call(self, method: str, attempts: int = 5, **kwargs) -> Any:
        last: Exception | None = None
        for attempt in range(attempts):
            try:
                return getattr(self._client.models, method)(**kwargs)
            except Exception as exc:  # noqa: BLE001
                last = exc
                message = str(exc)
                if any(m in message for m in AUTH_MARKERS):
                    self._rebuild()
                    continue
                if any(m in message for m in RATE_MARKERS):
                    time.sleep(min(2 ** attempt * 2, 30))
                    continue
                raise
        raise last  # type: ignore[misc]

    def generate_content(self, **kwargs) -> Any:
        return self._call("generate_content", **kwargs)

    def embed_content(self, **kwargs) -> Any:
        return self._call("embed_content", **kwargs)
