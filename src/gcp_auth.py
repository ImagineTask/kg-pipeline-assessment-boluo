"""Google credentials, with a gcloud-CLI fallback.

Standard Application Default Credentials are used when they are present and
usable. When ADC has lapsed (a common local-dev state), fall back to minting
short-lived access tokens from the already-authenticated `gcloud` CLI rather
than dropping a long-lived service-account key on disk. Tokens are refreshed
automatically, so a multi-hour pipeline run does not die at the one-hour mark.
"""
from __future__ import annotations

import datetime as dt
import subprocess

import google.auth
import google.auth.transport.requests
from google.auth.credentials import Credentials

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
_TOKEN_TTL = dt.timedelta(minutes=45)


class GcloudCLICredentials(Credentials):
    """Credentials backed by `gcloud auth print-access-token`."""

    def __init__(self, quota_project_id: str | None = None):
        super().__init__()
        self._quota_project_id = quota_project_id

    @property
    def quota_project_id(self) -> str | None:  # adds the x-goog-user-project header
        return self._quota_project_id

    def refresh(self, request) -> None:  # noqa: ARG002 - signature fixed by the base class
        token = subprocess.check_output(
            ["gcloud", "auth", "print-access-token"], text=True
        ).strip()
        if not token:
            raise RuntimeError("gcloud returned an empty access token")
        self.token = token
        self.expiry = dt.datetime.utcnow() + _TOKEN_TTL

    def with_quota_project(self, quota_project_id):
        return GcloudCLICredentials(quota_project_id)


def credentials(project_id: str) -> Credentials:
    try:
        creds, _ = google.auth.default(scopes=SCOPES, quota_project_id=project_id)
        creds.refresh(google.auth.transport.requests.Request())
        return creds
    except Exception:
        creds = GcloudCLICredentials(quota_project_id=project_id)
        creds.refresh(None)
        return creds
