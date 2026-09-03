"""Shared helpers: settings, paths, jsonl IO."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def settings() -> dict[str, Any]:
    with open(ROOT / "config" / "settings.yaml") as fh:
        return yaml.safe_load(fh)


SETTINGS = settings()


def path(key: str) -> Path:
    """Resolve a logical path from settings.paths to an absolute path."""
    p = ROOT / SETTINGS["paths"][key]
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def env(key: str, default: str | None = None) -> str:
    val = os.environ.get(key, default)
    if val is None:
        raise RuntimeError(f"missing environment variable {key}")
    return val


def write_jsonl(dest: Path, rows: Iterable[dict]) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(dest, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(src: Path) -> Iterator[dict]:
    with open(src) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl(src: Path) -> list[dict]:
    return list(read_jsonl(src))


def write_json(dest: Path, obj: Any) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def report(name: str, obj: Any) -> Path:
    dest = ROOT / SETTINGS["paths"]["reports"] / name
    write_json(dest, obj)
    return dest
