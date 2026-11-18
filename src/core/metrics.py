from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from .config import project_root, resolve_path


class MetricsEvent(BaseModel):
    timestamp: str
    request_id: str
    provider: str
    model: str
    prompt_version: str
    status: str
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    latency_seconds: float = Field(ge=0)
    cost_usd: float = Field(ge=0)
    currency: str = "USD"
    tokens_source: str = "api"
    error_type: str | None = None


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class MetricsLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = resolve_path(path, project_root())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._lock = threading.Lock()

    def log(self, event: MetricsEvent) -> None:
        line = event.model_dump_json()
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
