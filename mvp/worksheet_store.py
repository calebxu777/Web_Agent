from __future__ import annotations

import os
from typing import Protocol

import redis

from mvp.worksheet_models import WorksheetInstance


class WorksheetStore(Protocol):
    def get(self, session_id: str) -> WorksheetInstance | None:
        ...

    def save(self, session_id: str, instance: WorksheetInstance) -> None:
        ...

    def clear(self, session_id: str) -> None:
        ...


class InMemoryWorksheetStore:
    def __init__(self):
        self._items: dict[str, WorksheetInstance] = {}

    def get(self, session_id: str) -> WorksheetInstance | None:
        return self._items.get(session_id)

    def save(self, session_id: str, instance: WorksheetInstance) -> None:
        self._items[session_id] = instance.model_copy(deep=True)

    def clear(self, session_id: str) -> None:
        self._items.pop(session_id, None)


class RedisWorksheetStore:
    def __init__(
        self,
        host: str = "redis",
        port: int = 6379,
        db: int = 0,
        ttl_seconds: int = 3600,
        key_prefix: str = "worksheet:session:",
        socket_connect_timeout: float = 0.15,
        socket_timeout: float = 0.15,
    ):
        self.ttl_seconds = ttl_seconds
        self.key_prefix = key_prefix
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True,
            socket_connect_timeout=socket_connect_timeout,
            socket_timeout=socket_timeout,
        )
        self.redis_client.ping()

    def _key(self, session_id: str) -> str:
        return f"{self.key_prefix}{session_id}"

    def get(self, session_id: str) -> WorksheetInstance | None:
        raw = self.redis_client.get(self._key(session_id))
        if not raw:
            return None
        return WorksheetInstance.model_validate_json(raw)

    def save(self, session_id: str, instance: WorksheetInstance) -> None:
        self.redis_client.set(
            self._key(session_id),
            instance.model_dump_json(),
            ex=self.ttl_seconds,
        )

    def clear(self, session_id: str) -> None:
        self.redis_client.delete(self._key(session_id))


def build_worksheet_store(config: dict | None = None) -> WorksheetStore:
    episodic_cfg = (config or {}).get("memory", {}).get("episodic", {})
    port = int(os.environ.get("MVP_REDIS_PORT", episodic_cfg.get("port", 6379)))
    db = int(os.environ.get("MVP_REDIS_DB", episodic_cfg.get("db", 0)))
    ttl_seconds = int(os.environ.get("MVP_REDIS_TTL_SECONDS", episodic_cfg.get("ttl_seconds", 3600)))

    # MVP source of truth is the docker-compose service name `redis`.
    # Fall back for local runs or tests where Docker service discovery is absent.
    host_candidates = [
        os.environ.get("MVP_REDIS_HOST", "").strip(),
        "redis",
        str(episodic_cfg.get("host", "")).strip(),
        "localhost",
    ]

    seen: set[str] = set()
    for host in host_candidates:
        if not host or host in seen:
            continue
        seen.add(host)
        try:
            return RedisWorksheetStore(
                host=host,
                port=port,
                db=db,
                ttl_seconds=ttl_seconds,
            )
        except Exception:
            continue

    return InMemoryWorksheetStore()
