from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Protocol

import redis

from mvp.preference_models import (
    PreferenceItem,
    SessionPreferenceProfile,
    StoredPreferenceProfile,
    merge_preference_maps,
)


class SessionPreferenceStore(Protocol):
    def get(self, session_id: str) -> SessionPreferenceProfile | None:
        ...

    def save(self, profile: SessionPreferenceProfile) -> None:
        ...

    def clear(self, session_id: str) -> None:
        ...


class InMemorySessionPreferenceStore:
    def __init__(self):
        self._items: dict[str, SessionPreferenceProfile] = {}

    def get(self, session_id: str) -> SessionPreferenceProfile | None:
        profile = self._items.get(session_id)
        return profile.model_copy(deep=True) if profile else None

    def save(self, profile: SessionPreferenceProfile) -> None:
        self._items[profile.session_id] = profile.model_copy(deep=True)

    def clear(self, session_id: str) -> None:
        self._items.pop(session_id, None)


class RedisSessionPreferenceStore:
    def __init__(
        self,
        host: str = "redis",
        port: int = 6379,
        db: int = 0,
        ttl_seconds: int = 3600,
        key_prefix: str = "prefs:session:",
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

    def get(self, session_id: str) -> SessionPreferenceProfile | None:
        raw = self.redis_client.get(self._key(session_id))
        if not raw:
            return None
        return SessionPreferenceProfile.model_validate_json(raw)

    def save(self, profile: SessionPreferenceProfile) -> None:
        self.redis_client.set(
            self._key(profile.session_id),
            profile.model_dump_json(),
            ex=self.ttl_seconds,
        )

    def clear(self, session_id: str) -> None:
        self.redis_client.delete(self._key(session_id))


def build_session_preference_store(
    config: dict | None = None,
    ttl_seconds: int = 3600,
) -> SessionPreferenceStore:
    episodic_cfg = (config or {}).get("memory", {}).get("episodic", {})
    port = int(os.environ.get("MVP_REDIS_PORT", episodic_cfg.get("port", 6379)))
    db = int(os.environ.get("MVP_REDIS_DB", episodic_cfg.get("db", 0)))
    resolved_ttl = int(
        os.environ.get(
            "MVP_PREFERENCE_REDIS_TTL_SECONDS",
            ttl_seconds or episodic_cfg.get("ttl_seconds", 3600),
        )
    )

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
            return RedisSessionPreferenceStore(
                host=host,
                port=port,
                db=db,
                ttl_seconds=resolved_ttl,
            )
        except Exception:
            continue

    return InMemorySessionPreferenceStore()


class SQLitePreferenceProfileStore:
    def __init__(self, db_path: str = "data/processed/user_preferences.db"):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id            TEXT PRIMARY KEY,
                preferences_json   TEXT NOT NULL,
                updated_at         REAL NOT NULL,
                source_session_id  TEXT
            )
            """
        )
        self.conn.commit()

    def get(self, user_id: str) -> StoredPreferenceProfile | None:
        row = self.conn.execute(
            """
            SELECT user_id, preferences_json, updated_at, source_session_id
            FROM user_preferences
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()
        if not row:
            return None

        return StoredPreferenceProfile(
            user_id=row["user_id"],
            preferences=json.loads(row["preferences_json"] or "{}"),
            updated_at=float(row["updated_at"] or time.time()),
            source_session_id=row["source_session_id"] or "",
        )

    def save(self, profile: StoredPreferenceProfile) -> None:
        self.conn.execute(
            """
            INSERT INTO user_preferences (user_id, preferences_json, updated_at, source_session_id)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                preferences_json = excluded.preferences_json,
                updated_at = excluded.updated_at,
                source_session_id = excluded.source_session_id
            """,
            (
                profile.user_id,
                json.dumps(profile.preferences, sort_keys=True),
                profile.updated_at,
                profile.source_session_id,
            ),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


class PreferenceStore:
    def __init__(
        self,
        session_store: SessionPreferenceStore,
        durable_store: SQLitePreferenceProfileStore,
    ):
        self.session_store = session_store
        self.durable_store = durable_store

    def get_session_profile(self, session_id: str) -> SessionPreferenceProfile | None:
        return self.session_store.get(session_id)

    def get_stored_profile(self, user_id: str) -> StoredPreferenceProfile | None:
        if not user_id:
            return None
        return self.durable_store.get(user_id)

    def get_combined_profile(self, user_id: str, session_id: str) -> StoredPreferenceProfile | None:
        session_profile = self.get_session_profile(session_id)
        stored_profile = self.get_stored_profile(user_id)
        if not session_profile and not stored_profile:
            return None

        merged_preferences = merge_preference_maps(
            stored_profile.preferences if stored_profile else {},
            session_profile.preferences if session_profile else {},
        )
        updated_at = max(
            stored_profile.updated_at if stored_profile else 0.0,
            session_profile.updated_at if session_profile else 0.0,
        )
        if updated_at <= 0.0:
            updated_at = time.time()
        return StoredPreferenceProfile(
            user_id=user_id or (session_profile.user_id if session_profile else ""),
            preferences=merged_preferences,
            updated_at=updated_at,
            source_session_id=(
                session_profile.session_id
                if session_profile
                else (stored_profile.source_session_id if stored_profile else "")
            ),
        )

    def update_session_preferences(
        self,
        user_id: str,
        session_id: str,
        preferences: list[PreferenceItem],
    ) -> SessionPreferenceProfile:
        profile = self.get_session_profile(session_id) or SessionPreferenceProfile(
            user_id=user_id,
            session_id=session_id,
        )
        profile.user_id = user_id or profile.user_id
        profile.session_id = session_id
        profile.merge_items(preferences)
        self.session_store.save(profile)
        return profile

    def finalize_session(self, user_id: str, session_id: str) -> StoredPreferenceProfile | None:
        session_profile = self.get_session_profile(session_id)
        if not session_profile or not session_profile.preferences:
            self.session_store.clear(session_id)
            return self.get_stored_profile(user_id)

        stored_profile = self.get_stored_profile(user_id)
        merged = StoredPreferenceProfile(
            user_id=user_id or session_profile.user_id,
            preferences=merge_preference_maps(
                stored_profile.preferences if stored_profile else {},
                session_profile.preferences,
            ),
            updated_at=time.time(),
            source_session_id=session_id,
        )
        self.durable_store.save(merged)
        self.session_store.clear(session_id)
        return merged

    def clear_session(self, session_id: str) -> None:
        self.session_store.clear(session_id)

    def close(self) -> None:
        self.durable_store.close()
