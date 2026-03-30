from __future__ import annotations

import json
import time
from pathlib import Path

from src.schema import ChatMessage


class InMemoryConversationRecordingStore:
    def __init__(self):
        self._messages_by_session: dict[str, list[ChatMessage]] = {}

    def add_message(self, session_id: str, message: ChatMessage) -> None:
        self._messages_by_session.setdefault(session_id, []).append(message.model_copy(deep=True))

    def get_messages(self, session_id: str) -> list[ChatMessage]:
        return [message.model_copy(deep=True) for message in self._messages_by_session.get(session_id, [])]

    def clear_session(self, session_id: str) -> None:
        self._messages_by_session.pop(session_id, None)


def _conversation_entry(message: ChatMessage) -> dict[str, object]:
    role_key = "agent" if message.role == "assistant" else (message.role or "user")
    payload: dict[str, object] = {
        role_key: message.content,
        "timestamp": message.timestamp or time.time(),
    }
    if message.image_url:
        payload["image_url"] = message.image_url
    return payload


def build_conversation_record(
    *,
    user_id: str,
    session_id: str,
    messages: list[ChatMessage],
    inferred_preferences: dict[str, object] | None = None,
    record_type: str = "mvp",
) -> dict[str, object]:
    conversation_started_at = (
        messages[0].timestamp
        if messages and messages[0].timestamp is not None
        else time.time()
    )
    return {
        "type": record_type,
        "created_at": conversation_started_at,
        "finalized_at": time.time(),
        "user_id": user_id,
        "session_id": session_id,
        "conversation": [_conversation_entry(message) for message in messages],
        "inferred_preferences": dict(inferred_preferences or {}),
    }


def append_record_to_jsonl(output_path: str | Path, record: dict[str, object]) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return path
