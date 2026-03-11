"""
Session persistence abstraction.

InMemorySessionStore (default) — module-level dict, identical to the previous
_sessions behaviour. Sessions are lost on restart.

RedisSessionStore — JSON-serialized sessions stored in Redis with a 24-hour TTL.
Activated by setting REDIS_URL in the environment. Falls back to in-memory if
redis-py is not installed or the server is unreachable.

Usage:
    store = make_session_store(settings.redis_url)   # called once in lifespan
    app.state.session_store = store
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from phflow.settings import settings


class SessionStore(ABC):

    @abstractmethod
    def get(self, session_id: str) -> dict | None: ...

    @abstractmethod
    def create(self) -> tuple[str, dict]: ...

    @abstractmethod
    def save(self, session_id: str, session: dict) -> None: ...

    @abstractmethod
    def exists(self, session_id: str) -> bool: ...

    def _new_session(self, sid: str) -> dict:
        return {
            "session_id": sid,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "messages": [],
        }

    @staticmethod
    def _trim_messages(session: dict) -> None:
        """Keep only the most recent MAX_MESSAGES to bound memory usage."""
        msgs = session.get("messages", [])
        if len(msgs) > settings.session_max_messages:
            session["messages"] = msgs[-settings.session_max_messages:]


class InMemorySessionStore(SessionStore):
    def __init__(self) -> None:
        self._store: dict[str, dict] = {}

    def get(self, session_id: str) -> dict | None:
        return self._store.get(session_id)

    def create(self) -> tuple[str, dict]:
        sid = str(uuid.uuid4())
        session = self._new_session(sid)
        self._store[sid] = session
        return sid, session

    def save(self, session_id: str, session: dict) -> None:
        self._trim_messages(session)
        self._store[session_id] = session

    def exists(self, session_id: str) -> bool:
        return session_id in self._store


class RedisSessionStore(SessionStore):
    _TTL = 86400  # 24 hours

    def __init__(self, redis_url: str) -> None:
        import redis  # lazy import — optional dependency

        self._redis = redis.from_url(redis_url, decode_responses=True)

    def _key(self, session_id: str) -> str:
        return f"phflow:session:{session_id}"

    def get(self, session_id: str) -> dict | None:
        raw = self._redis.get(self._key(session_id))
        return json.loads(raw) if raw else None

    def create(self) -> tuple[str, dict]:
        sid = str(uuid.uuid4())
        session = self._new_session(sid)
        self.save(sid, session)
        return sid, session

    def save(self, session_id: str, session: dict) -> None:
        self._trim_messages(session)
        self._redis.set(
            self._key(session_id),
            json.dumps(session, default=str),
            ex=self._TTL,
        )

    def exists(self, session_id: str) -> bool:
        return bool(self._redis.exists(self._key(session_id)))


def make_session_store(redis_url: str | None) -> SessionStore:
    """
    Return a RedisSessionStore when redis_url is set and redis-py is available
    and the server responds to PING. Falls back to InMemorySessionStore silently.
    """
    if redis_url:
        try:
            store = RedisSessionStore(redis_url)
            store._redis.ping()
            return store
        except Exception:
            pass  # fall through to in-memory
    return InMemorySessionStore()
