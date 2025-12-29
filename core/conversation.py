from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import List, Optional, Sequence

DEFAULT_CLEAR_PHRASES = (
    "new conversation",
    "new topic",
    "new question",
    "start over",
    "clear context",
    "reset conversation",
    "forget that",
    "forget what i said",
)


def parse_clear_phrases(value: Optional[str]) -> List[str]:
    if not value:
        return list(DEFAULT_CLEAR_PHRASES)
    phrases = [part.strip().lower() for part in value.split(",") if part.strip()]
    return phrases or list(DEFAULT_CLEAR_PHRASES)


def should_clear_history(text: str, phrases: Sequence[str]) -> bool:
    normalized = " ".join(text.lower().split())
    return any(phrase in normalized for phrase in phrases)


@dataclass
class ConversationMemory:
    max_turns: int
    ttl_seconds: float
    _messages: List[dict] = field(default_factory=list)
    _last_updated: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def reset(self) -> None:
        with self._lock:
            self._messages.clear()
            self._last_updated = 0.0

    def maybe_expire(self) -> None:
        if self.ttl_seconds <= 0:
            return
        now = time.time()
        with self._lock:
            if self._messages and (now - self._last_updated) > self.ttl_seconds:
                self._messages.clear()
                self._last_updated = 0.0

    def add(self, role: str, text: str) -> None:
        if not text or self.max_turns <= 0:
            return
        with self._lock:
            self._messages.append({"role": role, "text": text})
            max_messages = self.max_turns * 2
            if max_messages and len(self._messages) > max_messages:
                self._messages = self._messages[-max_messages:]
            self._last_updated = time.time()

    def get_messages(self) -> List[dict]:
        with self._lock:
            return list(self._messages)
