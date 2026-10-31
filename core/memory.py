"""短期记忆（带容量的滑动窗口）和长期记忆（内存字典）。

演示场景下长期记忆只存在于内存，不落盘。
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MemoryEntry:
    key: str
    value: Any
    timestamp: float = field(default_factory=time.time)
    ttl_seconds: Optional[float] = None
    importance: float = 1.0

    def expired(self) -> bool:
        return self.ttl_seconds is not None and time.time() - self.timestamp > self.ttl_seconds


class ShortTermMemory:
    def __init__(self, max_entries: int = 50):
        self._storage: deque[MemoryEntry] = deque(maxlen=max_entries)

    def add(self, key: str, value: Any, importance: float = 1.0,
            ttl_seconds: Optional[float] = None) -> None:
        self._storage.append(
            MemoryEntry(key=key, value=value, importance=importance, ttl_seconds=ttl_seconds)
        )

    def get(self, key: str) -> Optional[Any]:
        for entry in reversed(self._storage):
            if entry.key == key and not entry.expired():
                return entry.value
        return None

    def recent(self, n: int = 10) -> List[MemoryEntry]:
        return list(self._storage)[-n:]

    def search(self, keyword: str, limit: int = 5) -> List[MemoryEntry]:
        hits = []
        for entry in reversed(self._storage):
            if entry.expired():
                continue
            if keyword.lower() in str(entry.value).lower():
                hits.append(entry)
                if len(hits) >= limit:
                    break
        return hits

    def clear(self) -> None:
        self._storage.clear()

    def __len__(self) -> int:
        return len(self._storage)


class LongTermMemory:
    def __init__(self) -> None:
        self._patient_profiles: Dict[str, Dict[str, Any]] = {}

    def save_patient_profile(self, patient_id: str, profile: Dict[str, Any]) -> None:
        self._patient_profiles.setdefault(patient_id, {}).update(profile)
        self._patient_profiles[patient_id]["last_updated"] = time.time()

    def get_patient_profile(self, patient_id: str) -> Optional[Dict[str, Any]]:
        return self._patient_profiles.get(patient_id)

    def patient_count(self) -> int:
        return len(self._patient_profiles)


class MemoryManager:
    def __init__(self) -> None:
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()
