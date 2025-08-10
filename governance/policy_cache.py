"""
Simple in-memory cache for policy evaluation results.
"""
from __future__ import annotations

from typing import Any, Dict
from datetime import datetime, timedelta


class PolicyCache:
    def __init__(self, ttl_seconds: int = 300):
        self.ttl = ttl_seconds
        self.store: Dict[str, Dict[str, Any]] = {}

    def get(self, key: str) -> Any | None:
        item = self.store.get(key)
        if not item:
            return None
        if datetime.now() - item["ts"] > timedelta(seconds=self.ttl):
            del self.store[key]
            return None
        return item["value"]

    def set(self, key: str, value: Any) -> None:
        self.store[key] = {"value": value, "ts": datetime.now()}


