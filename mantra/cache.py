"""In-memory LRU cache for semantic search results."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Generic, Hashable, Optional, TypeVar


K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    """Simple LRU cache with hit/miss metrics."""

    def __init__(self, capacity: int = 256):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = int(capacity)
        self._items: OrderedDict[K, V] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: K) -> Optional[V]:
        if key in self._items:
            self._items.move_to_end(key)
            self.hits += 1
            return self._items[key]
        self.misses += 1
        return None

    def set(self, key: K, value: V) -> None:
        if key in self._items:
            self._items.move_to_end(key)
        self._items[key] = value
        if len(self._items) > self.capacity:
            self._items.popitem(last=False)

    def clear(self) -> None:
        self._items.clear()
        self.hits = 0
        self.misses = 0

    def metrics(self) -> Dict[str, int]:
        return {
            "hits": int(self.hits),
            "misses": int(self.misses),
            "size": int(len(self._items)),
            "capacity": int(self.capacity),
        }

