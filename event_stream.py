"""Threaded event stream with publish/consume and handlers."""

from __future__ import annotations

import queue
import threading
from typing import Callable, Dict, List, Optional


class EventStream:
    def __init__(self):
        self._queue: "queue.Queue[Dict[str, object]]" = queue.Queue()
        self._handlers: List[Callable[[Dict[str, object]], None]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def register_handler(self, handler: Callable[[Dict[str, object]], None]) -> None:
        self._handlers.append(handler)

    def publish_event(self, event: Dict[str, object]) -> None:
        self._queue.put(dict(event))

    def consume_events(self) -> List[Dict[str, object]]:
        events: List[Dict[str, object]] = []
        while True:
            try:
                event = self._queue.get_nowait()
            except queue.Empty:
                break
            events.append(event)
        return events

    def _run(self) -> None:
        while self._running:
            try:
                event = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            for handler in self._handlers:
                try:
                    handler(event)
                except Exception:
                    continue

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
