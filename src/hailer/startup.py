"""Lightweight startup milestones, visible with ``hailer --verbose``.

All times share one monotonic origin. Input readiness and readiness to answer
are separate: displaying a prompt must not hide a wait before the first turn.
No prompts, paths, URLs or credentials belong in these records.
"""

from __future__ import annotations

import time

from hailer.log import get_logger

log = get_logger("hailer.startup")


class StartupTimings:
    """Record each milestone once, in seconds since session setup began."""

    def __init__(self, *, started_at: float | None = None) -> None:
        self.started_at = time.perf_counter() if started_at is None else started_at
        self.milestones: dict[str, float] = {}

    def mark(self, name: str) -> float:
        if name not in self.milestones:
            elapsed = time.perf_counter() - self.started_at
            self.milestones[name] = elapsed
            log.debug("startup %s: %.3f s", name, elapsed)
        return self.milestones[name]
