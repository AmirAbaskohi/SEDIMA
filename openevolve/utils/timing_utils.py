"""Timing utilities for evaluator instrumentation."""

from __future__ import annotations

import time
from typing import Dict, Optional


class TimingBlocks:
    """Collects block-level timings for a single evaluation run."""

    def __init__(self) -> None:
        self._blocks: Dict[str, float] = {}

    def block(self, name: str) -> "_BlockTimer":
        return _BlockTimer(self, name)

    def add(self, name: str, elapsed_sec: float) -> None:
        if not name:
            return
        self._blocks[name] = float(elapsed_sec)

    def to_dict(self) -> Dict[str, float]:
        return dict(self._blocks)


class _BlockTimer:
    def __init__(self, timing_blocks: TimingBlocks, name: str) -> None:
        self._timing_blocks = timing_blocks
        self._name = name
        self._start: Optional[float] = None

    def __enter__(self) -> "_BlockTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._start is None:
            return
        elapsed = time.perf_counter() - self._start
        self._timing_blocks.add(self._name, elapsed)
