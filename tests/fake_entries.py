"""Timing entry executors for the timing system's tests: fixed elapsed times, a counter a reset must restore, a setup
that cannot run, and a clock reader."""
from __future__ import annotations

import os

from rola_devtools.timing import Timed


def fixed(cell, params) -> Timed:
    ms = params["ms"] * cell.tokens
    return Timed(call=lambda: ms, built={"ms": ms, "held": os.environ.get("ROLA_GPU_LOCK_HELD")}, instrument="fixed")


def stateful(cell, params) -> Timed:
    """Each call reads a state the call advances; with its reset every call sees the first call's state."""
    state = {"step": 0}

    def call() -> float:
        state["step"] += 1
        return float(state["step"])

    def reset() -> None:
        state["step"] = 0

    return Timed(call=call, built={}, instrument="fixed", reset=reset if params.get("reset") else None)


def unbuilt(cell, params) -> Timed:
    raise LookupError(f"{cell.name}: this binary carries no kernel for it")


def other_clock(cell, params) -> Timed:
    return Timed(call=lambda: 1.0, built={}, instrument="another")


def clock() -> float:
    return float(os.environ.get("FAKE_CLOCK_GHZ", "1.0"))
