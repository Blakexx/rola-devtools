"""Timing entry executors for the timing system's tests: fixed elapsed times, a counter a reset must restore, a setup
that cannot run, and a clock reader."""
from __future__ import annotations

import os

from rola_devtools.timing import Timed


def fixed(cell, params) -> Timed:
    ms = params["ms"] * cell.tokens
    return Timed(call=lambda: ms, built={"ms": ms, "held": os.environ.get("ROLA_GPU_LOCK_HELD"), "pid": os.getpid()},
                 instrument="fixed")


def stateful(cell, params) -> Timed:
    """Each call reads a state the call advances; with its reset every call sees the first call's state."""
    state = {"step": 0}

    def call() -> float:
        state["step"] += 1
        return float(state["step"])

    def reset() -> None:
        state["step"] = 0

    return Timed(call=call, built={}, instrument="fixed", reset=reset if params.get("reset") else None)


def biased(cell, params) -> Timed:
    """A call whose time depends on the worker it runs in: the first worker to build it on a cell is twice as fast as any
    other, a worker's bias a null gate exists to find."""
    try:
        with open(os.path.join(os.environ["FAKE_BIAS_DIR"], cell.name), "x"):
            ms = 1.0
    except FileExistsError:
        ms = 2.0
    return Timed(call=lambda: ms, built={"pid": os.getpid()}, instrument="fixed")


def breaks(cell, params) -> Timed:
    """An entry that builds and then fails a call: a domain failure the session records while the rest are timed."""
    def call() -> float:
        raise RuntimeError(f"{cell.name}: this launch cannot run here")

    return Timed(call=call, built={}, instrument="fixed")


def unbuilt(cell, params) -> Timed:
    raise LookupError(f"{cell.name}: this binary carries no kernel for it")


def other_clock(cell, params) -> Timed:
    return Timed(call=lambda: 1.0, built={}, instrument="another")


def clock() -> float:
    return float(os.environ.get("FAKE_CLOCK_GHZ", "1.0"))
