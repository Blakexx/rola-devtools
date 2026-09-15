"""THE TIMING SYSTEM: interleaved timing and memory, as targets of the declared build system (`rola_devtools.build`).

Two layers, two vocabularies. The build system runs TARGETS; the timing system holds ENTRIES -- one checkout's timed
callable on one central cell -- in a SERVER (a pool of workers, one per checkout environment, living in the build
system's own worker for the run). The targets (`declare.py` builds them):

    start_timing_server    the server, for this build
    register_timing        ONE target that registers one entry per cell it takes (it runs in the checkout's
                           environment, so an entry knows where it runs; registering builds nothing)
    register_clock_reader  the checkout code that reads the device's clock, for the proof
    measure_timing         a session: its entries set up behind a barrier, warmed, then called one at a time in a
                           fresh random order each rep, each call preceded by the entry's untimed reset; the clock
                           proven before and after; every sample stored in the order taken
    measure_memory         each entry alone: the allocator's peak over its calls, plus what it holds outside it
    stop_timing_server     always_run: the server's workers stopped

An ENTRY EXECUTOR, in the checkout, is `executor(cell, params) -> Timed`: `call()` runs one launch and returns its
elapsed milliseconds by the stopwatch `instrument` names; `built` describes what it built; `reset()`, when the launch
changes what the next call reads (a carried state, a decode step), restores exactly what the first call saw, untimed;
`outside_allocator()` returns device bytes its framework's allocator does not see. Every call does the same work on the
same data. An entry that cannot set up (a kernel this binary lacks) is a DOMAIN failure: recorded in the session, the
rest timed. A clock read off the lock, two stopwatches in one session, or a crash of the timing itself is a BUILD
failure.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Timed:
    call: Callable[[], float]
    built: dict
    instrument: str
    reset: Callable[[], None] | None = None
    outside_allocator: Callable[[], int] | None = None


__all__ = ["Timed"]
