"""The timing targets, declared: helpers a declaration file calls on its graph (`rola_devtools.build.declare.Graph`)."""
from __future__ import annotations

EXECUTORS = "rola_devtools.timing.executors:"
#: launches of each entry discarded before any is recorded: the first build schedules and warm caches
WARMUP_FLOOR = 10


def start_timing_server(g, name: str = "timing-server"):
    return g.node(name, executor=EXECUTORS + "start_server", cache=False)


def stop_timing_server(g, name: str = "timing-server-stop", *, server, after):
    deps = {"server": server, **{f"a{i}": t for i, t in enumerate(after)}}
    return g.node(name, executor=EXECUTORS + "stop_server", deps=deps, cache=False, always_run=True)


def register_timing(g, name: str, *, server, env, executor: str, cells, params: dict | None = None, deps: dict | None = None,
                    code: dict | None = None):
    """One target registering `executor` (in `env`) on every cell TARGET of `cells` as a timing entry: the caller
    declares the cell nodes (`rola_devtools.cells.declare.cells`) and this takes their records as data inputs, so the
    timing system knows a cell only as the record a node it depends on produced."""
    return g.node(name, executor=EXECUTORS + "register", env=env, deps={"server": server, **(deps or {})},
                  inputs=cells, params={"executor": executor, "params": params or {}}, code=code, cache=False)


def register_clock_reader(g, name: str, *, server, env, executor: str, deps: dict | None = None, code: dict | None = None):
    return g.node(name, executor=EXECUTORS + "register_clock", env=env, deps={"server": server, **(deps or {})},
                  params={"executor": executor}, code=code, cache=False)


def measure_timing(g, name: str, *, server, entries, clock=None, cells=None, rounds: int = 8, reps: int = 11,
                   warmup: int = WARMUP_FLOOR, seed: int = 0, claim: str = ""):
    """A session over the entries `entries` register (on `cells` only, when given)."""
    if warmup < WARMUP_FLOOR or reps % 2 == 0 or rounds < 1:
        raise ValueError(f"{name}: warmup at least {WARMUP_FLOOR}, odd reps and a round, got {warmup}, {reps}, {rounds}")
    deps = {"server": server, **{f"r{i}": e for i, e in enumerate(entries)}, **({"clock": clock} if clock else {})}
    return g.node(name, executor=EXECUTORS + "measure", deps=deps, holds={"gpu": "all", "clock": 1}, cache=False,
                  params={"cells": list(cells) if cells is not None else None, "rounds": rounds, "reps": reps,
                          "warmup": warmup, "seed": seed, "claim": claim})


def measure_null_gate(g, name: str, *, server, entry, clock=None, cells=None, rounds: int = 8, reps: int = 11,
                      warmup: int = WARMUP_FLOOR, seed: int = 0):
    """A null gate over what one registration `entry` registers (on `cells` only, when given): each entry timed against
    itself in a second worker, so a comparison across workers is trusted only where the gate finds no worker's bias."""
    if warmup < WARMUP_FLOOR or reps % 2 == 0 or rounds < 1:
        raise ValueError(f"{name}: warmup at least {WARMUP_FLOOR}, odd reps and a round, got {warmup}, {reps}, {rounds}")
    deps = {"server": server, "r0": entry, **({"clock": clock} if clock else {})}
    return g.node(name, executor=EXECUTORS + "null_gate", deps=deps, holds={"gpu": "all", "clock": 1}, cache=False,
                  params={"cells": list(cells) if cells is not None else None, "rounds": rounds, "reps": reps,
                          "warmup": warmup, "seed": seed})


def measure_memory(g, name: str, *, server, entries, cells=None, calls: int = 5):
    deps = {"server": server, **{f"r{i}": e for i, e in enumerate(entries)}}
    return g.node(name, executor=EXECUTORS + "memory", deps=deps, holds={"gpu": "all"}, cache=False,
                  params={"cells": list(cells) if cells is not None else None, "calls": calls})


__all__ = ["WARMUP_FLOOR", "measure_memory", "measure_null_gate", "measure_timing", "register_clock_reader",
           "register_timing", "start_timing_server", "stop_timing_server"]
