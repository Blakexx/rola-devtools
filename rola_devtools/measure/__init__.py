"""THE MEASUREMENT SERVICE: packages register units, the service composes them with the central cells into build nodes
and runs them.

An owner -- rola, rola-bench, any package measured on the central cells (`rola_devtools.cells`) -- exposes a REGISTRY, a
function returning `Registration`s: a name unique in the owner, a unit class (`module:Class`) with its constructor
parameters, and the names of the registrations its nodes read. The unit kinds:

    Arm          timed on a cell: `setup(cell, ws)` returns a `Timed` whose `call()` is one timed launch; sessions
                 interleave arms, and every arm on every cell it accepts also gets a MEMORY node (the arm alone: the
                 allocator's peak over its calls, plus the bytes it holds outside the allocator)
    Instrument   exclusive on a cell, or on none: `setup(cell, ws)`, then `execute(prepared, ws)`
    Build        exclusive, on no cell: `execute(ws)` makes what the owner's other units run (a binary); its record is
                 local, so it counts only while `present(output)` finds the build still on this machine
    ClockReader  on no cell: `read_ghz()`, the device's clock as the owner's code reads it, which proves the host's clock
                 lock around every held phase

A unit states which cells it takes -- `accepts(cell)` returns None, or why not -- and what its result depends on besides
the cell -- `identity(cell)`, reduced to hashes (`rola_devtools.build.identity`) and computed in the owner's own
environment. Acceptance is declared, never discovered by a failure; a setup may still raise `Refusal` for a reason only
the device knows. Every unit's `post(ws, handle)` runs after the device is released and is the only phase that writes a
result: `handle.output(value)` or `handle.output_file(path)`.

`rola_devtools.measure.service` holds the composition (instances, sessions) and the execution; `python -m
rola_devtools.measure` runs one owner's registry from its own checkout.
"""
from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

#: the unit kinds, by the base class an owner's unit derives from
KINDS = ("arm", "instrument", "build", "clock")


class Refusal(Exception):
    """A setup declining for a reason only the device knows (an allocation that does not fit): stored as a refusal, so it
    is not retried until the node's semantics change, and never a failure."""


@dataclass(frozen=True)
class Registration:
    """One unit of an owner's registry: `unit` is `module:Class`, `params` its constructor's keywords, `deps` the names
    of registrations whose outputs its phases read (`ws/deps/<name>.json`; a per-cell dependency is read on the same
    cell)."""

    name: str
    unit: str
    params: dict = field(default_factory=dict)
    deps: tuple[str, ...] = ()


@dataclass(frozen=True)
class Timed:
    """What an arm's setup returns: `call()` runs one timed launch and returns its elapsed milliseconds by the stopwatch
    `instrument` names, the device synchronized on both sides; `built` is the arm's own JSON description of what it
    built; `outside_allocator()`, for an arm holding device memory its framework's caching allocator does not see (a
    state mapped through the driver's virtual memory API), returns those bytes now."""

    call: Callable[[], float]
    built: dict
    instrument: str
    outside_allocator: Callable[[], int] | None = None


class Handle:
    """Where a post writes its result, once: a JSON value or a file."""

    def __init__(self, ws: Path) -> None:
        self.dir = Path(ws) / ".result"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _once(self) -> None:
        if any(self.dir.iterdir()):
            raise RuntimeError("a post writes one result")

    def output(self, value) -> None:
        self._once()
        (self.dir / "output.json").write_text(json.dumps(value, sort_keys=True))

    def output_file(self, path: Path | str) -> None:
        self._once()
        path = Path(path)
        shutil.move(str(path), self.dir / f"output{path.suffix}")


class Unit:
    """The base of every unit kind. `location` is where the unit's records live; `repeatable` lets a run add samples to
    a complete record; `per_cell` is False for a unit that runs once, on no cell."""

    kind = ""
    location = ""
    repeatable = False
    per_cell = True

    def accepts(self, cell) -> str | None:
        return None

    def identity(self, cell) -> dict:
        return {}

    def post(self, ws: Path, handle: Handle) -> None:
        raise NotImplementedError(f"{type(self).__name__} posts nothing")


class Arm(Unit):
    """Timed on a cell. Its records are its sessions' (the session names their location) and its memory nodes'
    (`location`)."""

    kind = "arm"
    repeatable = True

    def setup(self, cell, ws: Path) -> Timed:
        raise NotImplementedError

    def post(self, ws: Path, handle: Handle) -> None:
        handle.output({"calls": len(json.loads((ws / "samples.json").read_text())["ms"])})


class Instrument(Unit):
    kind = "instrument"

    def setup(self, cell, ws: Path):
        return None

    def execute(self, prepared, ws: Path) -> None:
        raise NotImplementedError


class Build(Unit):
    kind = "build"
    per_cell = False

    def execute(self, ws: Path) -> None:
        raise NotImplementedError

    def present(self, output: dict) -> bool:
        """Whether this machine still holds what the stored `output` describes."""
        raise NotImplementedError


class ClockReader(Unit):
    kind = "clock"
    per_cell = False

    def read_ghz(self) -> float | None:
        raise NotImplementedError


__all__ = ["KINDS", "Arm", "Build", "ClockReader", "Handle", "Instrument", "Refusal", "Registration", "Timed", "Unit"]
