"""THE BUILD SYSTEM FOR MEASUREMENTS: owners declare units and graphs, a composer runs what is not yet stored.

A UNIT is an owner's code for one measurement, named `module:Class` and built with its parameters. It says where its
records live (`location`), what its result depends on (`identity()`, reduced to hashes and computed in the owner's own
environment), and runs in three phases:

    setup(ws)            build what the measurement needs (fixtures, a compiled arm); raise Refusal to decline
    execute(prepared, ws)   the measurement, writing raw files into the workspace and nothing else
    post(ws)             read the workspace, return the record to store (runs in a fresh process, after the device is
                         released, and can be retried without executing again)

A TIMED unit (`timed = True`) does not execute on its own: its setup returns a `Timed` (a `call()` returning the elapsed
milliseconds of one timed call, and a description of what it built), and a session interleaves the calls of every member.

A GRAPH is an owner's function (`module:function`) returning `Node` declarations: a name, a unit, its parameters, the
nodes it depends on. An INSTANCE is a graph loaded in one environment (`Env`: a label, a python, a directory, variables);
`describe` asks that environment for the declarations with every node's location and identity. A node's KEY is a hash of
its unit, parameters, identity and each dependency's key and output digest -- never a label, a path or the driver that
ran it -- so an owner's own command line and a composer that includes the owner's graph reach one key.

`engine.run` executes; `rola_devtools.graph` has no store of its own (the caller passes one, rola-results in practice).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


class Refusal(Exception):
    """A unit declining what it was given (an arm its binary lacks, a cell it does not take): stored as a refusal row, so
    it is not retried until its identity changes, and never a failure."""


@dataclass(frozen=True)
class Timed:
    """What a timed unit's setup returns: `call()` runs one timed call and returns its elapsed milliseconds measured by
    the stopwatch `instrument` names; `built` is the unit's own description of what it built."""

    call: Callable[[], float]
    built: dict
    instrument: str


class Unit:
    """The base of an owner's unit. Subclasses set `location` and implement the phases they use."""

    location: str = ""
    repeatable: bool = False
    timed: bool = False

    def identity(self) -> dict:
        return {}

    def setup(self, ws: Path):
        return None

    def execute(self, prepared, ws: Path) -> None:
        raise NotImplementedError(f"{type(self).__name__} executes nothing")

    def post(self, ws: Path) -> dict:
        raise NotImplementedError(f"{type(self).__name__} posts nothing")


@dataclass(frozen=True)
class Node:
    """One node of an owner's graph: `unit` is `module:Class`, `params` its constructor's keywords, `deps` the names of
    the nodes whose outputs its phases read (`ws/deps/<name>.json`)."""

    name: str
    unit: str
    params: dict = field(default_factory=dict)
    deps: tuple[str, ...] = ()


@dataclass(frozen=True)
class Env:
    """Where an instance's code runs: `python` in `cwd` with `env` added. `graph` is the owner's graph function."""

    label: str
    python: str
    cwd: str
    graph: str
    env: dict = field(default_factory=dict)


__all__ = ["Env", "Node", "Refusal", "Timed", "Unit"]
