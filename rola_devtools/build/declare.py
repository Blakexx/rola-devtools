"""DECLARATIONS: the targets a build runs, declared by functions in files and composed by code.

A DECLARATION FILE is plain Python loaded by path (`load`), never by importing the package it describes, so a root can
load the same file from several checkouts. It defines functions that take a `Graph` and declare targets on it:

    def declare(g, checkout, timing=None):
        binary = g.node("binary", executor="benchmarks.executors:compile_kernel", env=checkout, holds={"host_cpu": "all"})
        sass = g.node("sass", executor="benchmarks.executors:inspect_sass", env=checkout, deps={"binary": binary})
        return {"binary": binary, "sass": sass, "all": g.group("all", [binary, sass])}

A TARGET is a job: an executor (`module:function`) that runs in an environment (`Env`: a python, a directory, variables;
None is the build system's own), the targets it reads by ROLE, the central cells it takes as INPUTS, its parameters,
the resources it HOLDS while it runs (`{"gpu": "all"}`, `{"host_cpu": 8}`), whether its result is cached (`cache`),
whether it runs after a failure (`always_run`), a VERIFY function run on a cache hit (a build whose binary may have
left the machine), and the CODE its result depends on (files of its environment's directory, hashed by the build
system). A GROUP is a target with no executor: a terminator that depends on its members.

Labels are scoped: `g.scoped("tip")` declares under `tip/`, so one declaration function called for two checkouts
declares two sets of targets. A label names a target in one build and nothing else: it never enters a key.
"""
from __future__ import annotations

import runpy
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, eq=False)
class Env:
    """Where an executor runs: `python` in `cwd` with `vars` added to its environment. `label` names it in logs."""

    label: str
    python: str
    cwd: str
    vars: dict = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class Target:
    label: str
    executor: str | None
    env: Env | None = None
    deps: dict = field(default_factory=dict)
    inputs: tuple[str, ...] = ()
    params: dict = field(default_factory=dict)
    holds: dict = field(default_factory=dict)
    cache: bool = True
    always_run: bool = False
    verify: str | None = None
    code: dict | None = None

    @property
    def group(self) -> bool:
        return self.executor is None


class Graph:
    """The targets one build declares, under a label scope. Every scope of a graph shares its targets."""

    def __init__(self, scope: str = "", targets: dict[str, Target] | None = None) -> None:
        self.scope, self.targets = scope, targets if targets is not None else {}

    def scoped(self, name: str) -> Graph:
        return Graph(f"{self.scope}{name}/", self.targets)

    def _add(self, target: Target) -> Target:
        if target.label in self.targets:
            raise ValueError(f"{target.label} is declared twice")
        for role, dep in target.deps.items():
            if not isinstance(dep, Target) or self.targets.get(dep.label) is not dep:
                raise ValueError(f"{target.label} reads {role} from a target this graph did not declare")
        self.targets[target.label] = target
        return target

    def node(self, name: str, *, executor: str, env: Env | None = None, deps: dict | None = None, inputs=(),
             params: dict | None = None, holds: dict | None = None, cache: bool = True, always_run: bool = False,
             verify: str | None = None, code: dict | None = None) -> Target:
        if ":" not in executor:
            raise ValueError(f"{name}: an executor is module:function, got {executor!r}")
        if always_run and cache:
            raise ValueError(f"{name}: an always_run target runs every build, so it cannot be cached (cache=False)")
        for resource, claim in (holds or {}).items():
            if claim != "all" and not (isinstance(claim, int) and claim > 0):
                raise ValueError(f"{name}: holds {resource} as {claim!r}; a claim is a positive count or 'all'")
        return self._add(Target(f"{self.scope}{name}", executor, env, dict(deps or {}), tuple(inputs), dict(params or {}),
                                dict(holds or {}), cache, always_run, verify, code))

    def group(self, name: str, members) -> Target:
        return self._add(Target(f"{self.scope}{name}", None, deps={f"m{i}": m for i, m in enumerate(members)},
                                cache=False))


def load(path: Path | str) -> dict:
    """A declaration file's globals, loaded by path: its functions are called with a graph by whoever loaded it."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"no declaration file {path}")
    return runpy.run_path(str(path), run_name=f"declare:{path.parent.name}")


__all__ = ["Env", "Graph", "Target", "load"]
