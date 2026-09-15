"""THE BUILD SYSTEM: nodes keyed by their content; what is stored is skipped, the rest runs in dependency order.

    outcomes = run(nodes, store, execute)

A NODE is an id, its SEMANTICS, a LOCATION and its DEPENDENCIES. The id names the node in one build and nothing else: it
is never parsed and never enters a key. The semantics are a JSON object holding everything the node's result depends on
besides its dependencies. The location is where its record lives in the store. Each dependency is a ROLE (the name the
node reads it by) and the id of the node it names. Two flags qualify a node: REPEATABLE (a run asked to `repeat` adds a
sample to a complete record) and LOCAL (its output describes files on this machine, such as a built binary, so a stored
result counts only while `present` finds it still true here). `payload` is the executor's own and is never keyed.

A node's KEY is sha256 over the canonical JSON of its semantics with `deps` added -- each role's dependency key and the
sha256 of that dependency's stored output -- the first 24 hex digits. A result therefore moves with anything it read,
and one node declared by two drivers under two ids reaches one record.

`run` orders the nodes, dependencies first (an unknown dependency, a repeated id or a cycle is refused), and takes each
in turn:

- BLOCKED when a dependency has no usable output (it failed, refused or was blocked);
- COMPLETE when its record holds a successful sample, unless `force`, or `repeat` on a repeatable node, or a local node
  `present` no longer finds;
- PENDING under `dry` (with no key yet when a dependency is itself pending);
- otherwise `execute(node, deps)` runs it -- `deps` maps each role to its dependency's stored output file -- and returns
  an `Executed`: an output, an output file, a refusal or an error, stored as one sample under the key. A refusal is a
  successful sample whose output is `{"refused": why}`, so it is not retried until the semantics change; its dependents
  are blocked. An exception out of `execute` is stored as an error, its text given by `explain`.

The build system knows no cell, arm, lock or session: `rola_devtools.measure` composes measurement nodes and executes
them. `store(location)` returns the caller's store for a location -- `get(key)`, `put(semantics, output=|output_file=|
error=, wall_s=, provenance=)` and `output(record)` (rola-results' `Store`), whose key formula this module shares.
"""
from __future__ import annotations

import hashlib
import json
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

#: the statuses an `Outcome` can carry
STATUSES = ("complete", "ran", "refused", "failed", "blocked", "pending")


@dataclass(frozen=True)
class Node:
    id: str
    semantics: dict
    location: str
    #: ``((role, node id), ...)``
    deps: tuple[tuple[str, str], ...] = ()
    repeatable: bool = False
    local: bool = False
    payload: object = field(default=None, compare=False)


@dataclass(frozen=True)
class Executed:
    """One execution: exactly one of `output` (a JSON value), `output_file` (a file the store moves in), `refused` or
    `error`, with the wall time and where it came from."""

    output: object = None
    output_file: str | None = None
    refused: str | None = None
    error: str | None = None
    wall_s: float | None = None
    provenance: dict = field(default_factory=dict)


@dataclass
class Outcome:
    id: str
    key: str
    status: str
    wall_s: float = 0.0
    detail: str = ""


def key(semantics: dict) -> str:
    return hashlib.sha256(json.dumps(semantics, sort_keys=True).encode()).hexdigest()[:24]


def digest(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def order(nodes) -> list[Node]:
    """Every node, each after its dependencies."""
    by_id: dict[str, Node] = {}
    for node in nodes:
        if node.id in by_id:
            raise ValueError(f"two nodes share the id {node.id}")
        if "deps" in node.semantics:
            raise ValueError(f"{node.id}: `deps` is the key's own field, not a semantic")
        by_id[node.id] = node
    out: list[Node] = []
    state: dict[str, str] = {}

    def visit(node: Node) -> None:
        if state.get(node.id) == "done":
            return
        if state.get(node.id) == "open":
            raise ValueError(f"a dependency cycle through {node.id}")
        state[node.id] = "open"
        for role, dep in node.deps:
            if dep not in by_id:
                raise KeyError(f"{node.id} reads {dep} as {role}, which is not a node of this build")
            visit(by_id[dep])
        state[node.id] = "done"
        out.append(node)

    for node in by_id.values():
        visit(node)
    return out


def _refused(path: Path) -> str | None:
    if path.suffix != ".json":
        return None
    doc = json.loads(path.read_text())
    return doc["refused"] if isinstance(doc, dict) and set(doc) == {"refused"} else None


def _complete(record: dict | None) -> bool:
    return bool(record) and any(sample["ok"] for sample in record["samples"])


def run(nodes, store: Callable[[str], object], execute: Callable[[Node, dict[str, Path]], Executed], *,
        present: Callable[[Node, Path], bool] = lambda node, output: True,
        explain: Callable[[Node, BaseException], str] = lambda node, ex: f"{type(ex).__name__}: {ex}\n"
                                                                           f"{traceback.format_exc()[-2000:]}",
        repeat: bool = False, force: bool = False, dry: bool = False,
        log: Callable[[str], None] = print) -> list[Outcome]:
    usable: dict[str, tuple[str, Path]] = {}
    planned: set[str] = set()
    outcomes: list[Outcome] = []

    def done(outcome: Outcome) -> None:
        outcomes.append(outcome)
        log(f"{outcome.status:9s} {outcome.id}" + (f" [{outcome.key[:12]}]" if outcome.key else "")
            + (f" {outcome.wall_s:.1f}s" if outcome.status in ("ran", "failed") else "")
            + (f": {outcome.detail[-300:]}" if outcome.detail else ""))

    for node in order(nodes):
        waiting = [dep for _role, dep in node.deps if dep in planned]
        if waiting:
            planned.add(node.id)
            done(Outcome(node.id, "", "pending", detail=f"keyed once {', '.join(waiting)} has run"))
            continue
        missing = [dep for _role, dep in node.deps if dep not in usable]
        if missing:
            done(Outcome(node.id, "", "blocked", detail=f"no usable result for {', '.join(missing)}"))
            continue
        semantics = {**node.semantics, "deps": {role: {"key": usable[dep][0], "output": digest(usable[dep][1])}
                                                for role, dep in sorted(node.deps)}}
        k, st = key(semantics), store(node.location)
        record = st.get(k)
        if _complete(record) and not force and not (repeat and node.repeatable):
            path = st.output(record)
            why = _refused(path)
            if why is not None:
                done(Outcome(node.id, k, "refused", detail=why))
                continue
            if not node.local or present(node, path):
                usable[node.id] = (k, path)
                done(Outcome(node.id, k, "complete"))
                continue
        if dry:
            planned.add(node.id)
            done(Outcome(node.id, k, "pending"))
            continue
        started = time.time()
        try:
            result = execute(node, {role: usable[dep][1] for role, dep in node.deps})
        except Exception as ex:  # noqa: BLE001 -- stored as the node's failed sample
            result = Executed(error=explain(node, ex))
        given = [f for f in ("output", "output_file", "refused", "error") if getattr(result, f) is not None]
        if len(given) != 1:
            result = Executed(error=f"the executor returned {given or 'nothing'}: an execution is exactly one of an "
                                    "output, an output file, a refusal or an error", provenance=result.provenance)
        wall = result.wall_s if result.wall_s is not None else time.time() - started
        if result.error is not None:
            st.put(semantics, error=result.error, wall_s=wall, provenance=result.provenance)
            done(Outcome(node.id, k, "failed", wall, result.error.strip().splitlines()[-1] if result.error.strip() else ""))
            continue
        if result.refused is not None:
            st.put(semantics, output={"refused": result.refused}, wall_s=wall, provenance=result.provenance)
            done(Outcome(node.id, k, "refused", wall, result.refused))
            continue
        if result.output_file is not None:
            st.put(semantics, output_file=result.output_file, wall_s=wall, provenance=result.provenance)
        else:
            st.put(semantics, output=result.output, wall_s=wall, provenance=result.provenance)
        usable[node.id] = (k, st.output(st.get(k)))
        done(Outcome(node.id, k, "ran", wall))
    return outcomes


__all__ = ["STATUSES", "Executed", "Node", "Outcome", "digest", "key", "order", "run"]
