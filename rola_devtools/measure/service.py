"""THE SERVICE: instances of owners' registries composed with the central cells into build nodes, and the executor that
runs them.

    with Service(instances) as service:
        nodes = service.nodes(cells, sessions=[...])
        outcomes = service.run(nodes, store)

An INSTANCE is one owner's registry (`module:function`) in one environment -- a label, a python, a directory, variables
-- with the role the run gives it (`subject`, `reference`, `library`). Its worker (`rola_devtools.measure.worker`) starts
once and serves the whole run. The service loads the central registry once and hands every worker the records of the
cells it composes, so within a run a cell name is one input for every instance.

COMPOSITION. Every selected unit of every instance, on every cell it accepts, is a node: an instrument, a build, and for
an arm a MEMORY node (the arm alone). A unit on no cell is one node. A unit's dependencies are its instance's nodes of
the registrations it reads, on the same cell when they run per cell. A SESSION is timed arms interleaved: its members are
every instance's arms named in `arms` on every cell in `cells` each accepts, in instance order, then arm order, then cell
order. Its semantics are its members' and its timing; its reference instance and its relation (a group's claim, the
instances' roles) are recorded with it and never keyed.

A node's semantics are its kind, its unit and parameters, the identity its owner computed, the cell's record and the
digest of the central registry's code that draws it (`draw`); the ids -- `label:name@cell`, `memory label:name@cell`,
`session name` -- are for reading a log and nothing parses them.

BUILDS FIRST. A unit's acceptance and identity may read what its instance builds (a binary's arm tables, its digest), so
`build(store)` runs every instance's builds before `nodes` describes the units on cells; a build that fails or refuses
leaves its instance undescribable, and `nodes` refuses it by name.

EXECUTION, under `hold` (the GPU lock, exclusive, and the host's clock lock proven by an instance's `ClockReader` after
each hold): an instrument's setup and execute; an arm's memory measurement; a session's setups, then a barrier, then
warmup and interleaved rounds of calls in a fresh random order each rep. What a node prepared is dropped before the hold
is released; then each post writes its result through its handle. A build runs without the device held (it takes the
host's compute budget itself), and its instance's worker is restarted after it, since the old build may be loaded.
"""
from __future__ import annotations

import contextlib
import json
import os
import random
import selectors
import shutil
import statistics
import subprocess
import tempfile
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ..build import Executed, Node, Outcome, run
from ..interleave.driver import WARMUP_FLOOR, iqr

#: seconds a worker may take to answer one request before its node fails
REPLY_TIMEOUT_S = 3600.0
#: the calls a memory node's peak is taken over, after one warm call
MEMORY_CALLS = 5
ROLES = ("subject", "reference", "library")


@dataclass(frozen=True)
class Instance:
    label: str
    python: str
    cwd: str
    registry: str
    env: dict = field(default_factory=dict)
    role: str = "subject"


@dataclass(frozen=True)
class Session:
    """Timed arms interleaved on cells. `reference` is the instance label each member's ratios divide by (the member of
    the same arm on the same cell), by default the first instance with a member; `relation` is what the session holds."""

    name: str
    location: str
    arms: tuple[str, ...]
    cells: tuple[str, ...]
    reference: str | None = None
    relation: dict = field(default_factory=dict)
    rounds: int = 8
    reps: int = 11
    warmup: int = WARMUP_FLOOR
    seed: int = 0


class _Worker:
    def __init__(self, instance: Instance) -> None:
        self.instance = instance
        self.process = subprocess.Popen([instance.python, "-m", "rola_devtools.measure.worker", instance.registry],
                                        cwd=instance.cwd, env={**os.environ, **instance.env}, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, text=True, start_new_session=True)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)

    def ask(self, request: dict, what: str) -> dict:
        try:
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as ex:
            raise RuntimeError(f"{what}: the worker is gone (exit {self.process.poll()})") from ex
        if not self.selector.select(REPLY_TIMEOUT_S):
            raise TimeoutError(f"{what}: no reply to {request['op']} within {REPLY_TIMEOUT_S:.0f} s")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"{what}: the worker exited (exit {self.process.wait()})")
        reply = json.loads(line)
        if "error" in reply:
            raise RuntimeError(f"{what}: {request['op']} failed:\n{reply['error']}")
        return reply

    @property
    def alive(self) -> bool:
        return self.process.poll() is None

    def close(self) -> None:
        from ..process import stop_tree

        with contextlib.suppress(BrokenPipeError, ValueError, OSError):
            self.process.stdin.write(json.dumps({"op": "exit"}) + "\n")
            self.process.stdin.flush()
        try:
            self.process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            stop_tree(self.process)
        self.selector.close()
        for pipe in (self.process.stdin, self.process.stdout):
            with contextlib.suppress(OSError):
                pipe.close()


@dataclass(frozen=True)
class _Member:
    instance: Instance
    unit: dict
    cell: dict

    @property
    def id(self) -> str:
        return f"{self.instance.label}:{self.unit['name']}@{self.cell['name']}"


class Service:
    def __init__(self, instances: list[Instance], *, registry=None,
                 hold: Callable[[], contextlib.AbstractContextManager] | None = None,
                 provenance: Callable[[Instance], dict] = lambda instance: {},
                 log: Callable[[str], None] = print) -> None:
        labels = [i.label for i in instances]
        if len(set(labels)) != len(labels):
            raise ValueError(f"instance labels must differ, got {labels}")
        bad = [i.label for i in instances if i.role not in ROLES]
        if bad:
            raise ValueError(f"instances {bad} carry a role outside {ROLES}")
        if registry is None:
            from ..cells import central

            registry = central()
        self.instances, self.registry, self.provenance, self.log = list(instances), registry, provenance, log
        self.hold = hold or self._locks
        from ..build import identity
        from ..cells import FILES

        root = FILES[0].parent
        #: the central registry's code and records as this service reads them: a change to a draw moves every record
        self.draw = identity.files(root, sorted(p.name for p in root.iterdir() if p.suffix in (".py", ".json")))
        self._workers: dict[str, _Worker] = {}
        self._units: dict[str, dict[str, dict]] = {}
        self._clock: list = []
        #: where a result file waits between its post and the store moving it in
        self._results = tempfile.TemporaryDirectory(prefix="rola_measure_results_")

    def __enter__(self) -> Service:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def close(self) -> None:
        for worker in self._workers.values():
            worker.close()
        self._workers.clear()
        self._results.cleanup()

    def worker(self, instance: Instance) -> _Worker:
        worker = self._workers.get(instance.label)
        if worker is None or not worker.alive:
            if worker is not None:
                worker.close()
            worker = self._workers[instance.label] = _Worker(instance)
        return worker

    def restart(self, instance: Instance) -> None:
        worker = self._workers.pop(instance.label, None)
        if worker is not None:
            worker.close()

    # ------------------------------------------------------------------ composition

    def units(self, instance: Instance, cells: Iterable[str]) -> dict[str, dict]:
        """The instance's units, each with what it says of every cell: `on[cell]` is `{"identity"}` or `{"refused"}`."""
        records = [self.registry.cell(name) for name in cells]
        described = self.worker(instance).ask({"op": "describe", "cells": records}, f"{instance.label}: describe")
        self._units[instance.label] = {u["name"]: u for u in described["units"]}
        return self._units[instance.label]

    def nodes(self, cells: Iterable[str], *, sessions: Iterable[Session] = (),
              select: Callable[[Instance, dict], bool] = lambda instance, unit: True, memory: bool = True) -> list[Node]:
        """The build: every selected unit of every instance on every cell it accepts, every arm's memory node when
        `memory`, and every session with a member. `select(instance, unit)` picks the exclusive units; a session's arms
        are its own selection."""
        sessions = list(sessions)
        cells = list(dict.fromkeys([*cells, *(c for s in sessions for c in s.cells)]))
        nodes: dict[str, Node] = {}
        needed: set[str] = set()
        for instance in self.instances:
            units = self.units(instance, cells)
            for unit in units.values():
                if unit["kind"] == "clock":
                    continue
                for cell in (cells if unit["per_cell"] else [""]):
                    if "identity" not in unit["on"][cell]:
                        continue
                    node = self._unit_node(instance, unit, cell)
                    nodes[node.id] = node
                    if select(instance, unit) and (unit["kind"] != "arm" or memory):
                        needed.add(node.id)
        for session in sessions:
            node = self._session_node(session)
            if node is not None:
                nodes[node.id] = node
                needed.add(node.id)
        return [nodes[i] for i in self._closure(nodes, needed)]

    @staticmethod
    def _id(instance: Instance, unit: dict, cell: str, *, memory: bool) -> str:
        at = f"@{cell}" if cell else ""
        return f"{'memory ' if memory else ''}{instance.label}:{unit['name']}{at}"

    def _record(self, cell: str) -> dict | None:
        return self.registry.cell(cell) if cell else None

    def _deps(self, instance: Instance, unit: dict, cell: str, prefix: str = "") -> tuple[tuple[str, str], ...]:
        units = self._units[instance.label]
        deps = []
        for name in unit["deps"]:
            dep = units[name]
            on = cell if dep["per_cell"] else ""
            if dep["kind"] == "arm":
                raise ValueError(f"{instance.label}:{unit['name']} reads the arm {name}; an arm's samples belong to "
                                 "its sessions")
            if "identity" not in dep["on"].get(on, {}):
                raise ValueError(f"{instance.label}:{unit['name']}@{cell} reads {name}, which does not accept {cell}")
            deps.append((prefix + name, self._id(instance, dep, on, memory=False)))
        return tuple(deps)

    def _semantics(self, unit: dict, cell: str) -> dict:
        return {"unit": unit["unit"], "params": unit["params"], "identity": unit["on"][cell]["identity"],
                "cell": self._record(cell), "draw": self.draw if cell else None}

    def _unit_node(self, instance: Instance, unit: dict, cell: str) -> Node:
        """An instrument's or a build's node; for an arm, its memory node (its timed samples are its sessions')."""
        deps = self._deps(instance, unit, cell)
        if unit["kind"] == "arm":
            return Node(self._id(instance, unit, cell, memory=True),
                        {"kind": "memory", **self._semantics(unit, cell), "calls": MEMORY_CALLS}, unit["location"], deps,
                        repeatable=True, payload=("memory", instance, unit, cell))
        return Node(self._id(instance, unit, cell, memory=False), {"kind": unit["kind"], **self._semantics(unit, cell)},
                    unit["location"], deps, repeatable=unit["repeatable"], local=unit["kind"] == "build",
                    payload=(unit["kind"], instance, unit, cell))

    def _session_node(self, session: Session) -> Node | None:
        if session.warmup < WARMUP_FLOOR or session.reps % 2 == 0:
            raise ValueError(f"session {session.name}: warmup at least {WARMUP_FLOOR} and odd reps, got {session.warmup} "
                             f"and {session.reps}")
        members = [_Member(instance, self._units[instance.label][arm], self._record(cell))
                   for instance in self.instances for arm in session.arms
                   if self._units[instance.label].get(arm, {}).get("kind") == "arm"
                   for cell in session.cells if "identity" in self._units[instance.label][arm]["on"][cell]]
        if not members:
            self.log(f"session {session.name}: no instance has an arm of {list(session.arms)} accepting its cells")
            return None
        deps = tuple(dep for i, m in enumerate(members)
                     for dep in self._deps(m.instance, m.unit, m.cell["name"], prefix=f"{i}."))
        semantics = {"kind": "session", "members": [self._semantics(m.unit, m.cell["name"]) for m in members],
                     "timing": {"rounds": session.rounds, "reps": session.reps, "warmup": session.warmup,
                                "seed": session.seed}}
        return Node(f"session {session.name}", semantics, session.location, deps, repeatable=True,
                    payload=("session", session, members))

    @staticmethod
    def _closure(nodes: dict[str, Node], needed: set[str]) -> list[str]:
        out, todo = set(), list(needed)
        while todo:
            cur = todo.pop()
            if cur not in out:
                out.add(cur)
                todo += [dep for _role, dep in nodes[cur].deps]
        return [i for i in nodes if i in out]

    def build(self, store: Callable[[str], object], *, dry: bool = False) -> list[Outcome]:
        """Every instance's builds (and what they read), run unless stored and present: what `nodes` needs before it can
        describe an instance's units on cells."""
        nodes: dict[str, Node] = {}
        needed: set[str] = set()
        for instance in self.instances:
            for unit in self.units(instance, []).values():
                if not unit["per_cell"] and unit["kind"] in ("build", "instrument"):
                    node = self._unit_node(instance, unit, "")
                    nodes[node.id] = node
                    if unit["kind"] == "build":
                        needed.add(node.id)
        return self.run([nodes[i] for i in self._closure(nodes, needed)], store, dry=dry)

    # ------------------------------------------------------------------ execution

    def run(self, nodes: list[Node], store: Callable[[str], object], *, repeat: bool = False, force: bool = False,
            dry: bool = False) -> list[Outcome]:
        return run(nodes, store, self._execute, present=self._present, explain=self._explain, repeat=repeat,
                   force=force, dry=dry, log=self.log)

    def _portable(self, text: str) -> str:
        for instance in self.instances:
            text = text.replace(str(Path(instance.cwd).resolve()) + "/", "")
        return text.replace(str(Path.home()) + "/", "~/")

    def _explain(self, node: Node, ex: BaseException) -> str:
        return self._portable(f"{type(ex).__name__}: {ex}\n{traceback.format_exc()[-2000:]}")

    def _present(self, node: Node, output: Path) -> bool:
        _kind, instance, unit, _cell = node.payload
        reply = self.worker(instance).ask({"op": "present", "name": unit["name"],
                                           "output": json.loads(output.read_text())}, node.id)
        return reply["present"]

    @contextlib.contextmanager
    def _locks(self):
        """The GPU lock, exclusive, and the host's clock lock: engaged once a run, proven after every hold by the first
        instance with a clock reader."""
        from ..locks import clock
        from ..locks.gpu import gpu_lock

        with gpu_lock():
            if not self._clock:
                self._clock.append(clock.engage(self._read_clock))
            yield
            if self._clock[0] is not None:
                ghz = self._read_clock()
                if not clock.within(ghz, self._clock[0]):
                    raise RuntimeError(f"CLOCK: the device reads {ghz} GHz after the hold, off the lock at "
                                       f"{self._clock[0]['ghz']} GHz")

    def _read_clock(self) -> float | None:
        for instance in self.instances:
            reply = self.worker(instance).ask({"op": "clock"}, f"{instance.label}: clock")
            if reply["reader"] is not None:
                return reply["ghz"]
        return None

    def _execute(self, node: Node, deps: dict[str, Path]) -> Executed:
        kind = node.payload[0]
        with tempfile.TemporaryDirectory(prefix="rola_measure_") as tmp:
            if kind == "session":
                return self._interleave(node, deps, Path(tmp))
            _kind, instance, unit, cell = node.payload
            ws = Path(tmp)
            _copy(deps, ws)
            provenance = self.provenance(instance)
            worker, record = self.worker(instance), self._record(cell)
            if kind == "memory":
                with self.hold():
                    reply = worker.ask({"op": "memory", "name": unit["name"], "cell": record, "calls": MEMORY_CALLS,
                                        "ws": str(ws)}, node.id)
                if "refused" in reply:
                    return Executed(refused=self._portable(reply["refused"]), provenance=provenance)
                return Executed(output=reply["memory"], provenance=provenance)
            if kind == "build":
                try:
                    worker.ask({"op": "execute", "id": node.id, "name": unit["name"], "ws": str(ws)}, node.id)
                finally:
                    self.restart(instance)
                return self._post(self.worker(instance), unit, ws, node.id, provenance)
            with self.hold():
                reply = worker.ask({"op": "setup", "id": node.id, "name": unit["name"], "cell": record, "ws": str(ws)},
                                   node.id)
                if "refused" in reply:
                    return Executed(refused=self._portable(reply["refused"]), provenance=provenance)
                try:
                    worker.ask({"op": "execute", "id": node.id, "name": unit["name"], "ws": str(ws)}, node.id)
                finally:
                    worker.ask({"op": "drop", "id": node.id}, node.id)
            return self._post(worker, unit, ws, node.id, provenance)

    def _post(self, worker: _Worker, unit: dict, ws: Path, what: str, provenance: dict) -> Executed:
        result = ws / ".result" / worker.ask({"op": "post", "name": unit["name"], "ws": str(ws)}, what)["result"]
        if result.suffix == ".json":
            return Executed(output=json.loads(result.read_text()), provenance=provenance)
        kept = Path(tempfile.mkdtemp(dir=self._results.name)) / result.name
        shutil.move(str(result), kept)
        return Executed(output_file=str(kept), provenance=provenance)

    def _interleave(self, node: Node, deps: dict[str, Path], root: Path) -> Executed:
        _kind, session, members = node.payload
        provenance = {"members": [self.provenance(i) for i in {m.instance.label: m.instance for m in members}.values()]}
        ws = {m.id: root / f"m{i}" for i, m in enumerate(members)}
        for i, m in enumerate(members):
            ws[m.id].mkdir()
            _copy({role.split(".", 1)[1]: path for role, path in deps.items() if role.startswith(f"{i}.")}, ws[m.id])
        ready: dict[str, dict] = {}
        refused: dict[str, str] = {}
        samples: dict[str, list[float]] = {}
        order: list[str] = []
        rng = random.Random(session.seed)
        with self.hold():
            try:
                for m in members:
                    reply = self.worker(m.instance).ask({"op": "setup", "id": m.id, "name": m.unit["name"],
                                                         "cell": m.cell, "ws": str(ws[m.id])}, m.id)
                    if "refused" in reply:
                        refused[m.id] = self._portable(reply["refused"])
                    else:
                        ready[m.id] = reply["ready"]
                #: THE BARRIER: every member has set up before any timed call
                live = [m for m in members if m.id in ready]
                instruments = {ready[m.id]["instrument"] for m in live}
                if len(instruments) > 1:
                    raise ValueError(f"one session, one stopwatch: its members time with {sorted(instruments)}")

                def call(m: _Member) -> float:
                    return self.worker(m.instance).ask({"op": "call", "id": m.id}, m.id)["ms"]

                for _ in range(session.warmup):
                    for m in live:
                        call(m)
                samples = {m.id: [] for m in live}
                for _ in range(session.rounds * session.reps):
                    for m in rng.sample(live, len(live)):
                        samples[m.id].append(call(m))
                        order.append(m.id)
            finally:
                for m in members:
                    if m.id in ready and self.worker(m.instance).alive:
                        self.worker(m.instance).ask({"op": "drop", "id": m.id}, m.id)
        if not live:
            return Executed(refused=json.dumps(refused, sort_keys=True), provenance=provenance)

        def blocks(values: list[float]) -> list[float]:
            return [statistics.median(values[i:i + session.reps]) for i in range(0, len(values), session.reps)]

        reference = session.reference or live[0].instance.label
        partner = {(m.instance.label, m.unit["name"], m.cell["name"]): m.id for m in live}
        rows = []
        for m in live:
            (ws[m.id] / "samples.json").write_text(json.dumps({"ms": samples[m.id], "built": ready[m.id]["built"]}))
            post = self._post(self.worker(m.instance), m.unit, ws[m.id], m.id, {})
            row = {"member": m.id, "label": m.instance.label, "role": m.instance.role, "arm": m.unit["name"],
                   "unit": m.unit["unit"], "cell": m.cell["name"], "built": ready[m.id]["built"], "ms": samples[m.id],
                   "blocks_ms": blocks(samples[m.id]), "median_ms": statistics.median(blocks(samples[m.id])),
                   "iqr_ms": iqr(blocks(samples[m.id])), "post": post.output, "paired": []}
            other = partner.get((reference, m.unit["name"], m.cell["name"]))
            if other is not None and other != m.id:
                ratios = [a / r for a, r in zip(samples[m.id], samples[other], strict=True)]
                row["paired"].append({"reference": other, "ratios": ratios, "ratio_median": statistics.median(ratios),
                                      "ratio_iqr": iqr(ratios),
                                      "round_diffs_ms": [a - r for a, r in zip(blocks(samples[m.id]), blocks(samples[other]),
                                                                              strict=True)]})
            rows.append(row)
        return Executed(output={"session": session.name, "instrument": instruments.pop(), "reference": reference,
                                "rounds": session.rounds, "reps": session.reps, "warmup": session.warmup,
                                "seed": session.seed, "order": order, "members": rows, "refusals": refused,
                                "relation": session.relation},
                        wall_s=None, provenance=provenance)


def _copy(deps: dict[str, Path], ws: Path) -> None:
    if not deps:
        return
    (ws / "deps").mkdir(parents=True, exist_ok=True)
    for role, path in deps.items():
        shutil.copyfile(path, ws / "deps" / f"{role}{Path(path).suffix}")


def outcomes_line(outcomes: list[Outcome]) -> str:
    counts: dict[str, int] = {}
    for outcome in outcomes:
        counts[outcome.status] = counts.get(outcome.status, 0) + 1
    return ", ".join(f"{n} {status}" for status, n in sorted(counts.items()))


__all__ = ["MEMORY_CALLS", "ROLES", "Instance", "Service", "Session", "outcomes_line"]
