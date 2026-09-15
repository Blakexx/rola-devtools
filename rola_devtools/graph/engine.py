"""THE ENGINE: every instance's nodes in dependency order, what is stored skipped, what runs grouped into sessions.

    instances = [engine.load(env) for env in envs]            # each graph, described in its own environment
    engine.run(instances, store, sessions, hold=gpu_and_clock)

An EXCLUSIVE node (not timed) is a session of its own: its worker is started, its setup runs, then its execute, and the
worker is closed; its post runs in a fresh process in its own environment and its record is stored at its location. A
TIMED node runs in each `Session` that names it, beside its other members (a timed node no session names runs alone): every
member's setup in the workers of their environments, then a barrier, then warmup and interleaved rounds of calls in a
fresh random order each rep, then the workers closed, then each member's post over its samples, then the session's
record -- every member's samples, its post and its pairings to the reference member -- stored at the session's location.
Setups and executes run under `hold` (the caller's device and clock locks); posts run after it is released.

A node's record semantics are its unit, parameters, identity and each dependency's key and output digest; a session's are
its members' semantics in order and its timing parameters. A refusal is stored as a successful sample whose output is
`{"refused": why}` (so it is not retried until its identity changes), and a node depending on a refused or failed node is
BLOCKED and stays unrecorded. A session member that refuses leaves the session, which records why.

`store(location)` returns the caller's store for a location: `get(key)`, `put(semantics, output=|error=, wall_s=,
provenance=)` and `output(record)` (rola-results' `Store`). Keys are sha256 over the semantics as canonical JSON, the first
24 hex digits, which is the store's own formula.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import selectors
import shutil
import statistics
import subprocess
import tempfile
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..interleave.driver import WARMUP_FLOOR, iqr
from ..process import run as run_command
from . import Env

#: seconds a worker may take to answer one request before its session fails
REPLY_TIMEOUT_S = 3600.0


def key(semantics: dict) -> str:
    return hashlib.sha256(json.dumps(semantics, sort_keys=True).encode()).hexdigest()[:24]


@dataclass(frozen=True)
class Instance:
    env: Env
    nodes: tuple[dict, ...]

    def qualified(self, name: str) -> str:
        return f"{self.env.label}:{name}"


@dataclass(frozen=True)
class Session:
    """Timed members interleaved together. `members` are qualified node names (`label:name`). `reference` is the instance
    label the paired ratios divide by (the first member's by default): each member pairs with that instance's node of the
    same name, when the session holds one. `relation` is what the session holds (roles, what is equal). The reference and
    the relation are recorded with the session and never keyed: pairing is a reading of the samples."""

    name: str
    location: str
    members: tuple[str, ...]
    reference: str | None = None
    rounds: int = 8
    reps: int = 11
    warmup: int = WARMUP_FLOOR
    seed: int = 0
    relation: dict = field(default_factory=dict)


@dataclass
class Outcome:
    name: str
    key: str
    status: str  #: complete | ran | refused | failed | blocked | pending
    wall_s: float = 0.0
    error: str = ""


def load(env: Env, timeout: float = 1800.0) -> Instance:
    """The instance of `env.graph`: its nodes described by its own environment."""
    done = run_command([env.python, "-m", "rola_devtools.graph.worker", "describe", env.graph], cwd=env.cwd,
                       env={**_os_environ(), **env.env}, timeout=timeout)
    if done.returncode:
        raise RuntimeError(f"{env.label}: describing {env.graph} failed (exit {done.returncode}):\n{done.stderr[-2000:]}")
    return Instance(env, tuple(json.loads(done.stdout.strip().splitlines()[-1])))


def _os_environ() -> dict:
    return dict(os.environ)


def _portable(text: str, env: Env) -> str:
    """A message with the instance's directory and the home directory taken out of its paths: a store refuses a
    machine path."""
    return text.replace(str(Path(env.cwd).resolve()) + "/", "").replace(str(Path.home()) + "/", "~/")


class _Worker:
    def __init__(self, env: Env) -> None:
        self.env = env
        self.process = subprocess.Popen([env.python, "-m", "rola_devtools.graph.worker", "serve"], cwd=env.cwd,
                                        env={**_os_environ(), **env.env}, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, start_new_session=True)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)

    def ask(self, request: dict, what: str) -> dict:
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        if not self.selector.select(REPLY_TIMEOUT_S):
            raise TimeoutError(f"{what}: no reply to {request['op']} within {REPLY_TIMEOUT_S:.0f} s")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"{what}: the worker exited (exit {self.process.wait()})")
        reply = json.loads(line)
        if "error" in reply:
            raise RuntimeError(f"{what}: {request['op']} failed:\n{reply['error']}")
        return reply

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


def _post(env: Env, node: dict, ws: Path) -> dict:
    done = run_command([env.python, "-m", "rola_devtools.graph.worker", "post", node["unit"], json.dumps(node["params"]),
                        str(ws)], cwd=env.cwd, env={**_os_environ(), **env.env}, timeout=REPLY_TIMEOUT_S)
    if done.returncode or not (ws / "post.json").exists():
        raise RuntimeError(f"{env.label}:{node['name']}: post failed (exit {done.returncode}):\n{done.stderr[-2000:]}")
    return json.loads((ws / "post.json").read_text())


def _complete(record: dict | None) -> bool:
    return bool(record) and any(s["ok"] for s in record["samples"])


def _order(instances: list[Instance]) -> list[tuple[Instance, dict]]:
    """Every node, dependencies first within its instance."""
    out: list[tuple[Instance, dict]] = []
    for inst in instances:
        by_name = {n["name"]: n for n in inst.nodes}
        state: dict[str, str] = {}

        def visit(node: dict, inst=inst, by_name=by_name, state=state) -> None:
            if state.get(node["name"]) == "done":
                return
            if state.get(node["name"]) == "open":
                raise ValueError(f"{inst.env.label}: a dependency cycle through {node['name']}")
            state[node["name"]] = "open"
            for dep in node["deps"]:
                if by_name[dep]["timed"]:
                    raise ValueError(f"{inst.qualified(node['name'])} depends on the timed node {dep}; a timed node's "
                                     "samples belong to its session")
                visit(by_name[dep])
            state[node["name"]] = "done"
            out.append((inst, node))

        for node in inst.nodes:
            visit(node)
    return out


def _semantics(node: dict, deps: dict[str, tuple[str, str]]) -> dict:
    return {"unit": node["unit"], "params": node["params"], "identity": node["identity"],
            "deps": {d: {"key": k, "output": o} for d, (k, o) in sorted(deps.items())}}


def _digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _refused(path: Path) -> str | None:
    if path.suffix != ".json":
        return None
    doc = json.loads(path.read_text())
    return doc.get("refused") if isinstance(doc, dict) else None


def run(instances: list[Instance], store: Callable[[str], object], sessions: list[Session] = (), *,
        hold: Callable[[], contextlib.AbstractContextManager] = contextlib.nullcontext, select: set[str] | None = None,
        repeat: bool = False, force: bool = False, dry: bool = False, provenance: Callable[[Env], dict] = lambda env: {},
        log: Callable[[str], None] = print) -> list[Outcome]:
    """Run every selected node that is not stored (with `repeat`, repeatable ones again; with `force`, all), and every
    session with a selected member. `select` holds qualified names; their dependencies are selected with them."""
    ordered = _order(instances)
    nodes = {inst.qualified(n["name"]): (inst, n) for inst, n in ordered}
    timed = {q for q, (_i, n) in nodes.items() if n["timed"]}
    leaning = sorted(q for q in timed if nodes[q][1]["deps"])
    if leaning:
        raise ValueError(f"timed nodes {leaning} declare dependencies; a timed node's identity carries what it runs")
    named = [m for s in sessions for m in s.members]
    bad = sorted(set(named) - timed)
    if bad:
        raise ValueError(f"sessions name {bad}, which are not timed nodes of the instances")
    doubled = [s.name for s in sessions if len(set(s.members)) != len(s.members)]
    if doubled:
        raise ValueError(f"sessions {doubled} name a member twice")
    sessions = list(sessions) + [Session(q, nodes[q][1]["location"], (q,)) for q in sorted(timed - set(named))]

    wanted = set(nodes) if select is None else set()
    for q in select or ():
        if q not in nodes:
            raise KeyError(f"no node {q}")
        todo = [q]
        while todo:
            cur = todo.pop()
            if cur not in wanted:
                wanted.add(cur)
                inst, n = nodes[cur]
                todo += [inst.qualified(d) for d in n["deps"]]

    outputs: dict[str, Path] = {}
    keys: dict[str, tuple[str, str]] = {}
    outcomes: list[Outcome] = []
    for q, (inst, node) in nodes.items():
        if node["timed"] or q not in wanted:
            continue
        deps = [inst.qualified(d) for d in node["deps"]]
        missing = [d for d in deps if d not in outputs or _refused(outputs[d])]
        if missing:
            outcomes.append(Outcome(q, "", "blocked", error=f"no usable result for {', '.join(missing)}"))
            log(f"blocked   {q} (no usable result for {', '.join(missing)})")
            continue
        semantics = _semantics(node, {d.split(":", 1)[1]: keys[d] for d in deps})
        k, st = key(semantics), store(node["location"])
        record = st.get(k)
        if _complete(record):
            outputs[q] = st.output(record)
            keys[q] = (k, _digest(outputs[q]))
            if not force and not (repeat and node["repeatable"]):
                status = "refused" if _refused(outputs[q]) is not None else "complete"
                outcomes.append(Outcome(q, k, status, error=_refused(outputs[q]) or ""))
                log(f"{status:9s} {q} [{k[:12]}]" + (f": {_refused(outputs[q])}" if status == "refused" else ""))
                continue
        if dry:
            outcomes.append(Outcome(q, k, "pending"))
            log(f"pending   {q} [{k[:12]}]")
            continue
        outcome, path = _exclusive(inst, node, q, k, semantics, st, outputs, deps, hold, provenance)
        outcomes.append(outcome)
        log(f"{outcome.status:9s} {q} [{k[:12]}] {outcome.wall_s:.1f}s"
            + (f": {outcome.error[-200:]}" if outcome.error else ""))
        if path is not None:
            outputs[q] = path
            keys[q] = (k, _digest(path))

    for session in sessions:
        if select is not None and not set(session.members) & wanted:
            continue
        outcomes.append(_session(session, nodes, store, hold, repeat, force, dry, provenance, log))
    return outcomes


def _copy_deps(ws: Path, deps: list[str], outputs: dict[str, Path]) -> None:
    (ws / "deps").mkdir(parents=True, exist_ok=True)
    for dep in deps:
        shutil.copyfile(outputs[dep], ws / "deps" / f"{dep.split(':', 1)[1]}{outputs[dep].suffix}")


def _exclusive(inst, node, q, k, semantics, st, outputs, deps, hold, provenance) -> tuple[Outcome, Path | None]:
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="rola_graph_") as tmp:
        ws = Path(tmp)
        try:
            _copy_deps(ws, deps, outputs)
            refused = None
            with hold():
                worker = _Worker(inst.env)
                try:
                    reply = worker.ask({"op": "setup", "node": node["name"], "unit": node["unit"],
                                        "params": node["params"], "ws": str(ws)}, q)
                    refused = None if reply.get("refused") is None else _portable(reply["refused"], inst.env)
                    if refused is None:
                        worker.ask({"op": "execute", "node": node["name"], "ws": str(ws)}, q)
                finally:
                    worker.close()
            output = {"refused": refused} if refused is not None else _post(inst.env, node, ws)
        except Exception as ex:  # noqa: BLE001 -- recorded as the node's failed sample
            wall = time.time() - started
            message = _portable(f"{type(ex).__name__}: {ex}\n{traceback.format_exc()[-2000:]}", inst.env)
            st.put(semantics, error=message, wall_s=wall, provenance=provenance(inst.env))
            return Outcome(q, k, "failed", wall, _portable(str(ex), inst.env)), None
        wall = time.time() - started
        st.put(semantics, output=output, wall_s=wall, provenance=provenance(inst.env))
    record = st.get(k)
    return Outcome(q, k, "refused" if refused is not None else "ran", wall), st.output(record)


def _session(session: Session, nodes, store, hold, repeat, force, dry, provenance, log) -> Outcome:
    members = [nodes[q] for q in session.members]
    semantics = {"members": [_semantics(n, {}) for _i, n in members],
                 "timing": {"rounds": session.rounds, "reps": session.reps, "warmup": session.warmup,
                            "seed": session.seed}}
    k, st = key(semantics), store(session.location)
    record = st.get(k)
    if _complete(record) and not force and not repeat:
        log(f"complete  session {session.name} [{k[:12]}]")
        return Outcome(session.name, k, "complete")
    if dry:
        log(f"pending   session {session.name} [{k[:12]}] ({len(members)} members)")
        return Outcome(session.name, k, "pending")
    if session.warmup < WARMUP_FLOOR or session.reps % 2 == 0:
        raise ValueError(f"session {session.name}: warmup at least {WARMUP_FLOOR} and odd reps, got "
                         f"{session.warmup} and {session.reps}")
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="rola_session_") as tmp:
        try:
            doc = _interleave(session, nodes, Path(tmp), hold)
        except Exception as ex:  # noqa: BLE001 -- a crash or a timeout fails the whole session
            wall = time.time() - started
            message = f"{type(ex).__name__}: {ex}\n{traceback.format_exc()[-2000:]}"
            for inst, _n in members:
                message = _portable(message, inst.env)
            st.put(semantics, error=message, wall_s=wall, provenance={"members": [provenance(i.env) for i, _n in members]})
            log(f"failed    session {session.name} [{k[:12]}] {wall:.1f}s: {str(ex)[-200:]}")
            return Outcome(session.name, k, "failed", wall, str(ex))
        wall = time.time() - started
        st.put(semantics, output={**doc, "relation": session.relation}, wall_s=wall,
               provenance={"members": [provenance(i.env) for i, _n in members]})
    status = "ran" if doc["members"] else "refused"
    log(f"{status:9s} session {session.name} [{k[:12]}] {wall:.1f}s" + ("" if doc["members"] else f": {doc['refused']}"))
    return Outcome(session.name, k, status, wall, "" if doc["members"] else json.dumps(doc["refused"]))


def _interleave(session: Session, nodes, root: Path, hold) -> dict:
    ws = {q: root / f"m{i}" for i, q in enumerate(session.members)}
    for path in ws.values():
        path.mkdir()
    ready: dict[str, dict] = {}
    refused: dict[str, str] = {}
    rng = random.Random(session.seed)
    samples: dict[str, list[float]] = {}
    order: list[str] = []
    with hold():
        workers: dict[tuple, _Worker] = {}
        try:
            for q in session.members:
                inst, node = nodes[q]
                env_key = (inst.env.python, inst.env.cwd, tuple(sorted(inst.env.env.items())))
                if env_key not in workers:
                    workers[env_key] = _Worker(inst.env)
                reply = workers[env_key].ask({"op": "setup", "node": q, "unit": node["unit"], "params": node["params"],
                                              "ws": str(ws[q])}, q)
                if "refused" in reply:
                    refused[q] = _portable(reply["refused"], inst.env)
                else:
                    ready[q] = {**reply["ready"], "worker": env_key}
            #: THE BARRIER: every member has set up before any timed call
            live = [q for q in session.members if q in ready]
            instruments = {ready[q]["instrument"] for q in live}
            if len(instruments) > 1:
                raise ValueError(f"one session, one stopwatch: its members time with {sorted(instruments)}")

            def call(q: str) -> float:
                return workers[ready[q]["worker"]].ask({"op": "call", "node": q}, q)["ms"]

            for _ in range(session.warmup):
                for q in live:
                    call(q)
            samples = {q: [] for q in live}
            for _ in range(session.rounds * session.reps):
                for q in rng.sample(live, len(live)):
                    samples[q].append(call(q))
                    order.append(q)
        finally:
            for worker in workers.values():
                worker.close()

    def blocks(values: list[float]) -> list[float]:
        return [statistics.median(values[i:i + session.reps]) for i in range(0, len(values), session.reps)]

    reference = session.reference or (session.members[0].split(":", 1)[0] if session.members else None)
    rows = []
    for q in live:
        inst, node = nodes[q]
        (ws[q] / "samples.json").write_text(json.dumps({"ms": samples[q], "built": ready[q]["built"]}))
        row = {"member": q, "label": inst.env.label, "node": node["name"], "unit": node["unit"],
               "location": node["location"], "built": ready[q]["built"], "ms": samples[q],
               "blocks_ms": blocks(samples[q]), "median_ms": statistics.median(blocks(samples[q])),
               "iqr_ms": iqr(blocks(samples[q])), "post": _post(inst.env, node, ws[q]), "paired": []}
        partner = f"{reference}:{node['name']}"
        if partner != q and partner in samples:
            ratios = [a / r for a, r in zip(samples[q], samples[partner], strict=True)]
            row["paired"].append({"reference": partner, "ratios": ratios, "ratio_median": statistics.median(ratios),
                                  "ratio_iqr": iqr(ratios),
                                  "round_diffs_ms": [a - r for a, r in zip(blocks(samples[q]), blocks(samples[partner]),
                                                                          strict=True)]})
        rows.append(row)
    return {"session": session.name, "instrument": instruments.pop() if live else None, "reference": reference,
            "rounds": session.rounds, "reps": session.reps, "warmup": session.warmup, "seed": session.seed,
            "order": order, "members": rows, "refused": refused}
