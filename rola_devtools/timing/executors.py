"""THE TIMING TARGETS' EXECUTORS. `register` and `register_clock` run in a checkout's environment; the rest in the
build system's own, sharing the server (`pool.SERVER`) that start_server leaves in that worker."""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

from ..build.resources import MARKERS
from ..build.worker import load
from ..locks import clock
from . import pool
from .entry_worker import _draw


def _here(ctx) -> dict:
    return {"label": ctx.label, "python": sys.executable, "cwd": os.getcwd(), "pythonpath": os.environ.get("PYTHONPATH")}


def _portable(text: str) -> str:
    return text.replace(str(Path.home()) + "/", "~/")


def _markers() -> dict:
    return {marker: os.environ[marker] for marker in MARKERS if marker in os.environ}


def start_server(ctx) -> dict:
    if pool.SERVER:
        pool.SERVER[0].close()
    pool.SERVER[:] = [pool.Pool()]
    return {"server": "started"}


def stop_server(ctx) -> dict:
    stopped = bool(pool.SERVER)
    if pool.SERVER:
        pool.SERVER[0].close()
        pool.SERVER.clear()
    return {"stopped": stopped}


def register(ctx) -> dict:
    """One entry per input cell. It imports the entry executor, so a registration that names code this checkout lacks
    fails here, at the build layer; it builds nothing."""
    load(ctx.params["executor"])
    return {"entries": [{"cell": record["name"], "record": record, "executor": ctx.params["executor"],
                         "params": ctx.params["params"]} for record in ctx.inputs],
            "local": {"owner": ctx.label, "env": _here(ctx)}}


def register_clock(ctx) -> dict:
    load(ctx.params["executor"])
    return {"clock": ctx.params["executor"], "local": {"owner": ctx.label, "env": _here(ctx)}}


def _entries(ctx) -> tuple[list[dict], object]:
    entries, reader = [], None
    for role, dep in sorted(ctx.deps.items()):
        if role == "clock":
            reader = dep
        elif role.startswith("r"):
            entries += [{**e, "owner": dep.output["local"]["owner"], "env": dep.output["local"]["env"]}
                        for e in dep.output["entries"]]
    wanted = ctx.params.get("cells")
    return ([e for e in entries if wanted is None or e["cell"] in wanted], reader)


def _read_clock(srv, reader, markers):
    if reader is None:
        return None
    return srv.worker(reader.output["local"]["env"]).ask(
        {"op": "clock", "executor": reader.output["clock"], "env": markers}, "the clock reader")["ghz"]


def _session(ctx, entries: list[dict], reader) -> dict:
    """One interleaved session over `entries`, kept in the workspace's session.json. Raises for what fails the build: a
    clock off the lock, two stopwatches, a worker lost."""
    srv, markers, cfg = pool.server(), _markers(), clock.load()
    if cfg is not None and reader is None:
        raise RuntimeError("this host locks its clock and the session has no clock reader to prove it (register_clock_reader)")
    before = _read_clock(srv, reader, markers)
    if cfg is not None and not clock.within(before, cfg):
        raise RuntimeError(f"CLOCK: the device reads {before} GHz before the session, off the lock at {cfg['ghz']} GHz")
    p, draw = ctx.params, _draw()
    members, set_up, instruments = [], [], []
    try:
        for i, entry in enumerate(entries):
            member = {"id": f"m{i}", "owner": entry["owner"], "cell": entry["cell"], "executor": entry["executor"]}
            members.append(member)
            worker = srv.worker(entry["env"])
            reply = worker.ask({"op": "setup", "id": f"{ctx.run}:{ctx.label}:{i}", "executor": entry["executor"],
                                "cell": entry["record"], "params": entry["params"], "env": markers}, member["id"])
            if "failed" in reply:
                member.update(status="failed", error=_portable(reply["failed"]))
                continue
            set_up.append((member, worker, f"{ctx.run}:{ctx.label}:{i}"))
            if reply["ready"]["draw"] != draw:
                member.update(status="failed", error="the checkout's rola_devtools draws cells with other code than the "
                                                     "build system's")
                continue
            member.update(status="ok", built=reply["ready"]["built"], instrument=reply["ready"]["instrument"])
        #: THE BARRIER: every entry has set up before any timed call
        live = [(m, w, i) for m, w, i in set_up if m["status"] == "ok"]
        instruments = sorted({m["instrument"] for m, _w, _i in live})
        if len(instruments) > 1:
            raise RuntimeError(f"one session, one stopwatch: its entries time with {instruments}")
        for _ in range(p["warmup"]):
            for _m, worker, ident in live:
                worker.ask({"op": "call", "id": ident, "env": markers}, "warmup")
        rng, samples = random.Random(p["seed"]), []
        for rnd in range(p["rounds"]):
            for rep in range(p["reps"]):
                for position, (member, worker, ident) in enumerate(rng.sample(live, len(live))):
                    ms = worker.ask({"op": "call", "id": ident, "env": markers}, member["id"])["ms"]
                    samples.append({"member": member["id"], "round": rnd, "rep": rep, "position": position, "ms": ms})
    finally:
        for _m, worker, ident in set_up:
            if worker.alive:
                worker.ask({"op": "drop", "id": ident}, "drop")
    after = _read_clock(srv, reader, markers)
    session = {"session": ctx.label, "run": ctx.run, "claim": p.get("claim", ""), "instrument": instruments[0] if instruments
               else None, "rounds": p["rounds"], "reps": p["reps"], "warmup": p["warmup"], "seed": p["seed"],
               "clock": {"declared": cfg["ghz"] if cfg else None, "before": before, "after": after},
               "members": members, "samples": samples}
    (ctx.workspace / "session.json").write_text(json.dumps(session, sort_keys=True))
    if cfg is not None and not clock.within(after, cfg):
        raise RuntimeError(f"CLOCK: the device reads {after} GHz after the session, off the lock at {cfg['ghz']} GHz; "
                           "its samples are kept in session.json and stored by nothing")
    return session


def measure(ctx) -> dict:
    entries, reader = _entries(ctx)
    session = _session(ctx, entries, reader)
    return {"members": [{k: m.get(k) for k in ("id", "owner", "cell", "status", "error")} for m in session["members"]],
            "samples": len(session["samples"]), "file": "session.json",
            "local": {"envs": {e["owner"]: e["env"] for e in entries}}}


def _quartiles(values: list[float]) -> tuple[float, float, float]:
    ordered = sorted(values)
    return ordered[int(0.25 * len(ordered))], ordered[len(ordered) // 2], ordered[int(0.75 * len(ordered))]


def null_gate(ctx) -> dict:
    """ONE ENTRY IN TWO WORKERS: each entry the registration makes, set up twice -- `#a` and `#b`, each in its own worker
    of the entry's environment -- and timed as one session. On a cell the gate is `trusted` when the per-rep ratios of
    the two copies, a over b, put one inside their interquartile range: a comparison across workers then carries no
    worker's bias. An untrusted cell is the gate's finding, recorded in its output; the build goes on."""
    entries, reader = _entries(ctx)
    copies = [{**e, "owner": f"{e['owner']}#{copy}", "env": {**e["env"], "instance": copy}} for e in entries for copy in "ab"]
    session = _session(ctx, copies, reader)
    ids = {(m["owner"], m["cell"]): m for m in session["members"]}
    ms = {(s["member"], s["round"], s["rep"]): s["ms"] for s in session["samples"]}
    cells = {}
    for e in entries:
        a, b = ids[(f"{e['owner']}#a", e["cell"])], ids[(f"{e['owner']}#b", e["cell"])]
        if a["status"] != "ok" or b["status"] != "ok":
            cells[e["cell"]] = {"trusted": None, "error": a.get("error") or b.get("error")}
            continue
        ratios = [ms[(a["id"], r, k)] / ms[(b["id"], r, k)] for r in range(session["rounds"]) for k in range(session["reps"])
                  if ms[(b["id"], r, k)] > 0]
        q1, median, q3 = _quartiles(ratios)
        cells[e["cell"]] = {"trusted": q1 <= 1.0 <= q3, "ratio_median": median, "ratio_q1": q1, "ratio_q3": q3}
    return {"members": [{k: m.get(k) for k in ("id", "owner", "cell", "status", "error")} for m in session["members"]],
            "samples": len(session["samples"]), "cells": cells, "file": "session.json",
            "local": {"envs": {e["owner"]: e["env"] for e in entries}}}


def memory(ctx) -> dict:
    srv, markers = pool.server(), _markers()
    entries, _reader = _entries(ctx)
    rows = []
    for entry in entries:
        reply = srv.worker(entry["env"]).ask({"op": "memory", "executor": entry["executor"], "cell": entry["record"],
                                              "params": entry["params"], "calls": ctx.params["calls"], "env": markers},
                                             f"{entry['owner']}@{entry['cell']}")
        row = {"owner": entry["owner"], "cell": entry["cell"], "executor": entry["executor"]}
        row.update({"status": "failed", "error": _portable(reply["failed"])} if "failed" in reply
                   else {"status": "ok", **reply["memory"]})
        rows.append(row)
    (ctx.workspace / "memory.json").write_text(json.dumps({"run": ctx.run, "rows": rows}, sort_keys=True))
    return {"members": [{k: r.get(k) for k in ("owner", "cell", "status", "error")} for r in rows], "file": "memory.json",
            "local": {"envs": {e["owner"]: e["env"] for e in entries}}}
