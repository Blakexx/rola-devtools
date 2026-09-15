"""THE INSTANCE WORKER: one owner's registry in its own environment, serving the measurement service for a whole run.

    python -m rola_devtools.measure.worker module:registry

One worker runs per instance, so an owner's device state -- its extension loaded, its kernels compiled, its CUDA context
-- lives for the run, and what one node prepared is dropped before the next is held. The protocol is JSON lines on the
process's original stdout; anything a unit or a native library prints goes to stderr. Cells arrive as central registry
records (`{name, data, params}`), which the worker builds with its own `rola_devtools.cells.build`.

    {"op": "describe", "cells": [RECORD]}          ->  {"units": [{name, kind, unit, params, deps, location, repeatable,
                                                        per_cell, "on": {CELL | "": {"identity": ...} | {"refused": WHY}}}]}
    {"op": "setup", "id", "name", "cell", "ws"}    ->  {"ready": {"built", "instrument"} | {}} | {"refused": WHY}
    {"op": "execute", "id", "name", "ws"}          ->  {"done": true}      (an instrument's prepared node, or a build)
    {"op": "call", "id"}                           ->  {"ms": MS}
    {"op": "memory", "name", "cell", "calls", "ws"} ->  {"memory": {...}} | {"refused": WHY}
    {"op": "drop", "id"}                           ->  {"dropped": true}   (the prepared state released, the cache emptied)
    {"op": "post", "name", "ws"}                   ->  {"result": FILE}    (under WS/.result)
    {"op": "present", "name", "output"}            ->  {"present": BOOL}
    {"op": "clock"}                                ->  {"ghz": GHZ | null, "reader": NAME | null}
    {"op": "exit"}                                 ->  (the process exits)

A failure replies {"error": "<the traceback's tail>"} and leaves the worker serving.
"""
from __future__ import annotations

import gc
import importlib
import json
import os
import sys
import traceback
from pathlib import Path

from . import KINDS, Handle, Refusal, Registration, Timed, Unit


def load(ref: str):
    module, _, name = ref.partition(":")
    if not module or not name:
        raise ValueError(f"expected module:name, got {ref!r}")
    return getattr(importlib.import_module(module), name)


class Registry:
    """An owner's registrations and their units, each built once."""

    def __init__(self, ref: str) -> None:
        self.registrations: dict[str, Registration] = {}
        for reg in load(ref)():
            if not isinstance(reg, Registration):
                raise TypeError(f"{ref} returned {type(reg).__name__}, not a Registration")
            if reg.name in self.registrations:
                raise ValueError(f"{ref} registers {reg.name} twice")
            self.registrations[reg.name] = reg
        missing = sorted({d for reg in self.registrations.values() for d in reg.deps} - set(self.registrations))
        if missing:
            raise ValueError(f"{ref}: registrations read {missing}, which it does not register")
        self.units: dict[str, Unit] = {}
        for name, reg in self.registrations.items():
            unit = load(reg.unit)(**reg.params)
            if unit.kind not in KINDS:
                raise TypeError(f"{reg.unit} is not an Arm, Instrument, Build or ClockReader")
            if unit.kind != "clock" and not unit.location:
                raise ValueError(f"{reg.unit} declares no location")
            if unit.kind == "arm" and not unit.per_cell:
                raise ValueError(f"{reg.unit}: an arm runs on cells")
            self.units[name] = unit


def describe(registry: Registry, records: list[dict]) -> list[dict]:
    from rola_devtools.cells import build

    cells = {record["name"]: build(record) for record in records}
    out = []
    for name, unit in registry.units.items():
        reg = registry.registrations[name]
        on: dict[str, dict] = {}
        if unit.kind != "clock":
            for cell_name, cell in (cells.items() if unit.per_cell else [("", None)]):
                why = unit.accepts(cell)
                on[cell_name] = {"refused": why} if why is not None else {"identity": unit.identity(cell)}
        out.append({"name": name, "kind": unit.kind, "unit": reg.unit, "params": reg.params, "deps": list(reg.deps),
                    "location": unit.location, "repeatable": unit.repeatable, "per_cell": unit.per_cell, "on": on})
    return out


def _cell(record: dict | None):
    if record is None:
        return None
    from rola_devtools.cells import build

    return build(record)


def _release() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def memory(unit: Unit, cell, calls: int, ws: Path) -> dict:
    """The arm alone: torch's caching allocator's peak allocated and reserved bytes over `calls` calls after one warm
    call, what stays allocated after them, what was allocated before the arm was built, and the bytes the arm holds
    outside the allocator (which only grow during a sequence, so their value after the calls is their peak)."""
    import torch

    _release()
    before = torch.cuda.memory_allocated()
    timed = unit.setup(cell, ws)
    try:
        timed.call()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(calls):
            timed.call()
        torch.cuda.synchronize()
        return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "allocated_after_bytes": torch.cuda.memory_allocated(), "allocated_before_build_bytes": before,
                "outside_allocator_bytes": int(timed.outside_allocator()) if timed.outside_allocator else 0,
                "calls": calls, "built": timed.built}
    finally:
        del timed
        _release()


def serve(ref: str) -> None:
    replies = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    registry: Registry | None = None
    prepared: dict[str, object] = {}
    for line in sys.stdin:
        request = json.loads(line)
        op = request["op"]
        if op == "exit":
            return
        try:
            if registry is None:
                registry = Registry(ref)
            if op == "describe":
                reply = {"units": describe(registry, request["cells"])}
            elif op == "setup":
                unit = registry.units[request["name"]]
                try:
                    ready = unit.setup(_cell(request["cell"]), Path(request["ws"]))
                except Refusal as refusal:
                    reply = {"refused": str(refusal)}
                else:
                    if unit.kind == "arm" and not isinstance(ready, Timed):
                        raise TypeError(f"{request['name']}: an arm's setup returns a Timed, got {type(ready).__name__}")
                    prepared[request["id"]] = ready
                    reply = {"ready": {"built": ready.built, "instrument": ready.instrument} if unit.kind == "arm" else {}}
            elif op == "execute":
                unit = registry.units[request["name"]]
                if unit.kind == "build":
                    unit.execute(Path(request["ws"]))
                else:
                    unit.execute(prepared[request["id"]], Path(request["ws"]))
                reply = {"done": True}
            elif op == "call":
                reply = {"ms": float(prepared[request["id"]].call())}
            elif op == "memory":
                try:
                    reply = {"memory": memory(registry.units[request["name"]], _cell(request["cell"]), request["calls"],
                                              Path(request["ws"]))}
                except Refusal as refusal:
                    reply = {"refused": str(refusal)}
            elif op == "drop":
                prepared.pop(request["id"], None)
                _release()
                reply = {"dropped": True}
            elif op == "post":
                ws = Path(request["ws"])
                handle = Handle(ws)
                registry.units[request["name"]].post(ws, handle)
                written = sorted(p.name for p in handle.dir.iterdir())
                if len(written) != 1:
                    raise RuntimeError(f"{request['name']}: its post wrote {written or 'nothing'} through its handle")
                reply = {"result": written[0]}
            elif op == "present":
                reply = {"present": bool(registry.units[request["name"]].present(request["output"]))}
            elif op == "clock":
                readers = [name for name, unit in registry.units.items() if unit.kind == "clock"]
                reply = {"ghz": registry.units[readers[0]].read_ghz() if readers else None,
                         "reader": readers[0] if readers else None}
            else:
                raise ValueError(f"unknown op {op!r}")
        except Exception:  # noqa: BLE001 -- the service names the node and fails it
            reply = {"error": traceback.format_exc()[-3000:]}
        replies.write(json.dumps(reply) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    serve(sys.argv[1])
