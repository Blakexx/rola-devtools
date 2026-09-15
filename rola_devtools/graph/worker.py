"""THE UNIT WORKER: an instance's units in its own environment, answering the engine one line at a time.

    python -m rola_devtools.graph.worker describe module:function   ->  prints the graph's nodes, each with its location,
                                                                        identity, repeatable and timed
    python -m rola_devtools.graph.worker post module:Class PARAMS WS  ->  writes the unit's post to WS/post.json
    python -m rola_devtools.graph.worker serve                      ->  the persistent worker below

A served worker keeps each node's prepared state in memory between its phases (device state does not serialize). The
protocol is JSON lines on the process's original stdout; anything a unit or a native library prints goes to stderr.

    {"op": "setup", "node": NAME, "unit": U, "params": P, "ws": WS}   ->  {"ready": BUILT} | {"refused": WHY}
    {"op": "execute", "node": NAME, "ws": WS}                       ->  {"done": true}
    {"op": "call", "node": NAME}                                    ->  {"ms": MS}
    {"op": "exit"}                                                  ->  (the process exits)

A failure replies {"error": "<the traceback's tail>"} and leaves the worker serving.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import traceback
from pathlib import Path

from . import Node, Refusal, Timed, Unit


def _load(ref: str):
    module, _, name = ref.partition(":")
    if not module or not name:
        raise ValueError(f"expected module:name, got {ref!r}")
    return getattr(importlib.import_module(module), name)


def unit(ref: str, params: dict) -> Unit:
    return _load(ref)(**params)


def describe(graph: str) -> list[dict]:
    """The graph's nodes, each with what its unit says of itself."""
    out, names = [], set()
    for node in _load(graph)():
        if not isinstance(node, Node):
            raise TypeError(f"{graph} returned {type(node).__name__}, not a Node")
        if node.name in names:
            raise ValueError(f"{graph} declares {node.name} twice")
        names.add(node.name)
        u = unit(node.unit, node.params)
        if not u.location:
            raise ValueError(f"{node.unit} declares no location")
        out.append({"name": node.name, "unit": node.unit, "params": node.params, "deps": list(node.deps),
                    "location": u.location, "identity": u.identity(), "repeatable": u.repeatable, "timed": u.timed})
    missing = sorted({d for n in out for d in n["deps"]} - names)
    if missing:
        raise ValueError(f"{graph}: nodes depend on {missing}, which it does not declare")
    return out


def serve() -> None:
    replies = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    units: dict[str, Unit] = {}
    prepared: dict[str, object] = {}
    for line in sys.stdin:
        request = json.loads(line)
        if request["op"] == "exit":
            return
        try:
            name = request["node"]
            if request["op"] == "setup":
                units[name] = unit(request["unit"], request["params"])
                try:
                    prepared[name] = units[name].setup(Path(request["ws"]))
                except Refusal as refusal:
                    reply = {"refused": str(refusal)}
                else:
                    ready = prepared[name]
                    if units[name].timed and not isinstance(ready, Timed):
                        raise TypeError(f"{name}: a timed unit's setup returns a Timed, got {type(ready).__name__}")
                    reply = {"ready": {"built": ready.built, "instrument": ready.instrument}
                             if isinstance(ready, Timed) else {}}
            elif request["op"] == "execute":
                units[name].execute(prepared[name], Path(request["ws"]))
                reply = {"done": True}
            elif request["op"] == "call":
                reply = {"ms": float(prepared[name].call())}
            else:
                raise ValueError(f"unknown op {request['op']!r}")
        except Exception:  # noqa: BLE001 -- the engine names the node and fails it
            reply = {"error": traceback.format_exc()[-3000:]}
        replies.write(json.dumps(reply) + "\n")


def main(argv: list[str]) -> int:
    if argv[:1] == ["serve"]:
        serve()
        return 0
    if argv[:1] == ["describe"] and len(argv) == 2:
        reply = os.fdopen(os.dup(1), "w")
        os.dup2(2, 1)
        sys.stdout = sys.stderr
        reply.write(json.dumps(describe(argv[1])) + "\n")
        reply.close()
        return 0
    if argv[:1] == ["post"] and len(argv) == 4:
        ws = Path(argv[3])
        (ws / "post.json").write_text(json.dumps(unit(argv[1], json.loads(argv[2])).post(ws), sort_keys=True))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
