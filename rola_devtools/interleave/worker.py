"""THE WORKER: one provider's arms in one process, answering the driver one line at a time.

    python -m rola_devtools.interleave.worker module:function

The protocol is JSON lines on the process's original stdout; anything the provider or a native library prints goes to
stderr, so a print can never corrupt a reply. Requests and replies:

    {"op": "prepare", "point": {...}}  ->  {"arms": {name: {"cell": {...}, "instrument": "..."}}}
    {"op": "call", "arm": name}        ->  {"ms": <the arm's own elapsed milliseconds>}
    {"op": "exit"}                     ->  (the process exits)

Any failure replies {"error": "<the traceback's tail>"} and leaves the worker serving.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import traceback
from collections.abc import Callable

from .arm import Arm


def load(provider: str) -> Callable[[dict], dict[str, Arm]]:
    module, _, function = provider.partition(":")
    if not module or not function:
        raise ValueError(f"a provider is module:function, got {provider!r}")
    return getattr(importlib.import_module(module), function)


def serve(provider: str) -> None:
    replies = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    arms: dict[str, Arm] = {}
    make = None
    for line in sys.stdin:
        request = json.loads(line)
        if request["op"] == "exit":
            return
        try:
            if request["op"] == "prepare":
                make = make or load(provider)
                arms = make(request["point"])
                reply = {"arms": {name: {"cell": arm.cell, "instrument": arm.instrument} for name, arm in arms.items()}}
            elif request["op"] == "call":
                reply = {"ms": float(arms[request["arm"]].call())}
            else:
                raise ValueError(f"unknown op {request['op']!r}")
        except Exception:  # noqa: BLE001 -- the driver names the arm and raises
            reply = {"error": traceback.format_exc()[-2000:]}
        replies.write(json.dumps(reply) + "\n")


if __name__ == "__main__":
    serve(sys.argv[1])
