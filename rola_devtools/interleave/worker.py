"""THE WORKER: one runner's arms in one process, answering the driver one line at a time.

    python -m rola_devtools.interleave.worker module:function

The runner (`module:function`) is called with the data of a cell (`rola_devtools.cells.build`) and returns a builder per
arm name it accepts for that cell; raising refuses the cell. The protocol is JSON lines on the process's original stdout;
anything the runner or a native library prints goes to stderr, so a print can never corrupt a reply. Requests, replies:

    {"op": "list", "cells": [record, ...]}      ->  {"cells": {name: {"arms": [arm, ...]} | {"refused": "<why>"}}}
    {"op": "prepare", "cells": [record, ...], "arms": [[cell, arm], ...]}
                                                ->  {"arms": {"cell|arm": {"cell": {...}, "instrument": "..."}}}
    {"op": "call", "key": "cell|arm"}           ->  {"ms": <the arm's own elapsed milliseconds>}
    {"op": "exit"}                              ->  (the process exits)

Any failure replies {"error": "<the traceback's tail>"} and leaves the worker serving.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import traceback
from collections.abc import Callable

from ..cells import build
from .arm import Arm


def load(runner: str) -> Callable[[object], dict[str, Callable[[], Arm]]]:
    module, _, function = runner.partition(":")
    if not module or not function:
        raise ValueError(f"a runner is module:function, got {runner!r}")
    return getattr(importlib.import_module(module), function)


def serve(runner: str) -> None:
    replies = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    arms: dict[str, Arm] = {}
    data: dict[str, object] = {}
    make = None

    def data_of(record: dict):
        if record["name"] not in data:
            data[record["name"]] = build(record)
        return data[record["name"]]

    for line in sys.stdin:
        request = json.loads(line)
        if request["op"] == "exit":
            return
        try:
            make = make or load(runner)
            if request["op"] == "list":
                listed = {}
                for record in request["cells"]:
                    try:
                        listed[record["name"]] = {"arms": sorted(make(data_of(record)))}
                    except Exception as refusal:  # noqa: BLE001 -- a refusal is the answer for that cell
                        listed[record["name"]] = {"refused": f"{type(refusal).__name__}: {refusal}"}
                reply = {"cells": listed}
            elif request["op"] == "prepare":
                for record in request["cells"]:
                    builders = make(data_of(record))
                    wanted = [arm for cell, arm in request["arms"] if cell == record["name"]]
                    missing = sorted(set(wanted) - set(builders))
                    if missing:
                        raise KeyError(f"no arm {missing} for cell {record['name']}; the runner has {sorted(builders)}")
                    for arm in wanted:
                        arms[f"{record['name']}|{arm}"] = builders[arm]()
                reply = {"arms": {key: {"cell": arm.cell, "instrument": arm.instrument} for key, arm in arms.items()}}
            elif request["op"] == "call":
                reply = {"ms": float(arms[request["key"]].call())}
            else:
                raise ValueError(f"unknown op {request['op']!r}")
        except Exception:  # noqa: BLE001 -- the driver names the arm and raises
            reply = {"error": traceback.format_exc()[-2000:]}
        replies.write(json.dumps(reply) + "\n")


if __name__ == "__main__":
    serve(sys.argv[1])
