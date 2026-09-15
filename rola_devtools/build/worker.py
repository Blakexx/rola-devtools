"""THE ENVIRONMENT WORKER: runs targets' executors inside one environment, for a whole build.

    python -m rola_devtools.build.worker

The build system starts one worker per environment a build names (a checkout's venv and directory, or its own) and
keeps it for the run, so an executor may leave live state in its module (a timing server's worker pool) for a later
target in the same environment. The protocol is JSON lines on the process's original stdout; while an executor runs,
its stdout and stderr go to `log.txt` in its workspace.

    {"op": "hello"}                                          ->  {"protocol": PROTOCOL}
    {"op": "run", "executor", "context", "env"}              ->  {"output": JSON}
    {"op": "verify", "verify", "output"}                     ->  {"ok": BOOL}
    {"op": "exit"}                                           ->  (the process exits)

`env` holds variables set for that request alone (the lock-held markers of what the target holds). A failure replies
{"error": "<the traceback's tail>"} and leaves the worker serving.
"""
from __future__ import annotations

import contextlib
import importlib
import json
import os
import sys
import traceback
from pathlib import Path

#: the protocol between the build system and its workers; a worker of another version is refused at hello
PROTOCOL = 1


def load(ref: str):
    module, _, name = ref.partition(":")
    if not module or not name:
        raise ValueError(f"expected module:function, got {ref!r}")
    return getattr(importlib.import_module(module), name)


@contextlib.contextmanager
def _logged(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    with open(path, "a") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            for fd in saved:
                os.close(fd)


def serve() -> None:
    from .context import Context

    replies = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    for line in sys.stdin:
        request = json.loads(line)
        op = request["op"]
        if op == "exit":
            return
        saved = {name: os.environ.get(name) for name in request.get("env", {})}
        os.environ.update(request.get("env", {}))
        try:
            if op == "hello":
                reply = {"protocol": PROTOCOL}
            elif op == "run":
                ctx = Context.from_json(request["context"])
                with _logged(ctx.workspace / "log.txt"):
                    output = load(request["executor"])(ctx)
                reply = {"output": json.loads(json.dumps(output))}
            elif op == "verify":
                reply = {"ok": bool(load(request["verify"])(request["output"]))}
            else:
                raise ValueError(f"unknown op {op!r}")
        except Exception:  # noqa: BLE001 -- the build system names the target and fails it
            reply = {"error": traceback.format_exc()[-4000:]}
        finally:
            for name, value in saved.items():
                os.environ.pop(name, None) if value is None else os.environ.__setitem__(name, value)
        replies.write(json.dumps(reply) + "\n")


if __name__ == "__main__":
    serve()
