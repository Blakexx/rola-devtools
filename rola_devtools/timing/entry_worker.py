"""THE ENTRY WORKER: a checkout's timed callables, set up, called, reset and dropped for the timing server.

    python -m rola_devtools.timing.entry_worker

The protocol is JSON lines on the process's original stdout; what an executor prints goes to stderr.

    {"op": "setup", "id", "executor", "cell", "params"}   ->  {"ready": {"built", "instrument", "draw"}} | {"failed": WHY}
    {"op": "call", "id"}                                  ->  {"ms": MS}
    {"op": "drop", "id"}                                  ->  {"dropped": true}
    {"op": "memory", "executor", "cell", "params", "calls"} ->  {"memory": {...}} | {"failed": WHY}
    {"op": "clock", "executor"}                           ->  {"ghz": GHZ | null}
    {"op": "exit"}

A request's `env` holds variables set for that request alone (the lock-held markers). `call` runs the entry's reset
before timing, untimed. A setup that raises is the entry's failure; any other exception replies {"error": ...}.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import traceback

from ..build import identity
from ..build.worker import load


def _draw() -> str:
    from .. import cells

    root = cells.FILES[0].parent
    return identity.files(root, sorted(p.name for p in root.iterdir() if p.suffix in (".py", ".json")))


def _cell(record):
    from ..cells import build

    return build(record)


def _release() -> None:
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _memory(timed, calls: int, before: int) -> dict:
    import torch

    if timed.reset:
        timed.reset()
    timed.call()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(calls):
        if timed.reset:
            timed.reset()
        timed.call()
    torch.cuda.synchronize()
    return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "allocated_after_bytes": torch.cuda.memory_allocated(), "allocated_before_build_bytes": before,
            "outside_allocator_bytes": int(timed.outside_allocator()) if timed.outside_allocator else 0,
            "calls": calls, "built": timed.built}


def serve() -> None:
    replies = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    live: dict[str, object] = {}
    for line in sys.stdin:
        request = json.loads(line)
        op = request["op"]
        if op == "exit":
            return
        saved = {name: os.environ.get(name) for name in request.get("env", {})}
        os.environ.update(request.get("env", {}))
        try:
            if op == "setup":
                try:
                    timed = load(request["executor"])(_cell(request["cell"]), request["params"])
                except Exception:  # noqa: BLE001 -- the entry's domain failure
                    reply = {"failed": traceback.format_exc()[-3000:]}
                else:
                    live[request["id"]] = timed
                    reply = {"ready": {"built": timed.built, "instrument": timed.instrument, "draw": _draw()}}
            elif op == "call":
                timed = live[request["id"]]
                if timed.reset:
                    timed.reset()
                reply = {"ms": float(timed.call())}
            elif op == "drop":
                live.pop(request["id"], None)
                _release()
                reply = {"dropped": True}
            elif op == "memory":
                import torch

                _release()
                before = torch.cuda.memory_allocated()
                try:
                    timed = load(request["executor"])(_cell(request["cell"]), request["params"])
                    reply = {"memory": _memory(timed, request["calls"], before)}
                except Exception:  # noqa: BLE001 -- the entry's domain failure
                    reply = {"failed": traceback.format_exc()[-3000:]}
                finally:
                    timed = None
                    _release()
            elif op == "clock":
                reply = {"ghz": load(request["executor"])()}
            else:
                raise ValueError(f"unknown op {op!r}")
        except Exception:  # noqa: BLE001 -- the timing system names the entry and fails the session
            reply = {"error": traceback.format_exc()[-3000:]}
        finally:
            for name, value in saved.items():
                os.environ.pop(name, None) if value is None else os.environ.__setitem__(name, value)
        replies.write(json.dumps(reply) + "\n")


if __name__ == "__main__":
    serve()
