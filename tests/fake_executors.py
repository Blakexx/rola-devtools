"""Executors for the declared build system's tests: each appends to a ledger so a test sees what ran, and returns what
it was given so a test sees what reached it."""
from __future__ import annotations

import os
from pathlib import Path

#: a file every executor appends a line to (set by the test's environment)
LEDGER = "FAKE_EXECUTORS_LEDGER"


def _ledger(line: str) -> None:
    with open(os.environ[LEDGER], "a") as ledger:
        ledger.write(line + "\n")


def echo(ctx) -> dict:
    _ledger(f"echo {ctx.label}")
    print(f"a log line from {ctx.label}")
    (ctx.workspace / "raw.txt").write_text(ctx.label)
    return {"params": ctx.params, "inputs": [c["name"] for c in ctx.inputs],
            "deps": {role: dep.output for role, dep in ctx.deps.items()},
            "held": os.environ.get("ROLA_GPU_LOCK_HELD")}


def boom(ctx) -> dict:
    _ledger(f"boom {ctx.label}")
    raise RuntimeError(f"{ctx.label} failed on purpose")


def domain(ctx) -> dict:
    _ledger(f"domain {ctx.label}")
    return {"entries": [{"cell": "t8", "ok": True}, {"cell": "t16", "ok": False, "error": "no kernel for 16 tokens"}]}


def cleanup(ctx) -> dict:
    _ledger(f"cleanup {ctx.label}")
    return {"saw": {role: dep.status for role, dep in ctx.deps.items()}}


def present(output) -> bool:
    return Path(os.environ[LEDGER]).with_name("present").exists()
