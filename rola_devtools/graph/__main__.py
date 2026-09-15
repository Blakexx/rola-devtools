"""`python -m rola_devtools.graph plan|run GRAPH`: an owner's graph run from its own checkout, stored through rola-results.

    python -m rola_devtools.graph plan benchmarks.graph:graph [--select NAME ...]
    python -m rola_devtools.graph run  benchmarks.graph:graph [--select NAME ...] [--hold module:function] [--repeat]
                                       [--force] [--store-root DIR]

The graph is described and run in this process's environment and directory (its instance label is `local`, which no key
reads), so a node run here and the same node run by a composer that includes this graph share one record. `--hold` names a
context-manager factory the setups and executes run under (the device and clock locks).
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path

from . import Env
from .engine import load, run
from .worker import _load


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m rola_devtools.graph", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("plan", "run"))
    ap.add_argument("graph", help="the graph function, module:function")
    ap.add_argument("--select", action="append", default=[], help="a node name (repeatable; its dependencies come along)")
    ap.add_argument("--hold", help="module:function returning the context setups and executes run under")
    ap.add_argument("--repeat", action="store_true", help="add a sample to complete repeatable nodes")
    ap.add_argument("--force", action="store_true", help="run every selected node again")
    ap.add_argument("--store-root", type=Path, help="the rola-results backend directory (default: its own records/)")
    a = ap.parse_args()
    try:
        from rola_results import ROOT, Store, checkout
    except ImportError as ex:
        raise SystemExit(f"the graph's records are stored through rola-results, which is not importable here: {ex}") from ex
    root = a.store_root or ROOT
    env = Env("local", sys.executable, os.getcwd(), a.graph)
    instance = load(env)
    hold = _load(a.hold) if a.hold else contextlib.nullcontext
    outcomes = run([instance], lambda location: Store(location, root), hold=hold,
                   select={instance.qualified(name) for name in a.select} or None, repeat=a.repeat, force=a.force,
                   dry=a.cmd == "plan", provenance=lambda e: checkout(e.cwd), log=lambda line: print(line, flush=True))
    counts: dict[str, int] = {}
    for outcome in outcomes:
        counts[outcome.status] = counts.get(outcome.status, 0) + 1
    print("graph: " + ", ".join(f"{n} {status}" for status, n in sorted(counts.items())))
    return 1 if counts.get("failed") or counts.get("blocked") else 0


if __name__ == "__main__":
    sys.exit(main())
