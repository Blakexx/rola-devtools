"""`python -m rola_devtools.build plan|run FILE:TARGET`: a declared build, run from its root declaration file.

    python -m rola_devtools.build plan declare.py:all
    python -m rola_devtools.build run  ../rola-bench/declare.py:suite --arg target=/path/to/rola --arg reference=...

FILE defines `root(g, **args)`, which declares on the graph `g` (loading and calling other declaration files as it
likes) and returns its public targets by name; TARGET is one of those names. `--arg name=value` is handed to `root` as a
string. The build system runs in this interpreter, in FILE's directory; targets run in the environments they declare.
`--only GLOB` and `--skip GLOB` prune the graph by LABEL -- the one selection mechanism, which knows nothing of what a
label means: `--only 'rola/side/*'` runs those targets and what they need, `--skip '*/timeline'` drops that instrument
and everything that reads it.
The cache is the dev config's `host.build_cache` unless `--cache` names another, and a `run` sweeps it first:
the oldest runs' workspaces, then the least recently read keyed entries over the budget.
"""
from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

from ..config import machine
from .cache import Cache
from .declare import Graph, load
from .runtime import Runtime
from .scheduler import build, select, summary


def main() -> int:
    #: a SIGTERM unwinds like an interrupt, so the workers are stopped with their trees
    signal.signal(signal.SIGTERM, lambda signum, _frame: sys.exit(128 + signum))
    ap = argparse.ArgumentParser(prog="python -m rola_devtools.build", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("plan", "run"))
    ap.add_argument("target", help="FILE:TARGET -- a root declaration file and one of the targets its root() returns")
    ap.add_argument("--arg", action="append", default=[], help="name=value handed to root() (repeatable)")
    ap.add_argument("--only", action="append", default=[], metavar="GLOB",
                    help="build only the reachable targets whose label matches, and what they need (repeatable)")
    ap.add_argument("--skip", action="append", default=[], metavar="GLOB",
                    help="drop the targets whose label matches, and everything that needs them (repeatable)")
    ap.add_argument("--force", action="store_true", help="run cached targets again")
    ap.add_argument("--cache", type=Path, help="the build cache directory (default: host.build_cache)")
    a = ap.parse_args()
    file, _, name = a.target.rpartition(":")
    if not file or not name:
        raise SystemExit(f"expected FILE:TARGET, got {a.target!r}")
    args = dict(pair.split("=", 1) for pair in a.arg)
    declared = load(file)
    if "root" not in declared:
        raise SystemExit(f"{file} defines no root(g, **args)")
    public = declared["root"](Graph(), **args)
    if name not in public:
        raise SystemExit(f"{file}: root() declares no target {name!r}; it declares {sorted(public)}")
    cache = Cache(a.cache or machine("host.build_cache"))
    if a.cmd == "run":
        #: BEFORE this run's own directory exists, so a sweep never touches the run it is sweeping for
        swept = cache.sweep()
        if swept["runs_dropped"] or swept["keys_evicted"]:
            print(f"cache: swept {swept['runs_dropped']} run(s) and {swept['keys_evicted']} key(s), "
                  f"{swept['freed_bytes'] / 1024**2:.0f} MiB", flush=True)
    with Runtime(Path(file).resolve().parent) as runtime:
        results = build(select([public[name]], only=a.only, skip=a.skip), cache=cache, runtime=runtime,
                        dry=a.cmd == "plan", force=a.force, log=lambda line: print(line, flush=True))
    print(f"build: {summary(results)}")
    return 1 if any(r.status == "failed" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
