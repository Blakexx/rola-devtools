"""`python -m rola_devtools.measure plan|run REGISTRY`: one owner's registry run from its own checkout, stored through
rola-results.

    python -m rola_devtools.measure plan benchmarks.registry:registry --cells flagship-dense --units carry.phases
    python -m rola_devtools.measure run  benchmarks.registry:registry --cells flagship-dense,flagship-alt-k4 \\
        --session carry_forward@flagship-dense,flagship-alt-k4 [--no-memory] [--repeat] [--force] [--store-root DIR]

The registry runs in this process's environment and directory as the one instance `local` (a label no key reads), so a
node run here and the same node run by a composer that includes this owner share one record. `--cells` names central
cells (`all` for every one); `--units` names the registrations whose nodes run (`all`, or none with `--units ""` for
sessions alone); each `--session ARMS@CELLS` interleaves those arms on those cells. The device and the host's clock are
held by the service's locks.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .service import Instance, Service, Session, outcomes_line

#: where a session's record lives
SESSION_LOCATION = "sessions"


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m rola_devtools.measure", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("plan", "run"))
    ap.add_argument("registry", help="the owner's registry function, module:function")
    ap.add_argument("--cells", default="", help="all, or a comma list of central cells")
    ap.add_argument("--units", default="all", help="all, or a comma list of registrations whose nodes run")
    ap.add_argument("--session", action="append", default=[], help="ARM[+ARM...]@CELL[,CELL...] (repeatable)")
    ap.add_argument("--no-memory", action="store_true", help="leave the arms' memory nodes out")
    ap.add_argument("--repeat", action="store_true", help="add a sample to complete repeatable nodes and sessions")
    ap.add_argument("--force", action="store_true", help="run every selected node again")
    ap.add_argument("--store-root", type=Path, help="the rola-results backend directory (default: its own records/)")
    a = ap.parse_args()
    try:
        from rola_results import ROOT, Store, checkout
    except ImportError as ex:
        raise SystemExit(f"records are stored through rola-results, which is not importable here: {ex}") from ex
    from ..cells import central

    cells = list(central().cells) if a.cells == "all" else [c for c in a.cells.split(",") if c]
    units = None if a.units == "all" else {u for u in a.units.split(",") if u}
    sessions = []
    for spec in a.session:
        arms, _, names = spec.partition("@")
        if not arms or not names:
            raise SystemExit(f"--session takes ARM[+ARM...]@CELL[,CELL...], got {spec!r}")
        sessions.append(Session(spec, SESSION_LOCATION, tuple(arms.split("+")), tuple(names.split(","))))
    instance = Instance("local", sys.executable, os.getcwd(), a.registry)
    root = a.store_root or ROOT
    with Service([instance], provenance=lambda inst: checkout(inst.cwd), log=lambda line: print(line, flush=True)) as service:
        nodes = service.nodes(cells, sessions=sessions, memory=not a.no_memory,
                              select=lambda inst, unit: units is None or unit["name"] in units)
        outcomes = service.run(nodes, lambda location: Store(location, root), repeat=a.repeat, force=a.force,
                               dry=a.cmd == "plan")
    print(f"measure: {outcomes_line(outcomes)}")
    return 1 if any(o.status in ("failed", "blocked") for o in outcomes) else 0


if __name__ == "__main__":
    sys.exit(main())
