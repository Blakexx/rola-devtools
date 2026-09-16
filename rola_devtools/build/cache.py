"""THE BUILD CACHE: which target keys have completed on this machine, with their outputs and kept files.

One directory an entry, `<root>/keys/<key>/`: the semantics its key hashes, its JSON output, and the files its workspace
held. An entry is written whole or not at all (built beside and renamed into place). The cache is the build system's
own and wipeable -- deleting it only means work runs again; records worth keeping are a result store's (rola-results),
written by targets that choose to. A run's workspaces live under `<root>/runs/<run id>/`.

IT EVICTS, because a cache that only grows is a disk that only shrinks, and this one lives on the same volume the
toolchains and the vhdx do. `sweep()` runs at the start of a build and applies two policies to the two halves, which
grow for different reasons:

* `runs/` is SCRATCH, and it is where the bytes actually are (measured 2026-09-15: 35 MB of runs against 244 KB of
  keys, six runs). A run's workspaces are what its targets wrote -- kept after the build for the log, the error text
  and the raw files of a target the build did not cache. The newest `KEEP_RUNS` are kept and older ones are deleted
  whole; a workspace worth more than that is a store target's job, not a cache's.
* `keys/` is the CACHE PROPER, and an entry is worth keeping as long as builds still ask for it. `get` touches the
  entry it hands back, so the policy is a true LRU: entries are dropped oldest-read first, until what is left fits
  `MAX_KEY_BYTES`.

Neither policy can lose work that cannot be redone: an evicted key is a target that runs again, and a deleted run is a
build that already reported.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

#: how many runs' workspaces survive a sweep, and the byte budget the keyed entries fit into
KEEP_RUNS = 10
MAX_KEY_BYTES = 4 * 1024**3


def _bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


class Cache:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def entry(self, key: str) -> Path:
        return self.root / "keys" / key

    def get(self, key: str) -> dict | None:
        entry = self.entry(key)
        if not (entry / "output.json").is_file():
            return None
        #: THE LRU'S CLOCK: a hit is a use, and the sweep evicts by it
        os.utime(entry, None)
        return {"output": json.loads((entry / "output.json").read_text()), "dir": str(entry / "files")}

    def put(self, key: str, semantics: dict, output, workspace: Path) -> str:
        entry = self.entry(key)
        tmp = entry.with_name(f".{key}.{os.getpid()}")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        shutil.copytree(workspace, tmp / "files", dirs_exist_ok=True)
        (tmp / "semantics.json").write_text(json.dumps(semantics, indent=1, sort_keys=True))
        (tmp / "output.json").write_text(json.dumps(output, indent=1, sort_keys=True))
        shutil.rmtree(entry, ignore_errors=True)
        tmp.rename(entry)
        return str(entry / "files")

    def workspace(self, run: str, label: str) -> Path:
        path = self.root / "runs" / run / label
        path.mkdir(parents=True, exist_ok=True)
        return path

    def sweep(self, *, keep_runs: int = KEEP_RUNS, max_key_bytes: int = MAX_KEY_BYTES) -> dict:
        """Apply both policies and report what went. Called at the start of a build, BEFORE its own run directory
        exists, so a sweep never touches the run it is sweeping for."""
        runs = sorted((d for d in (self.root / "runs").glob("*") if d.is_dir()), key=lambda d: d.name)
        dropped_runs = runs[:-keep_runs] if keep_runs >= 0 and len(runs) > keep_runs else []
        freed = 0
        for run in dropped_runs:
            freed += _bytes(run)
            shutil.rmtree(run, ignore_errors=True)

        entries = [(d.stat().st_mtime, _bytes(d), d) for d in (self.root / "keys").glob("*") if d.is_dir()]
        total = sum(size for _, size, _ in entries)
        dropped_keys = []
        for _mtime, size, entry in sorted(entries):  # oldest read first
            if total <= max_key_bytes:
                break
            shutil.rmtree(entry, ignore_errors=True)
            dropped_keys.append(entry.name)
            total -= size
            freed += size
        return {"runs_dropped": len(dropped_runs), "keys_evicted": len(dropped_keys), "freed_bytes": freed,
                "keys_bytes": total, "swept": time.time()}


__all__ = ["Cache"]
