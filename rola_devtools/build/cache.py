"""THE BUILD CACHE: which target keys have completed on this machine, with their outputs and kept files.

One directory an entry, `<root>/keys/<key>/`: the semantics its key hashes, its JSON output, and the files its workspace
held. An entry is written whole or not at all (built beside and renamed into place). The cache is the build system's
own and wipeable -- deleting it only means work runs again; records worth keeping are a result store's (rola-results),
written by targets that choose to. A run's workspaces live under `<root>/runs/<run id>/`.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


class Cache:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def entry(self, key: str) -> Path:
        return self.root / "keys" / key

    def get(self, key: str) -> dict | None:
        entry = self.entry(key)
        if not (entry / "output.json").is_file():
            return None
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


__all__ = ["Cache"]
