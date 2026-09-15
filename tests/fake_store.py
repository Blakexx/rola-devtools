"""The store's interface over a directory -- `get`, `put`, `output` -- with the build system's key, for the tests."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from rola_devtools.build import key


class FakeStore:
    def __init__(self, root: Path, location: str) -> None:
        self.dir = Path(root) / location

    def get(self, k):
        path = self.dir / f"{k}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def put(self, semantics, *, output=None, output_file=None, error=None, wall_s=None, provenance=None):
        if (error is None) == (output is None and output_file is None):
            raise ValueError("a sample is exactly one of output, output_file or error")
        k = key(semantics)
        self.dir.mkdir(parents=True, exist_ok=True)
        record = self.get(k) or {"key": k, "semantics": semantics, "samples": []}
        n = len(record["samples"])
        sample = {"n": n, "ok": error is None, "provenance": provenance, "wall_s": wall_s}
        if error is not None:
            sample["error"] = error
        elif output_file is not None:
            sample["output"] = f"{k}.{n}.out{Path(output_file).suffix}"
            shutil.move(str(output_file), self.dir / sample["output"])
        else:
            sample["output"] = f"{k}.{n}.out.json"
            (self.dir / sample["output"]).write_text(json.dumps(output))
        record["samples"].append(sample)
        (self.dir / f"{k}.json").write_text(json.dumps(record))
        return sample

    def output(self, record):
        return self.dir / [s for s in record["samples"] if s["ok"]][-1]["output"]
