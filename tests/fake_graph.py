"""A graph for the engine's tests: exclusive units that count, refuse or fail, and timed units with fixed elapsed times,
so what runs, what is stored and what is skipped are checked exactly."""
from __future__ import annotations

import json
import os
from pathlib import Path

from rola_devtools.graph import Node, Refusal, Timed, Unit

#: a file every execute appends its node to, so a test sees which nodes executed (set by the test's environment)
LEDGER = "FAKE_GRAPH_LEDGER"


class Count(Unit):
    location = "fake/count"
    repeatable = True

    def __init__(self, n: int, refuse: bool = False, fail: bool = False) -> None:
        self.n, self.refuse, self.fail = n, refuse, fail

    def identity(self) -> dict:
        return {"n": self.n, "version": os.environ.get("FAKE_GRAPH_VERSION", "1")}

    def setup(self, ws: Path):
        if self.refuse:
            raise Refusal(f"{self.n} is not a number this unit takes")
        return self.n * 10

    def execute(self, prepared, ws: Path) -> None:
        if self.fail:
            raise RuntimeError(f"unit {self.n} failed on purpose")
        with open(os.environ[LEDGER], "a") as ledger:
            ledger.write(f"{self.n}\n")
        (ws / "raw.json").write_text(json.dumps({"prepared": prepared}))

    def post(self, ws: Path) -> dict:
        deps = {p.stem: json.loads(p.read_text()) for p in sorted((ws / "deps").glob("*.json"))}
        return {"value": json.loads((ws / "raw.json").read_text())["prepared"], "deps": deps}


class Fixed(Unit):
    location = "fake/timed"
    timed = True

    def __init__(self, ms: float, refuse: bool = False) -> None:
        self.ms, self.refuse = ms, refuse

    def identity(self) -> dict:
        return {"ms": self.ms}

    def setup(self, ws: Path):
        if self.refuse:
            raise Refusal("this arm is not built")
        return Timed(call=lambda: self.ms, built={"ms": self.ms}, instrument="fixed")

    def post(self, ws: Path) -> dict:
        return {"samples": len(json.loads((ws / "samples.json").read_text())["ms"])}


def graph() -> list[Node]:
    count = "fake_graph:Count"
    return [Node("a", count, {"n": 1}), Node("b", count, {"n": 2}, deps=("a",)),
            Node("no", count, {"n": 3, "refuse": True}), Node("after-no", count, {"n": 4}, deps=("no",)),
            Node("boom", count, {"n": 5, "fail": True}), Node("after-boom", count, {"n": 6}, deps=("boom",)),
            Node("fast", "fake_graph:Fixed", {"ms": 1.0}), Node("slow", "fake_graph:Fixed", {"ms": 3.0}),
            Node("unbuilt", "fake_graph:Fixed", {"ms": 2.0, "refuse": True})]
