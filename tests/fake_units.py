"""An owner's registry for the service's tests: instruments that count, refuse or fail, a build of a file, arms with
fixed elapsed times, and a clock reader -- so what runs, what is stored and what is skipped are checked exactly."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from rola_devtools.measure import Arm, Build, ClockReader, Instrument, Refusal, Registration, Timed

#: a file every execute appends a line to, so a test sees what executed (set by the test's environment)
LEDGER = "FAKE_UNITS_LEDGER"
#: the file the fake build writes
BINARY = "FAKE_UNITS_BINARY"


def _ledger(line: str) -> None:
    with open(os.environ[LEDGER], "a") as ledger:
        ledger.write(line + "\n")


class Count(Instrument):
    location = "fake/count"
    repeatable = True

    def __init__(self, n: int, refuse: bool = False, fail: bool = False, max_tokens: int = 256) -> None:
        self.n, self.refuse, self.fail, self.max_tokens = n, refuse, fail, max_tokens

    def accepts(self, cell) -> str | None:
        return None if cell.tokens <= self.max_tokens else f"{cell.name}: more than {self.max_tokens} tokens"

    def identity(self, cell) -> dict:
        return {"n": self.n, "version": os.environ.get("FAKE_UNITS_VERSION", "1")}

    def setup(self, cell, ws: Path):
        if self.refuse:
            raise Refusal(f"{self.n} does not fit on this device")
        return {"value": self.n * cell.tokens}

    def execute(self, prepared, ws: Path) -> None:
        if self.fail:
            raise RuntimeError(f"unit {self.n} failed on purpose")
        held = subprocess.run([sys.executable, "-c", "import os; print(os.environ.get('ROLA_GPU_LOCK_HELD', 'none'))"],
                              capture_output=True, text=True, check=True).stdout.strip()
        _ledger(f"count {self.n} {prepared['value']} held={held}")
        (ws / "raw.json").write_text(json.dumps(prepared))

    def post(self, ws: Path, handle) -> None:
        deps = {p.stem: json.loads(p.read_text()) for p in sorted((ws / "deps").glob("*.json"))}
        handle.output({"value": json.loads((ws / "raw.json").read_text())["value"], "deps": deps})


class Binary(Build):
    location = "fake/build"

    def identity(self, cell) -> dict:
        return {"source": os.environ.get("FAKE_UNITS_SOURCE", "1")}

    def execute(self, ws: Path) -> None:
        _ledger("build")
        Path(os.environ[BINARY]).write_text(os.environ.get("FAKE_UNITS_SOURCE", "1"))

    def post(self, ws: Path, handle) -> None:
        handle.output({"sha256": hashlib.sha256(Path(os.environ[BINARY]).read_bytes()).hexdigest()})

    def present(self, output: dict) -> bool:
        path = Path(os.environ[BINARY])
        return path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == output["sha256"]


class Fixed(Arm):
    location = "fake/memory"

    def __init__(self, ms: float, refuse: bool = False) -> None:
        self.ms, self.refuse = ms, refuse

    def identity(self, cell) -> dict:
        return {"ms": self.ms}

    def setup(self, cell, ws: Path) -> Timed:
        if self.refuse:
            raise Refusal("this arm does not fit on this device")
        return Timed(call=lambda: self.ms, built={"ms": self.ms, "tokens": cell.tokens}, instrument="fixed")


class Holding(Arm):
    """An arm that holds a device tensor of its cell's token count in bytes, for the memory node."""

    location = "fake/memory"

    def setup(self, cell, ws: Path) -> Timed:
        import torch

        held = torch.empty(cell.tokens * 1024, dtype=torch.uint8, device="cuda")

        def call() -> float:
            torch.empty(cell.tokens * 4096, dtype=torch.uint8, device="cuda")
            return float(held.numel())

        return Timed(call=call, built={"held": held.numel()}, instrument="fixed", outside_allocator=lambda: 7)


class Clock(ClockReader):
    def read_ghz(self) -> float:
        return 1.665


def registry() -> list[Registration]:
    here = "fake_units:"
    return [Registration("build", here + "Binary"),
            Registration("a", here + "Count", {"n": 1}, deps=("build",)),
            Registration("b", here + "Count", {"n": 2}, deps=("a", "build")),
            Registration("no", here + "Count", {"n": 3, "refuse": True}),
            Registration("after-no", here + "Count", {"n": 4}, deps=("no",)),
            Registration("boom", here + "Count", {"n": 5, "fail": True}),
            Registration("after-boom", here + "Count", {"n": 6}, deps=("boom",)),
            Registration("fast", here + "Fixed", {"ms": 1.0}),
            Registration("slow", here + "Fixed", {"ms": 3.0}),
            Registration("unfit", here + "Fixed", {"ms": 2.0, "refuse": True}),
            Registration("holding", here + "Holding"),
            Registration("clock", here + "Clock")]
