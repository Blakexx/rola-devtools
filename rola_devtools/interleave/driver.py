"""The driver's side: workers started per runner environment, each sent its cells, arms interleaved per call."""
from __future__ import annotations

import contextlib
import json
import os
import random
import statistics
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

#: Launches of each arm discarded before any is recorded. A floor, not a default: the first launches build a kernel's
#: schedule and warm the caches, and a median containing them is a median about module load.
WARMUP_FLOOR = 10


@dataclass(frozen=True)
class ArmSpec:
    """One row of a comparison, run on every cell its point sends its runner. `label` names it; `provider`
    (module:function) is the runner code, run by `python` in `cwd` with `env` added; `arm` is the runner's name for what
    it times; `runner` is the runner the point addresses (default: the label). Specs agreeing on the environment share a
    worker; `worker` names a worker explicitly, so one environment can be two processes."""

    label: str
    provider: str
    arm: str
    python: str = sys.executable
    cwd: str | None = None
    env: dict = field(default_factory=dict)
    worker: str = ""
    runner: str = ""

    @property
    def addressed(self) -> str:
        return self.runner or self.label

    def key(self) -> tuple:
        return self.python, self.cwd, self.provider, tuple(sorted(self.env.items())), self.worker


class _Worker:
    def __init__(self, spec: ArmSpec) -> None:
        self.spec = spec
        self.process = subprocess.Popen([spec.python, "-m", "rola_devtools.interleave.worker", spec.provider],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=spec.cwd, text=True,
                                        env={**os.environ, **spec.env})

    def ask(self, request: dict, label: str) -> dict:
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"{label}: the worker for {self.spec.provider} exited (exit {self.process.wait()})")
        reply = json.loads(line)
        if "error" in reply:
            raise RuntimeError(f"{label}: {self.spec.provider} refused {request['op']}:\n{reply['error']}")
        return reply

    def close(self) -> None:
        with contextlib.suppress(BrokenPipeError, ValueError):
            self.process.stdin.write(json.dumps({"op": "exit"}) + "\n")
            self.process.stdin.flush()
        self.process.wait(timeout=60)
        self.process.stdin.close()
        self.process.stdout.close()


def iqr(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[int(0.75 * len(ordered))] - ordered[int(0.25 * len(ordered))]


def accepts(spec: ArmSpec, cells: list[dict]) -> dict[str, dict]:
    """What `spec`'s runner makes of each cell record, without building an arm: `{name: {"arms": [...]}}` for a cell it
    takes, `{name: {"refused": why}}` for one it refuses."""
    worker = _Worker(spec)
    try:
        return worker.ask({"op": "list", "cells": cells}, spec.label)["cells"]
    finally:
        worker.close()


def interleave(point: dict, arms: list[ArmSpec], *, rounds: int = 8, reps: int = 11, warmup: int = WARMUP_FLOOR,
               reference: str | None = None, seed: int = 0,
               hold: Callable[[], contextlib.AbstractContextManager] = contextlib.nullcontext) -> dict:
    """Every arm on every cell `point` sends its runner, each rep calling each (arm, cell) once in a fresh random order.

    `point` is resolved (`rola_devtools.cells.Registry.point`): its cells are records. A row is `label|cell`. `reference`
    is what the paired ratios divide by: a LABEL (the first arm's by default), whose row on the same cell pairs with each
    row on that cell and whose every row pairs with a row on a cell it does not run; or one ROW, which pairs with every
    other. `hold` is the context the timed calls run under (the caller's GPU lock and clock lock). Returns the point, the
    stopwatch, and per row its cell, what the runner built from it, the raw samples in the order taken, per-round block
    medians, and its pairings: the ratios and per-round differences against each reference row it pairs with.
    """
    labels = [spec.label for spec in arms]
    if len(set(labels)) != len(labels):
        raise ValueError(f"arm labels must be unique, got {labels}")
    if warmup < WARMUP_FLOOR:
        raise ValueError(f"REFUSING to measure with {warmup} warmup launches; the floor is {WARMUP_FLOOR}")
    if reps % 2 == 0:
        raise ValueError(f"reps per round must be odd so a block median is a sample, got {reps}")
    rows: list[tuple[str, ArmSpec, dict]] = []
    for spec in arms:
        cells = point["runners"].get(spec.addressed)
        if not cells:
            raise ValueError(f"point {point['name']} sends no cell to runner {spec.addressed!r} ({spec.label}); it "
                             f"addresses {sorted(point['runners'])}")
        rows += [(f"{spec.label}|{cell['name']}", spec, cell) for cell in cells]
    names = [row for row, _spec, _cell in rows]
    reference = reference or labels[0]
    if reference not in names and reference not in labels:
        raise ValueError(f"reference {reference!r} is neither a label nor a row; rows are {names}")
    workers: dict[tuple, _Worker] = {}
    try:
        for spec in arms:
            if spec.key() not in workers:
                workers[spec.key()] = _Worker(spec)
        built: dict[str, dict] = {}
        for key, worker in workers.items():
            mine = [(row, spec, cell) for row, spec, cell in rows if spec.key() == key]
            records = list({cell["name"]: cell for _row, _spec, cell in mine}.values())
            pairs = sorted({(cell["name"], spec.arm) for _row, spec, cell in mine})
            reply = worker.ask({"op": "prepare", "cells": records, "arms": pairs},
                               ", ".join(sorted({spec.label for _row, spec, _cell in mine})))
            for row, spec, cell in mine:
                built[row] = reply["arms"][f"{cell['name']}|{spec.arm}"]
        instruments = {built[row]["instrument"] for row in names}
        if len(instruments) != 1:
            raise ValueError(f"one comparison, one stopwatch: the arms time with {sorted(instruments)}")

        def call(spec: ArmSpec, cell: dict) -> float:
            return workers[spec.key()].ask({"op": "call", "key": f"{cell['name']}|{spec.arm}"}, spec.label)["ms"]

        rng = random.Random(seed)
        samples: dict[str, list[float]] = {row: [] for row in names}
        order: list[str] = []
        with hold():
            for _ in range(warmup):
                for _row, spec, cell in rows:
                    call(spec, cell)
            for _ in range(rounds * reps):
                for row, spec, cell in rng.sample(rows, len(rows)):
                    samples[row].append(call(spec, cell))
                    order.append(row)
    finally:
        for worker in workers.values():
            worker.close()

    def blocks(values: list[float]) -> list[float]:
        return [statistics.median(values[i:i + reps]) for i in range(0, len(values), reps)]

    def against(row: str, cell: str) -> list[str]:
        if reference in names:
            return [] if row == reference else [reference]
        mine = [r for r, spec, _c in rows if spec.label == reference]
        if row in mine:
            return []
        same = [r for r, _spec, c in rows if r in mine and c["name"] == cell]
        return same or mine

    out = []
    for row, spec, cell in rows:
        ms = samples[row]
        entry = {"row": row, "label": spec.label, "runner": spec.addressed, "provider": spec.provider, "arm": spec.arm,
                 "cell": cell["name"], "built": built[row]["cell"], "ms": ms, "blocks_ms": blocks(ms),
                 "median_ms": statistics.median(blocks(ms)), "iqr_ms": iqr(blocks(ms)), "paired": []}
        for ref in against(row, cell["name"]):
            ratios = [a / r for a, r in zip(ms, samples[ref], strict=True)]
            entry["paired"].append({"reference": ref, "ratios": ratios, "ratio_median": statistics.median(ratios),
                                    "ratio_iqr": iqr(ratios),
                                    "round_diffs_ms": [a - r for a, r in zip(blocks(ms), blocks(samples[ref]),
                                                                            strict=True)]})
        out.append(entry)
    return {"point": point, "instrument": instruments.pop(), "rounds": rounds, "reps": reps, "warmup": warmup,
            "seed": seed, "reference": reference, "order": order, "arms": out}


def null_gate(spec: ArmSpec, point: dict, **options) -> dict:
    """ONE arm in TWO workers, interleaved: the check that a second process on the device biases nothing.

    Returns the comparison with `holds` added: whether a ratio of exactly one lies within the interquartile range of
    the per-rep ratios. A comparison across workers is trusted only once this holds for the arms it runs.
    """
    if len(point["runners"].get(spec.addressed, [])) != 1:
        raise ValueError(f"the null gate runs one arm on one cell; point {point['name']} sends runner "
                         f"{spec.addressed!r} {len(point['runners'].get(spec.addressed, []))}")
    twins = [ArmSpec(f"{spec.label}#{i}", spec.provider, spec.arm, spec.python, spec.cwd, spec.env, worker=f"null-{i}",
                     runner=spec.addressed) for i in (1, 2)]
    result = interleave(point, twins, **options)
    ratios = sorted(result["arms"][1]["paired"][0]["ratios"])
    low, high = ratios[int(0.25 * len(ratios))], ratios[int(0.75 * len(ratios))]
    return {**result, "holds": low <= 1.0 <= high}
