"""The driver's side: workers started per provider environment, arms prepared, warmed and interleaved per call."""
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
    """One row of a comparison: `label` names it, `provider` (module:function) builds its arms in a worker run by
    `python` in `cwd` with `env` added, and `arm` is the provider's name for it. Specs that agree on all four share a
    worker; `worker` names a worker explicitly, so one environment can be two processes."""

    label: str
    provider: str
    arm: str
    python: str = sys.executable
    cwd: str | None = None
    env: dict = field(default_factory=dict)
    worker: str = ""

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


def interleave(point: dict, arms: list[ArmSpec], *, matching: str, rounds: int = 8, reps: int = 11,
               warmup: int = WARMUP_FLOOR, reference: str | None = None, seed: int = 0,
               hold: Callable[[], contextlib.AbstractContextManager] = contextlib.nullcontext) -> dict:
    """Every arm at `point`, each rep calling each arm once in a fresh random order; the record of it.

    `matching` names the rule that makes the point fair across the libraries. `hold` is the context the timed calls run
    under (the caller's GPU lock and clock lock). `reference` is the label the paired ratios divide by (the first arm by
    default). Returns the point, the matching rule, the stopwatch, every arm's cell and raw samples in the order they
    were taken, per-round block medians, and each arm's paired ratios and per-round differences against the reference.
    """
    labels = [spec.label for spec in arms]
    if len(set(labels)) != len(labels):
        raise ValueError(f"arm labels must be unique, got {labels}")
    if warmup < WARMUP_FLOOR:
        raise ValueError(f"REFUSING to measure with {warmup} warmup launches; the floor is {WARMUP_FLOOR}")
    if reps % 2 == 0:
        raise ValueError(f"reps per round must be odd so a block median is a sample, got {reps}")
    reference = reference or labels[0]
    if reference not in labels:
        raise ValueError(f"reference {reference!r} is not an arm label; labels are {labels}")
    workers: dict[tuple, _Worker] = {}
    try:
        for spec in arms:
            if spec.key() not in workers:
                workers[spec.key()] = _Worker(spec)
        built: dict[str, dict] = {}
        for key, worker in workers.items():
            mine = [s for s in arms if s.key() == key]
            reply = worker.ask({"op": "prepare", "point": point, "arms": sorted({s.arm for s in mine})},
                               ", ".join(s.label for s in mine))
            for spec in mine:
                built[spec.label] = reply["arms"][spec.arm]
        instruments = {built[label]["instrument"] for label in labels}
        if len(instruments) != 1:
            raise ValueError(f"one comparison, one stopwatch: the arms time with {sorted(instruments)}")

        def call(spec: ArmSpec) -> float:
            return workers[spec.key()].ask({"op": "call", "arm": spec.arm}, spec.label)["ms"]

        rng = random.Random(seed)
        samples: dict[str, list[float]] = {label: [] for label in labels}
        order: list[str] = []
        with hold():
            for _ in range(warmup):
                for spec in arms:
                    call(spec)
            for _ in range(rounds * reps):
                for spec in rng.sample(arms, len(arms)):
                    samples[spec.label].append(call(spec))
                    order.append(spec.label)
    finally:
        for worker in workers.values():
            worker.close()

    def blocks(values: list[float]) -> list[float]:
        return [statistics.median(values[i:i + reps]) for i in range(0, len(values), reps)]

    rows = []
    for spec in arms:
        ms = samples[spec.label]
        row = {"label": spec.label, "provider": spec.provider, "arm": spec.arm, "cell": built[spec.label]["cell"],
               "ms": ms, "blocks_ms": blocks(ms), "median_ms": statistics.median(blocks(ms)), "iqr_ms": iqr(blocks(ms))}
        if spec.label != reference:
            ratios = [a / r for a, r in zip(ms, samples[reference], strict=True)]
            row["paired"] = {"reference": reference, "ratios": ratios, "ratio_median": statistics.median(ratios),
                             "ratio_iqr": iqr(ratios),
                             "round_diffs_ms": [a - r for a, r in zip(blocks(ms), blocks(samples[reference]),
                                                                     strict=True)]}
        rows.append(row)
    return {"point": point, "matching": matching, "instrument": instruments.pop(), "rounds": rounds, "reps": reps,
            "warmup": warmup, "seed": seed, "reference": reference, "order": order, "arms": rows}


def null_gate(spec: ArmSpec, point: dict, **options) -> dict:
    """ONE arm in TWO workers, interleaved: the check that a second process on the device biases nothing.

    Returns the comparison with `holds` added: whether a ratio of exactly one lies within the interquartile range of
    the per-rep ratios. A comparison across workers is trusted only once this holds for the arms it runs.
    """
    twins = [ArmSpec(f"{spec.label}#{i}", spec.provider, spec.arm, spec.python, spec.cwd, spec.env, worker=f"null-{i}")
             for i in (1, 2)]
    result = interleave(point, twins, matching="the same arm in two workers", **options)
    ratios = sorted(result["arms"][1]["paired"]["ratios"])
    low, high = ratios[int(0.25 * len(ratios))], ratios[int(0.75 * len(ratios))]
    return {**result, "holds": low <= 1.0 <= high}
