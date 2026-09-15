"""REQUIREMENTS: what a target holds while it runs, as counting semaphores over the machine's locks.

A target declares `holds={"gpu": "all", "clock": 1}`. Each resource has a capacity its provider reads from the dev
config; a claim is a count of slots or "all" (exclusive, the Bazel / Ninja / Gradle / Airflow model):

    gpu        capacity host.gpu_shared_slots; "all" is the GPU lock exclusive (a measurement), 1 a shared slot (a
               correctness run); a shared claim of more than one slot is refused
    host_cpu   capacity host.budget_slots; "all" is every slot, n waits for one and takes up to n (the host budget's draw)
    clock      capacity 1; holding it engages the host's clock lock for this process (released at exit); the proof
               is the reader's, against `rola_devtools.locks.clock.within`

The providers take the machine-wide locks, so tools and agents outside the build are excluded as well; every claim of a
target is taken in name order and released together, and the lock-held markers go to the target's worker, so a tool the
target starts sees what its build holds. The scheduler is sequential today, so claims never queue against each other;
a concurrent scheduler adds a fair queue here without changing a declaration.
"""
from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator

from ..config import machine
from ..locks import clock, gpu, host


@contextlib.contextmanager
def _gpu(claim) -> Iterator[None]:
    if claim not in ("all", 1):
        raise ValueError(f"the GPU is held as 'all' (exclusive) or 1 (a shared slot), not {claim!r}")
    with gpu.gpu_lock(mode="exclusive" if claim == "all" else "shared"):
        yield


@contextlib.contextmanager
def _host_cpu(claim) -> Iterator[None]:
    total = host.budget_slots()
    if claim != "all" and claim > total:
        raise ValueError(f"host_cpu holds {claim} of {total} slots")
    with host.acquire(total if claim == "all" else claim, exclusive=claim == "all", label="build"):
        yield


@contextlib.contextmanager
def _clock(claim) -> Iterator[None]:
    if claim not in ("all", 1):
        raise ValueError(f"the clock is held as 1, not {claim!r}")
    clock.lock()
    yield


PROVIDERS = {"gpu": _gpu, "host_cpu": _host_cpu, "clock": _clock}
#: the environment variables a held lock sets, handed to the worker that runs the holding target
MARKERS = (gpu.HELD_MARKER, host.HELD_MARKER)


def capacity(name: str) -> int:
    return {"gpu": machine("host.gpu_shared_slots"), "host_cpu": host.budget_slots(), "clock": 1}[name]


@contextlib.contextmanager
def hold(holds: dict) -> Iterator[dict[str, str]]:
    """Every claim of `holds`, taken in name order and released together; yields the lock-held markers set."""
    unknown = sorted(set(holds) - set(PROVIDERS))
    if unknown:
        raise KeyError(f"no resource {unknown}; the resources are {sorted(PROVIDERS)}")
    with contextlib.ExitStack() as stack:
        for name in sorted(holds):
            stack.enter_context(PROVIDERS[name](holds[name]))
        yield {marker: os.environ[marker] for marker in MARKERS if marker in os.environ}


__all__ = ["MARKERS", "PROVIDERS", "capacity", "hold"]
