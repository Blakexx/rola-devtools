"""ONE gpu_lock() every GPU-touching tool takes ITSELF (moved from rola's `tools/gpu_lock.py`; rola's
docs/KERNEL_STANDARDS.md §14/§18, "why even have agents manage locks").

**BEFORE IT.** The GPU lock was half-and-half: the probe harness and rola's `tools/sanitize_oracle.py` took the lock file
internally, but pytest and the bench harness relied on the CALLER wrapping the command in an EXTERNAL `flock` on the same
path. Two incidents came out of that split: a caller once wrapped a self-locking tool in such an external `flock` and
deadlocked 17 CPU-idle minutes (`flock` is held per open file description, so the wrapper's own lock blocks the wrapped
process's identical `fcntl.flock` call forever); and every brief had to restate the wrapping correctly by hand, a rule a
human has to remember rather than one a tool enforces.

**THE FIX.** Every GPU entry point calls `gpu_lock()` itself -- rola's pytest session fixture, `tools/sanitize_oracle.py`
and `tools/compare.py`, and the measurement service. Nothing is invoked wrapped in an external `flock` again (rola's
`tools/lint/lint_standards.py` finds the word if it comes back).

**REENTRANT BY CONSTRUCTION.** Acquiring the lock sets `ROLA_GPU_LOCK_HELD` in the process environment; a nested acquire
that finds the marker set is a no-op. `os.environ` is inherited by every subprocess a locked process spawns, so this
covers same-process re-entry (a fixture calling a helper that also locks) and the subprocess case (a tool locking once,
then launching `compute-sanitizer python -m pytest ...`, whose own fixture sees the marker and proceeds) -- a double
acquisition cannot deadlock by construction.

**THE LOCK FILE.** `host.gpu_lock` in the dev config, default `/tmp/rola_gpu.lock`. `/tmp` is not a home path and is the
directory every GPU-touching container bind-mounts from the host, so a container and the bare host serialize against the
same lock.

**TWO MODES (LOCKS brief, 2026-08-29).** Blake: "allow some controlled level of parallelism" -- MEASURED work needs the
device to itself, nothing else touching it while a number is read; CORRECTNESS work (the oracle, integration and CUDA
unit tiers, the sanitizer) only needs the device not to be mid-measurement, and two such runs interleaving corrupts
neither's answer. `mode="exclusive"` (default) holds the reader/writer file exclusively, so it also excludes every
shared holder. `mode="shared"` holds it shared plus one of `host.gpu_shared_slots` slot files, so that many correctness
runs proceed concurrently. The gate/iteration shape of `rola_devtools.locks.host`, applied to a fixed pool.
"""
from __future__ import annotations

import contextlib
import fcntl
import os
import time

from ..config import machine
from . import host

#: Set to this process's pid on acquire, cleared on release. Presence alone (not an
#: exact pid match) is what a nested acquire checks -- a marker inherited from an
#: ancestor process through `exec` still proves "an ancestor already holds this",
#: which is the case that matters (see module docstring's subprocess case).
HELD_MARKER = "ROLA_GPU_LOCK_HELD"


def _open_ex(path: str) -> int:
    return os.open(path, os.O_CREAT | os.O_RDWR, 0o666)


def _blocking_acquire(path: str, lock_type: int):
    fd = _open_ex(path)
    fcntl.flock(fd, lock_type)
    return fd


def _try_acquire(path: str, lock_type: int):
    fd = _open_ex(path)
    try:
        fcntl.flock(fd, lock_type | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


@contextlib.contextmanager
def gpu_lock(path: str | None = None, *, mode: str = "exclusive"):
    """GPU access, held for the `with` block's duration.

    Built on ONE reader/writer lock file (`{base}.rw`, native `flock(LOCK_SH
    | LOCK_EX)`) plus a `host.gpu_shared_slots`-sized counting pool that caps
    concurrent SHARED holders, plus one admission mutex (`{base}.wq`) that
    closes a starvation gap PLAIN `flock()` leaves open: Linux's `flock()`
    does not give a PENDING exclusive request priority over a fresh shared
    one -- a new `LOCK_SH` succeeds immediately as long as no `LOCK_EX` is
    CURRENTLY GRANTED, so a steady stream of shared acquirers can keep an
    exclusive acquirer waiting forever even though it asked first (found by
    this file's own gate: a three-process repro where a second shared
    acquirer walked straight past a pending exclusive one). Every acquirer,
    either mode, takes `{base}.wq` FIRST and holds it for its whole admission
    step: `mode="exclusive"` holds `wq` across the (possibly long) wait for
    `rw`'s `LOCK_EX`, so no new shared acquirer can even attempt `rw` while an
    exclusive request is outstanding; `mode="shared"` holds `wq` only across
    its own (uncontended, since a live shared holder never blocks another)
    `LOCK_SH` grab, then releases it immediately -- so shared acquirers still
    admit concurrently once no writer is queued, and only serialize with each
    other for the instant it takes to join.

    Reentrant: if `ROLA_GPU_LOCK_HELD` is already set (this process or an ancestor
    already holds a `gpu_lock()`, in EITHER mode), this is a no-op context -- it
    neither opens nor waits on any lock file a second time.
    """
    if mode not in ("exclusive", "shared"):
        raise ValueError(f"gpu_lock: mode must be 'exclusive' or 'shared', got {mode!r}")
    if HELD_MARKER in os.environ:
        yield
        return
    base = path or machine("host.gpu_lock")
    wq_path = f"{base}.wq"
    rw_path = f"{base}.rw"
    slot_paths = [f"{base}.slot{i}" for i in range(machine("host.gpu_shared_slots"))]
    held: list[int] = []
    try:
        if mode == "exclusive":
            wq_fd = _blocking_acquire(wq_path, fcntl.LOCK_EX)
            try:
                held.append(_blocking_acquire(rw_path, fcntl.LOCK_EX))
            finally:
                fcntl.flock(wq_fd, fcntl.LOCK_UN)
                os.close(wq_fd)
        else:
            wq_fd = _blocking_acquire(wq_path, fcntl.LOCK_EX)
            try:
                held.append(_blocking_acquire(rw_path, fcntl.LOCK_SH))
            finally:
                fcntl.flock(wq_fd, fcntl.LOCK_UN)
                os.close(wq_fd)
            for slot in slot_paths:
                fd = _try_acquire(slot, fcntl.LOCK_EX)
                if fd is not None:
                    held.append(fd)
                    break
            if len(held) == 1:
                #: NEVER a hardcoded slot index here: which slot a given
                #: holder ends up with is a race, so waiting on slot 0
                #: specifically can wait on a slot nobody is about to free
                #: while a different one frees immediately (this module's own
                #: gate found it: releasing "the other" shared holder never
                #: unblocked a waiter pinned to slot 0). Poll the whole sweep
                #: instead -- any slot freeing is progress.
                while len(held) == 1:
                    for slot in slot_paths:
                        fd = _try_acquire(slot, fcntl.LOCK_EX)
                        if fd is not None:
                            held.append(fd)
                            break
                    if len(held) == 1:
                        time.sleep(0.01)
        os.environ[HELD_MARKER] = str(os.getpid())
        host.lower_priority("gpu_lock")
        try:
            yield
        finally:
            os.environ.pop(HELD_MARKER, None)
    finally:
        for fd in held:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
