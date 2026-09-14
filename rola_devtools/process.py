"""A COMMAND STOPPED WHOLE: `run` is `subprocess.run` for a command that starts processes of its own.

A profiler (`ncu`) runs the measured process as a child of its own, and a driver runs its workers the same way. Stopping
only the command leaves them running: rola-bench's suite once timed out units that way and left nine profiled processes
holding the GPU while every later unit measured beside them. `run` starts the command as a process group of its own, and
a timeout, an interrupt or the caller's own exit stops the command, its group and every descendant found before the first
signal (a descendant can start a group of its own): SIGTERM, then SIGKILL to whatever outlives the grace. Linux (`/proc`).
"""
from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import time

#: seconds a stopped tree has to exit on SIGTERM before SIGKILL
GRACE_S = 10.0


def run(cmd: list[str], *, cwd=None, env: dict | None = None, timeout: float | None = None) -> subprocess.CompletedProcess:
    """The command's `CompletedProcess` with its text output captured; `subprocess.TimeoutExpired` once its tree is
    stopped."""
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout)
    except BaseException:
        stop_tree(proc)
        raise
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


def descendants(pid: int) -> list[int]:
    """Every live descendant of `pid`, read from `/proc`."""
    children: dict[int, list[int]] = {}
    for stat in os.listdir("/proc"):
        if stat.isdigit():
            with contextlib.suppress(OSError, ValueError, IndexError), open(f"/proc/{stat}/stat") as f:
                children.setdefault(int(f.read().rpartition(")")[2].split()[1]), []).append(int(stat))
    found, frontier = [], [pid]
    while frontier:
        frontier = [child for parent in frontier for child in children.get(parent, [])]
        found += frontier
    return found


def alive(pid: int) -> bool:
    """Whether `pid` is a process that has not exited (a zombie has)."""
    with contextlib.suppress(OSError, IndexError), open(f"/proc/{pid}/stat") as f:
        return f.read().rpartition(")")[2].split()[0] != "Z"
    return False


def stop_tree(proc: subprocess.Popen, grace_s: float = GRACE_S) -> None:
    """Stop `proc`, its process group and its descendants, and reap `proc`."""
    tree = [proc.pid, *descendants(proc.pid)]
    for sig, wait in ((signal.SIGTERM, grace_s), (signal.SIGKILL, grace_s)):
        with contextlib.suppress(OSError):
            os.killpg(proc.pid, sig)
        for pid in tree:
            with contextlib.suppress(OSError):
                os.kill(pid, sig)
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            proc.poll()
            if not any(alive(pid) for pid in tree):
                break
            time.sleep(0.1)
        if not any(alive(pid) for pid in tree):
            break
    with contextlib.suppress(subprocess.TimeoutExpired, ValueError, OSError):
        proc.communicate(timeout=grace_s)
