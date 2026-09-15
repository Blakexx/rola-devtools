"""THE TIMING SERVER'S POOL: one entry worker per checkout environment, started on first use, stopped by stop_timing_server.

It lives in the build system's own worker for the whole build (a module-level object the timing executors share), so
workers and whatever the device holds between sessions are the pool's, and a session sets up and drops its own entries.
"""
from __future__ import annotations

import contextlib
import json
import os
import selectors
import subprocess

from ..process import stop_tree


class EntryWorker:
    def __init__(self, env: dict) -> None:
        self.env = env
        vars_ = {k: v for k, v in (("PYTHONPATH", env.get("pythonpath")),) if v}
        self.process = subprocess.Popen([env["python"], "-m", "rola_devtools.timing.entry_worker"], cwd=env["cwd"],
                                        env={**os.environ, **vars_}, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, start_new_session=True)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)

    @property
    def alive(self) -> bool:
        return self.process.poll() is None

    def ask(self, request: dict, what: str) -> dict:
        try:
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as ex:
            raise RuntimeError(f"{what}: the entry worker is gone (exit {self.process.poll()})") from ex
        self.selector.select()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"{what}: the entry worker exited (exit {self.process.wait()})")
        reply = json.loads(line)
        if "error" in reply:
            raise RuntimeError(f"{what}: {request['op']} failed:\n{reply['error']}")
        return reply

    def close(self) -> None:
        with contextlib.suppress(BrokenPipeError, ValueError, OSError):
            self.process.stdin.write(json.dumps({"op": "exit"}) + "\n")
            self.process.stdin.flush()
        try:
            self.process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            stop_tree(self.process)
        self.selector.close()
        for pipe in (self.process.stdin, self.process.stdout):
            with contextlib.suppress(OSError):
                pipe.close()


class Pool:
    def __init__(self) -> None:
        self.workers: dict[tuple, EntryWorker] = {}

    def worker(self, env: dict) -> EntryWorker:
        key = (env["python"], env["cwd"], env.get("pythonpath") or "", env["label"])
        worker = self.workers.get(key)
        if worker is None or not worker.alive:
            worker = self.workers[key] = EntryWorker(env)
        return worker

    def close(self) -> None:
        for worker in self.workers.values():
            worker.close()
        self.workers.clear()


#: the server of this build system worker, set by start_timing_server
SERVER: list[Pool] = []


def server() -> Pool:
    if not SERVER:
        raise RuntimeError("no timing server in this build: declare start_timing_server and depend on it")
    return SERVER[0]


__all__ = ["EntryWorker", "Pool", "SERVER", "server"]
