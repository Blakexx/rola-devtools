"""THE RUNTIME: one worker per environment for a build, and the requests that run targets in them.

A target's environment is its `env`, or the build system's own (`build_env`: this interpreter, in the root's directory).
Two environments with one label must be the same python and directory. A worker that exits or whose protocol differs
fails the target that asked for it; the next request starts a fresh one.
"""
from __future__ import annotations

import contextlib
import json
import os
import selectors
import subprocess
import sys
from pathlib import Path

from ..process import stop_tree
from .declare import Env, Target
from .worker import PROTOCOL


class _Worker:
    def __init__(self, env: Env) -> None:
        self.env = env
        self.process = subprocess.Popen([env.python, "-m", "rola_devtools.build.worker"], cwd=env.cwd,
                                        env={**os.environ, **env.vars}, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        text=True, start_new_session=True)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        protocol = self.ask({"op": "hello"}, f"{env.label}: hello")["protocol"]
        if protocol != PROTOCOL:
            self.close()
            raise RuntimeError(f"{env.label}: its rola_devtools speaks build protocol {protocol}, this build system "
                               f"{PROTOCOL}")

    @property
    def alive(self) -> bool:
        return self.process.poll() is None

    def ask(self, request: dict, what: str) -> dict:
        try:
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as ex:
            raise RuntimeError(f"{what}: the {self.env.label} worker is gone (exit {self.process.poll()})") from ex
        self.selector.select()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"{what}: the {self.env.label} worker exited (exit {self.process.wait()})")
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


class Runtime:
    def __init__(self, root: Path | str) -> None:
        self.build_env = Env("build", sys.executable, str(Path(root).resolve()),
                             {"PYTHONPATH": os.pathsep.join(filter(None, [str(Path(root).resolve()),
                                                                          os.environ.get("PYTHONPATH", "")]))})
        self._workers: dict[str, _Worker] = {}
        self._envs: dict[str, Env] = {}

    def __enter__(self) -> Runtime:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def env(self, target: Target) -> Env:
        return target.env or self.build_env

    def worker(self, env: Env) -> _Worker:
        known = self._envs.setdefault(env.label, env)
        if (known.python, known.cwd, known.vars) != (env.python, env.cwd, env.vars):
            raise ValueError(f"two environments are labelled {env.label}")
        worker = self._workers.get(env.label)
        if worker is None or not worker.alive:
            worker = self._workers[env.label] = _Worker(env)
        return worker

    def run(self, target: Target, context: dict, markers: dict) -> object:
        request = {"op": "run", "executor": target.executor, "context": context, "env": markers}
        return self.worker(self.env(target)).ask(request, target.label)["output"]

    def verify(self, target: Target, output) -> bool:
        return self.worker(self.env(target)).ask({"op": "verify", "verify": target.verify, "output": output},
                                                 target.label)["ok"]

    def close(self) -> None:
        for worker in self._workers.values():
            worker.close()
        self._workers.clear()


__all__ = ["Runtime"]
