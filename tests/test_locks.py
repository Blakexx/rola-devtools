"""The machine's locks, proven with fake children: small driver processes that hold a lock and wait to be told to
release, never a compile or a GPU kernel, every lock file under a throwaway directory (moved from rola's
tests/unit/test_locks_accounting.py and test_lock_priority.py).

- N+1 acquirers of an N-slot host budget: the (N+1)th blocks, and releasing one of the N admits it.
- A nested acquire under an ancestor's marker is a no-op that opens no lock file.
- The GPU lock's modes: shared holders share a fixed pool; an exclusive acquire waits for every shared holder, and a
  shared acquire queued behind it waits too.
- A held GPU lock lowers the holder's priority, so a child it starts runs at nice 10, unless `host.nice` is false.
- The dev config reader refuses an undeclared key and an undeclared file, and takes a missing key's default.
`python -m unittest tests.test_locks`"""
from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rola_devtools import config
from rola_devtools.locks import host

ROOT = Path(__file__).resolve().parents[1]
TIMEOUT_S = 5.0

#: holds one host-budget slot, announces it, and blocks on stdin until told to let go
HOLD_HOST_BUDGET = ("import sys\n"
                    "from rola_devtools.locks import host\n"
                    "with host.acquire(1, label='test-hold'):\n"
                    "    print('ACQUIRED', flush=True)\n"
                    "    sys.stdin.readline()\n")
#: the same for the GPU lock: argv[1] the lock path, argv[2] the mode
HOLD_GPU_LOCK = ("import sys\n"
                 "from rola_devtools.locks.gpu import gpu_lock\n"
                 "with gpu_lock(sys.argv[1], mode=sys.argv[2]):\n"
                 "    print('ACQUIRED', flush=True)\n"
                 "    sys.stdin.readline()\n")
#: field 19 of /proc/self/stat (proc(5)) is the nice value; `comm` may hold spaces, so read past its closing paren
READ_NICE = ("import pathlib, sys\n"
             "raw = pathlib.Path('/proc/self/stat').read_text()\n"
             "sys.stdout.write(raw[raw.rindex(')') + 2:].split()[19 - 3])\n")
NICE_UNDER_GPU_LOCK = ("import subprocess, sys\n"
                       "from rola_devtools.locks.gpu import gpu_lock\n"
                       "with gpu_lock(sys.argv[1]):\n"
                       f"    sys.stdout.write(subprocess.run([sys.executable, '-c', {READ_NICE!r}], capture_output=True, "
                       "text=True, check=True).stdout)\n")


class Configured(unittest.TestCase):
    """A dev config directory of this test's own, handed to children through `ROLA_DEV_CONFIG`."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def env(self, **sections: dict) -> dict:
        root = self.tmp / f"config{len(list(self.tmp.glob('config*')))}"
        root.mkdir()
        for section, values in sections.items():
            (root / f"{section}.json").write_text(json.dumps(values))
        env = {k: v for k, v in os.environ.items() if k not in ("ROLA_HOST_BUDGET_HELD", "ROLA_GPU_LOCK_HELD")}
        env[config.POINTER] = str(root)
        env["PYTHONPATH"] = str(ROOT)
        return env

    def spawn(self, source: str, env: dict, *args: str) -> subprocess.Popen:
        proc = subprocess.Popen([sys.executable, "-c", source, *args], env=env, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        self.addCleanup(self.reap, proc)
        return proc

    @staticmethod
    def reap(proc: subprocess.Popen) -> None:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            stream.close()

    @staticmethod
    def line(proc: subprocess.Popen, timeout: float) -> str | None:
        """One line of the child's output, or None when none arrives in time: a blocked acquirer prints nothing."""
        ready, _, _ = select.select([proc.stdout], [], [], timeout)
        return proc.stdout.readline() if ready else None

    def release(self, proc: subprocess.Popen) -> None:
        proc.stdin.write("\n")
        proc.stdin.close()
        self.assertEqual(proc.wait(timeout=TIMEOUT_S), 0, proc.stderr.read())


class HostBudget(Configured):
    def test_n_plus_one_acquirers_block_and_a_release_admits_the_last(self):
        env = self.env(host={"lock_dir": str(self.tmp), "budget_slots": 2, "nice": False})
        holders = [self.spawn(HOLD_HOST_BUDGET, env) for _ in range(2)]
        for h in holders:
            self.assertEqual(self.line(h, TIMEOUT_S), "ACQUIRED\n")
        blocked = self.spawn(HOLD_HOST_BUDGET, env)
        self.assertIsNone(self.line(blocked, 1.0))
        self.release(holders[0])
        self.assertEqual(self.line(blocked, TIMEOUT_S), "ACQUIRED\n")
        self.release(blocked)
        self.release(holders[1])

    def test_a_nested_acquire_under_an_ancestors_marker_opens_nothing(self):
        locks = self.tmp / "locks"
        locks.mkdir()
        env = self.env(host={"lock_dir": str(locks), "budget_slots": 1, "nice": False})
        saved = {k: os.environ.get(k) for k in (config.POINTER, "ROLA_HOST_BUDGET_HELD")}
        os.environ[config.POINTER] = env[config.POINTER]
        os.environ["ROLA_HOST_BUDGET_HELD"] = "999999"
        config.reload()
        try:
            with host.acquire(5, label="nested") as held:
                self.assertEqual(held, 5)
            self.assertEqual(list(locks.iterdir()), [])
        finally:
            for k, v in saved.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
            config.reload()


class GpuLock(Configured):
    def test_the_shared_pool_admits_two_and_blocks_a_third(self):
        path = str(self.tmp / "gpu.lock")
        env = self.env(host={"gpu_shared_slots": 2, "nice": False})
        a, b = (self.spawn(HOLD_GPU_LOCK, env, path, "shared") for _ in range(2))
        self.assertEqual(self.line(a, TIMEOUT_S), "ACQUIRED\n")
        self.assertEqual(self.line(b, TIMEOUT_S), "ACQUIRED\n")
        c = self.spawn(HOLD_GPU_LOCK, env, path, "shared")
        self.assertIsNone(self.line(c, 1.0))
        self.release(a)
        self.assertEqual(self.line(c, TIMEOUT_S), "ACQUIRED\n")
        self.release(c)
        self.release(b)

    def test_exclusive_waits_for_every_shared_holder_and_a_queued_exclusive_holds_back_shared(self):
        path = str(self.tmp / "gpu.lock")
        env = self.env(host={"gpu_shared_slots": 2, "nice": False})
        shared = self.spawn(HOLD_GPU_LOCK, env, path, "shared")
        self.assertEqual(self.line(shared, TIMEOUT_S), "ACQUIRED\n")
        exclusive = self.spawn(HOLD_GPU_LOCK, env, path, "exclusive")
        self.assertIsNone(self.line(exclusive, 1.0))
        other = self.spawn(HOLD_GPU_LOCK, env, path, "shared")
        self.assertIsNone(self.line(other, 1.0))
        other.kill()
        other.wait(timeout=TIMEOUT_S)
        self.release(shared)
        self.assertEqual(self.line(exclusive, TIMEOUT_S), "ACQUIRED\n")
        self.release(exclusive)

    def test_a_holder_lowers_its_childrens_priority_unless_nice_is_off(self):
        path = str(self.tmp / "gpu.lock")
        for nice, expected in ((True, 10), (False, 0)):
            out = subprocess.run([sys.executable, "-c", NICE_UNDER_GPU_LOCK, path], env=self.env(host={"nice": nice}),
                                 capture_output=True, text=True, check=True, timeout=30)
            self.assertEqual(int(out.stdout.strip().splitlines()[-1]), expected, out.stderr)


class Reader(Configured):
    def read(self, env: dict, schema: dict, owns_directory: bool) -> dict:
        saved = os.environ.get(config.POINTER)
        os.environ[config.POINTER] = env[config.POINTER]
        try:
            return config.read(schema, owns_directory=owns_directory)
        finally:
            os.environ.pop(config.POINTER) if saved is None else os.environ.__setitem__(config.POINTER, saved)

    def test_a_missing_key_takes_its_default_and_names_its_source(self):
        env = self.env(host={"gpu_shared_slots": 3})
        read = self.read(env, config.MACHINE, owns_directory=False)
        self.assertEqual(read["host"]["gpu_shared_slots"], (3, str(Path(env[config.POINTER]) / "host.json")))
        self.assertEqual(read["host"]["nice"], (True, "default"))

    def test_an_undeclared_key_and_a_wrong_kind_are_refused(self):
        with self.assertRaises(SystemExit):
            self.read(self.env(host={"gpu_slots": 3}), config.MACHINE, owns_directory=False)
        with self.assertRaises(SystemExit):
            self.read(self.env(host={"nice": "yes"}), config.MACHINE, owns_directory=False)

    def test_an_undeclared_file_is_refused_only_by_the_directorys_owner(self):
        env = self.env(host={}, toolchain={"cuda_home": "/usr/local/cuda"})
        self.assertIn("host", self.read(env, config.MACHINE, owns_directory=False))
        with self.assertRaises(SystemExit):
            self.read(env, config.MACHINE, owns_directory=True)


if __name__ == "__main__":
    unittest.main()
