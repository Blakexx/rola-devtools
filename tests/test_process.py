"""A command stopped whole: a timeout ends the command and every process it started, including one that left the
command's process group the way a profiler's measured process can."""
from __future__ import annotations

import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from rola_devtools.process import alive, run

#: starts a grandchild in a session of its own, writes its pid, and waits on it
SPAWNER = ("import subprocess, sys; child = subprocess.Popen(['sleep', '300'], start_new_session=True); "
           "open(sys.argv[1], 'w').write(str(child.pid)); child.wait()")


class StopsTheTree(unittest.TestCase):
    def test_a_timeout_stops_a_grandchild_outside_the_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            pidfile = Path(tmp) / "pid"
            with self.assertRaises(subprocess.TimeoutExpired):
                run([sys.executable, "-c", SPAWNER, str(pidfile)], cwd=tmp, timeout=2)
            grandchild = int(pidfile.read_text())
            deadline = time.monotonic() + 5
            while alive(grandchild) and time.monotonic() < deadline:
                time.sleep(0.1)
            self.assertFalse(alive(grandchild), "the grandchild outlived the command")

    def test_plain_subprocess_run_leaves_it_behind(self):
        with tempfile.TemporaryDirectory() as tmp:
            pidfile = Path(tmp) / "pid"
            with self.assertRaises(subprocess.TimeoutExpired):
                subprocess.run([sys.executable, "-c", SPAWNER, str(pidfile)], cwd=tmp, capture_output=True, timeout=2)
            grandchild = int(pidfile.read_text())
            try:
                self.assertTrue(alive(grandchild), "the control no longer shows the defect")
            finally:
                subprocess.run(["kill", str(grandchild)], check=False)

    def test_a_command_that_finishes_returns_its_code_and_output(self):
        done = run([sys.executable, "-c", "import sys; print('out'); sys.stderr.write('err'); sys.exit(3)"], timeout=30)
        self.assertEqual((done.returncode, done.stdout, done.stderr), (3, "out\n", "err"))


if __name__ == "__main__":
    unittest.main()
