"""The build graph's shared machinery (`rola_devtools.graph`): the front door overrides a workspace's `rola_devtools` pin
with this checkout only where the workspace depends on the module, since Bazel refuses an override of a module the
graph does not hold; and the environment's store key follows a local package's content, its leftovers aside."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from rola_devtools import config
from rola_devtools.graph import front_door, make_env


class FrontDoor(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        (root / "config").mkdir()
        (root / "config" / "host.json").write_text(f'{{"build_cache": "{root / "cache"}"}}')
        self._saved = os.environ.get(config.POINTER)
        os.environ[config.POINTER] = str(root / "config")
        config.reload()
        self.root = root

    def tearDown(self):
        if self._saved is None:
            os.environ.pop(config.POINTER, None)
        else:
            os.environ[config.POINTER] = self._saved
        config.reload()
        self._tmp.cleanup()

    def _workspace(self, module: str) -> Path:
        workspace = self.root / "workspace"
        workspace.mkdir(exist_ok=True)
        (workspace / "MODULE.bazel").write_text(module)
        return workspace

    def test_a_workspace_that_depends_on_the_module_gets_this_checkout_as_its_override(self):
        workspace = self._workspace('module(name = "w")\nbazel_dep(name = "rola_devtools", version = "0.1.0")\n')
        rc = front_door.workspace_rc(workspace)
        self.assertIsNotNone(rc)
        self.assertIn(f"--override_module=rola_devtools={front_door.CHECKOUT}", rc.read_text())

    def test_a_workspace_that_does_not_gets_no_override(self):
        self.assertIsNone(front_door.workspace_rc(self._workspace('module(name = "w")\nbazel_dep(name = "other")\n')))

    def test_the_machine_rc_names_the_machine_s_caches(self):
        text = front_door.machine_rc().read_text()
        self.assertIn(f"--disk_cache={self.root / 'cache' / 'bazel' / 'disk'}", text)


class StoreKey(unittest.TestCase):
    def test_a_local_package_s_key_follows_its_content_and_not_its_leftovers(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp)
            (package / "m.py").write_text("x = 1\n")
            first = make_env._identity(str(package))
            (package / "__pycache__").mkdir()
            (package / "__pycache__" / "m.cpython-311.pyc").write_bytes(b"\0")
            self.assertEqual(make_env._identity(str(package)), first)
            (package / "m.py").write_text("x = 2\n")
            self.assertNotEqual(make_env._identity(str(package)), first)

    def test_a_requirement_s_key_is_its_text(self):
        pin = "zoology @ git+https://github.com/HazyResearch/zoology@0123456789abcdef0123456789abcdef01234567"
        self.assertEqual(make_env._identity(pin), pin.encode())


if __name__ == "__main__":
    unittest.main()
