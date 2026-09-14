"""The public mirror's export (rola_devtools.mirror), in a throwaway git repository: every tracked path is declared, the
longest declaration wins, the export holds only committed shipping files, and the tripwire refuses agent files and
private-shaped strings, including in the tool's own text."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from rola_devtools import mirror

#: a home-directory path, joined so this file carries none; the fixtures below split their other private shapes the same
#: way, so the file passes the tripwire it tests
SOMEONES_HOME = "/".join(("", "home", "someone"))
DECLARED = {"public": "owner/repo", "ships": [".github", "README.md", "src"],
            "private": [".github/workflows/mirror.yml"], "allow": {}}


class Mirror(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name) / "repo"
        (self.repo / "src").mkdir(parents=True)
        (self.repo / ".github" / "workflows").mkdir(parents=True)
        (self.repo / ".github" / "mirror").mkdir()
        (self.repo / "README.md").write_text("a repository\n")
        (self.repo / "src" / "code.py").write_text("print('hi')\n")
        (self.repo / ".github" / "workflows" / "mirror.yml").write_text("name: mirror\n")
        (self.repo / ".github" / "mirror" / "declarations.json").write_text(json.dumps(DECLARED))
        self.git("init", "-q", "-b", "main")
        self.git("add", "-A")
        self.git("-c", "user.name=t", "-c", "user.email=t@example.org", "commit", "-q", "-m", "c")
        self.cwd = os.getcwd()
        os.chdir(self.repo)

    def tearDown(self) -> None:
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def git(self, *args: str) -> None:
        subprocess.run(["git", "-C", str(self.repo), *args], check=True, capture_output=True)

    def test_the_longest_declaration_wins(self):
        self.assertEqual(mirror.declaration(".github/mirror/declarations.json", DECLARED), "ships")
        self.assertEqual(mirror.declaration(".github/workflows/mirror.yml", DECLARED), "private")
        self.assertEqual(mirror.declaration("src/code.py", DECLARED), "ships")
        self.assertIsNone(mirror.declaration("srcx/code.py", DECLARED))

    def test_the_export_holds_the_committed_shipping_files_only(self):
        (self.repo / "src" / "uncommitted.py").write_text("x = 1\n")
        export = Path(self.tmp.name) / "export"
        self.assertEqual(mirror.export(export, mirror.declarations()), [])
        shipped = sorted(p.relative_to(export).as_posix() for p in export.rglob("*") if p.is_file())
        self.assertEqual(shipped, [".github/mirror/declarations.json", "README.md", "src/code.py"])

    def test_an_undeclared_tracked_path_refuses(self):
        (self.repo / "notes.txt").write_text("n\n")
        self.git("add", "notes.txt")
        self.assertEqual(mirror.main(["--check"]), 1)

    def test_the_tripwire_refuses_private_shaped_files_but_not_its_own_text(self):
        scan = Path(self.tmp.name) / "scan"
        (scan / "tool").mkdir(parents=True)
        shutil.copy(mirror.__file__, scan / "tool" / "mirror.py")
        self.assertEqual(mirror.tripwire(scan, {}), [])
        for name, text, finding in [("csrc/CLAUDE.md", "notes", "an agent file"),
                                    ("docs/a/.claude/settings.json", "{}", "an agent file"),
                                    ("docs/x.md", f"built at {SOMEONES_HOME}/rola", "a home directory"),
                                    ("docs/y.md", "contact someone" + "@gmail.com", "a personal email address"),
                                    ("tools/x.py", "token = 'ghp_" + "a" * 36 + "'", "a credential")]:
            path = scan / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            self.assertIn(f"{name}: {finding}", mirror.tripwire(scan, {}))
            path.unlink()

    def test_an_allowed_shape_passes_but_a_credential_can_never_be_allowed(self):
        scan = Path(self.tmp.name) / "scan"
        scan.mkdir()
        (scan / "r.json").write_text(f'{{"rola_file": "{SOMEONES_HOME}/rola/__init__.py"}}')
        self.assertEqual(mirror.tripwire(scan, {"a home directory": "records name their checkout"}), [])
        bad = scan / "declarations.json"
        bad.write_text('{"public": "o/r", "ships": [], "private": [], "allow": {"a credential": "no"}}')
        with self.assertRaisesRegex(SystemExit, "no allowable shape"):
            mirror.declarations(bad)


if __name__ == "__main__":
    unittest.main()
