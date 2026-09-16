"""The declared build system: declaration files loaded by path and composed with scoped labels, keys that no label or
path enters, a cache that skips and a verify that re-runs, build failures that stop the build while always_run targets
still run, domain failures that do not, requirements held around a target with their markers in its worker, and a
target's output reaching its dependents. `python -m unittest tests.test_declared_build`"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from rola_devtools import config
from rola_devtools.build.cache import Cache
from rola_devtools.build.declare import Env, Graph, load
from rola_devtools.build.runtime import Runtime
from rola_devtools.build.scheduler import build
from rola_devtools.cells import Registry

HERE = Path(__file__).resolve().parent
DECLARE = '''
def declare(g, env, n):
    first = g.node("first", executor="fake_executors:echo", env=env, params={"n": n},
                   inputs=[g.node("cells/t8", executor="fake_executors:record", params={"cell": "t8"})])
    second = g.node("second", executor="fake_executors:echo", env=env, deps={"first": first})
    return {"first": first, "second": second, "all": g.group("all", [first, second])}
'''


class Declared(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "ledger").write_text("")
        (self.root / "cells.json").write_text(json.dumps({"schema": 1, "data": "fake_provider:tokens",
                                                           "cells": [{"name": "t8", "tokens": 8}]}))
        (self.root / "declare.py").write_text(DECLARE)
        (self.root / "config").mkdir()
        (self.root / "config" / "host.json").write_text(json.dumps(
            {"gpu_lock": str(self.root / "gpu.lock"), "lock_dir": str(self.root), "nice": False}))
        self.saved = {k: os.environ.get(k) for k in (config.POINTER, "FAKE_EXECUTORS_LEDGER")}
        os.environ[config.POINTER] = str(self.root / "config")
        os.environ["FAKE_EXECUTORS_LEDGER"] = str(self.root / "ledger")
        config.reload()
        self.registry = Registry.load([self.root / "cells.json"])

    def tearDown(self):
        for k, v in self.saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        config.reload()
        self._tmp.cleanup()

    def env(self, label="fake", cwd=None):
        return Env(label, sys.executable, str(cwd or HERE),
                   {"PYTHONPATH": os.pathsep.join([str(HERE.parent), str(HERE)])})

    def go(self, roots, **kw):
        with Runtime(HERE) as runtime:
            return {r.label: r for r in build(roots, cache=Cache(self.root / "cache"), runtime=runtime,
                                              log=lambda _l: None, **kw)}

    def ledger(self):
        return (self.root / "ledger").read_text().splitlines()

    def test_a_declaration_file_composes_under_scopes_and_its_output_reaches_dependents(self):
        g = Graph()
        declare = load(self.root / "declare.py")["declare"]
        tip, master = declare(g.scoped("tip"), self.env(), 1), declare(g.scoped("master"), self.env(), 2)
        out = self.go([tip["all"], master["all"]])
        self.assertEqual({out[k].status for k in ("tip/first", "tip/second", "master/second")}, {"ran"})
        self.assertEqual(out["tip/all"].status, "ok")
        self.assertEqual(out["tip/second"].output["deps"]["first"]["params"], {"n": 1})
        self.assertEqual(out["tip/first"].output["inputs"], ["t8"])
        self.assertIn("a log line from tip/first", (Path(out["tip/first"].dir) / "log.txt").read_text())

    def test_no_label_or_path_enters_a_key_and_a_cached_target_is_skipped(self):
        first = load(self.root / "declare.py")["declare"](Graph().scoped("a"), self.env("one"), 7)
        ran = self.go([first["second"]])
        again = load(self.root / "declare.py")["declare"](Graph().scoped("b"), self.env("two"), 7)
        out = self.go([again["second"]])
        self.assertEqual((out["b/first"].key, out["b/second"].key), (ran["a/first"].key, ran["a/second"].key))
        self.assertEqual({r.status for r in out.values()}, {"cached"})
        self.assertEqual(len([line for line in self.ledger() if line.startswith("echo")]), 2)

    def test_a_setting_reaches_the_executor_and_stays_out_of_the_key(self):
        """WHERE a result is filed is not WHAT it is: two targets that differ only in a setting are one job, and the
        second is a cache hit at the first's key -- which is what lets the store file into any results checkout."""
        one = Graph().node("s", executor="fake_executors:echo", env=self.env(), params={"n": 1},
                           settings={"root": "/one/records"})
        first = self.go([one])
        two = Graph().node("s", executor="fake_executors:echo", env=self.env(), params={"n": 1},
                           settings={"root": "/two/records"})
        again = self.go([two])
        self.assertEqual(first["s"].output["settings"], {"root": "/one/records"})
        self.assertEqual((again["s"].key, again["s"].status), (first["s"].key, "cached"))

    def test_the_cache_sweeps_old_runs_and_evicts_the_least_recently_read_keys(self):
        """A cache that only grows is a disk that only shrinks. Runs go by age, keys by LAST READ -- and a key a build
        just hit is the one that survives, which is the whole difference between an LRU and a coin toss."""
        from rola_devtools.build.cache import Cache

        cache = Cache(self.root / "swept")
        for run in ("r1", "r2", "r3"):
            (cache.workspace(run, "t") / "raw.txt").write_text("x" * 1000)
        for key in ("old", "fresh"):
            work = cache.workspace("w", key)
            (work / "raw.txt").write_text("y" * 4000)
            cache.put(key, {"k": key}, {"out": key}, work)
        os.utime(cache.entry("old"), (1, 1))
        self.assertIsNotNone(cache.get("fresh"))  #: a hit is a use

        swept = cache.sweep(keep_runs=1, max_key_bytes=6000)
        self.assertEqual(sorted(d.name for d in (cache.root / "runs").glob("*")), ["w"])
        self.assertEqual([d.name for d in (cache.root / "keys").glob("*")], ["fresh"])
        self.assertEqual((swept["runs_dropped"], swept["keys_evicted"]), (3, 1))
        self.assertIsNone(cache.get("old"))

    def test_a_build_failure_stops_the_build_and_always_run_still_runs(self):
        g = Graph()
        bad = g.node("bad", executor="fake_executors:boom", env=self.env())
        after = g.node("after", executor="fake_executors:echo", env=self.env(), deps={"bad": bad})
        unrelated = g.node("unrelated", executor="fake_executors:echo", env=self.env(), params={"x": 1})
        stop = g.node("stop", executor="fake_executors:cleanup", env=self.env(), deps={"bad": bad, "u": unrelated},
                      cache=False, always_run=True)
        out = self.go([g.group("all", [bad, after, unrelated, stop])])
        self.assertEqual((out["bad"].status, out["after"].status, out["unrelated"].status, out["stop"].status),
                         ("failed", "skipped", "skipped", "ran"))
        self.assertIn("failed on purpose", out["bad"].detail)
        self.assertEqual(out["stop"].output["saw"], {"bad": "failed", "u": "skipped"})

    def test_a_domain_failure_is_the_targets_output_and_the_build_goes_on(self):
        g = Graph()
        measured = g.node("measured", executor="fake_executors:domain", env=self.env(), cache=False)
        after = g.node("after", executor="fake_executors:echo", env=self.env(), deps={"measured": measured})
        out = self.go([after])
        self.assertEqual((out["measured"].status, out["after"].status), ("ran", "ran"))
        self.assertFalse(out["measured"].output["entries"][1]["ok"])

    def test_an_uncached_target_runs_every_build_and_verify_reruns_a_missing_output(self):
        g = Graph()
        sample = g.node("sample", executor="fake_executors:echo", env=self.env(), cache=False)
        binary = g.node("binary", executor="fake_executors:echo", env=self.env(), params={"b": 1},
                        verify="fake_executors:present")
        self.go([sample, binary])
        (self.root / "present").write_text("")
        out = self.go([sample, binary])
        self.assertEqual((out["sample"].status, out["binary"].status), ("ran", "cached"))
        (self.root / "present").unlink()
        self.assertEqual(self.go([binary])["binary"].status, "ran")

    def test_a_held_resource_reaches_the_targets_worker_and_a_plan_runs_nothing(self):
        g = Graph()
        held = g.node("held", executor="fake_executors:echo", env=self.env(), holds={"gpu": "all"})
        after = g.node("after", executor="fake_executors:echo", env=self.env(), deps={"held": held})
        plan = self.go([after], dry=True)
        self.assertEqual((plan["held"].status, plan["after"].status), ("pending", "pending"))
        self.assertIsNone(plan["after"].key)
        self.assertEqual(self.ledger(), [])
        self.assertIsNotNone(self.go([after])["held"].output["held"])

    def test_what_a_graph_cannot_mean_is_refused(self):
        g = Graph()
        g.node("x", executor="fake_executors:echo")
        with self.assertRaises(ValueError):
            g.node("x", executor="fake_executors:echo")
        with self.assertRaises(ValueError):
            g.node("y", executor="fake_executors:cleanup", always_run=True)
        with self.assertRaises(ValueError):
            g.node("z", executor="fake_executors:echo", holds={"gpu": 0})
        with self.assertRaises(ValueError):
            Graph().node("w", executor="fake_executors:echo", deps={"x": g.targets["x"]})


class Identity(unittest.TestCase):
    def test_a_code_key_follows_imports_inside_the_root_and_moves_with_their_bytes(self):
        from rola_devtools.build import identity

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pkg").mkdir()
            (root / "pkg" / "__init__.py").write_text("")
            (root / "pkg" / "helper.py").write_text("X = 1\n")
            (root / "entry.py").write_text("import json\nfrom pkg.helper import X\n")
            (root / "data.json").write_text("{}")
            before = identity.code(root, "entry.py", data=("data.json",))
            self.assertEqual(before, identity.code(root, "entry.py", data=("data.json",)))
            (root / "pkg" / "helper.py").write_text("X = 2\n")
            after = identity.code(root, "entry.py", data=("data.json",))
            self.assertNotEqual(before, after)
            (root / "data.json").write_text('{"a": 1}')
            self.assertNotEqual(after, identity.code(root, "entry.py", data=("data.json",)))
            self.assertEqual(identity.files(root, ["pkg"]), identity.files(root, ["pkg"]))


if __name__ == "__main__":
    unittest.main()
