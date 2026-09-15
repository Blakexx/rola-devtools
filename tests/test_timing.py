"""The timing system on the declared build system: registrations run in their checkout's environment and build
nothing, a session sets its entries up behind a barrier and interleaves them with an untimed reset before every call,
an entry that cannot set up is recorded while the rest are timed, two stopwatches or a clock off the lock fail the
build, the server's workers are stopped however the build ends, memory is each entry alone, registrations in one
environment share its worker, and a null gate finds a worker's bias.
`python -m unittest tests.test_timing`"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from rola_devtools import config
from rola_devtools.build.cache import Cache
from rola_devtools.build.declare import Env, Graph
from rola_devtools.build.runtime import Runtime
from rola_devtools.build.scheduler import build
from rola_devtools.cells import Registry
from rola_devtools.timing.declare import (
    measure_memory,
    measure_null_gate,
    measure_timing,
    register_clock_reader,
    register_timing,
    start_timing_server,
    stop_timing_server,
)

HERE = Path(__file__).resolve().parent


class Timing(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "cells.json").write_text(json.dumps({"schema": 1, "data": "fake_provider:tokens",
                                                           "cells": [{"name": "t8", "tokens": 8},
                                                                     {"name": "t16", "tokens": 16}]}))
        (self.root / "config").mkdir()
        self.host = {"gpu_lock": str(self.root / "gpu.lock"), "lock_dir": str(self.root), "nice": False}
        (self.root / "config" / "host.json").write_text(json.dumps(self.host))
        self.saved = {k: os.environ.get(k) for k in (config.POINTER, "FAKE_CLOCK_GHZ", "FAKE_BIAS_DIR")}
        os.environ[config.POINTER] = str(self.root / "config")
        config.reload()
        self.registry = Registry.load([self.root / "cells.json"])

    def tearDown(self):
        for k, v in self.saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        config.reload()
        self._tmp.cleanup()

    def env(self, label):
        return Env(label, sys.executable, str(HERE), {"PYTHONPATH": os.pathsep.join([str(HERE.parent), str(HERE)])})

    def go(self, roots):
        with Runtime(HERE) as runtime:
            return {r.label: r for r in build(roots, cache=Cache(self.root / "cache"), runtime=runtime,
                                              registry=self.registry, log=lambda _l: None)}

    def session(self, result):
        return json.loads((Path(result.dir) / "session.json").read_text())

    def graph(self, registrations, **measure):
        g = Graph()
        server = start_timing_server(g)
        entries = [register_timing(g, name, server=server, env=self.env(env), executor=f"fake_entries:{ex}", cells=cells,
                                   params=params) for name, env, ex, cells, params in registrations]
        clock = register_clock_reader(g, "clock", server=server, env=self.env("tip"), executor="fake_entries:clock")
        session = measure_timing(g, "session", server=server, entries=entries, clock=clock, rounds=2, reps=3, **measure)
        stop = stop_timing_server(g, server=server, after=[session])
        return g, session, stop

    def test_a_session_interleaves_every_entry_and_records_what_could_not_run(self):
        _g, session, stop = self.graph([("tip/fast", "tip", "fixed", ["t8", "t16"], {"ms": 0.5}),
                                        ("master/fast", "master", "fixed", ["t8"], {"ms": 0.5}),
                                        ("tip/unbuilt", "tip", "unbuilt", ["t16"], {})])
        out = self.go([stop])
        self.assertEqual((out["session"].status, out["timing-server-stop"].status), ("ran", "ran"))
        doc = self.session(out["session"])
        statuses = {(m["owner"], m["cell"]): m["status"] for m in doc["members"]}
        self.assertEqual(statuses, {("tip/fast", "t8"): "ok", ("tip/fast", "t16"): "ok", ("master/fast", "t8"): "ok",
                                    ("tip/unbuilt", "t16"): "failed"})
        self.assertEqual(len(doc["samples"]), 3 * 2 * 3)
        rep0 = [s for s in doc["samples"] if (s["round"], s["rep"]) == (0, 0)]
        self.assertEqual(sorted(s["position"] for s in rep0), [0, 1, 2])
        self.assertEqual({s["ms"] for s in doc["samples"] if s["member"] == "m1"}, {8.0})
        built = next(m for m in doc["members"] if m["id"] == "m0")["built"]
        self.assertIsNotNone(built["held"])

    def test_a_reset_gives_every_call_the_first_calls_state(self):
        _g, session, stop = self.graph([("tip/reset", "tip", "stateful", ["t8"], {"reset": True}),
                                        ("tip/drift", "tip", "stateful", ["t16"], {})])
        doc = self.session(self.go([stop])["session"])
        by = {m["cell"]: m["id"] for m in doc["members"]}
        self.assertEqual({s["ms"] for s in doc["samples"] if s["member"] == by["t8"]}, {1.0})
        self.assertGreater(max(s["ms"] for s in doc["samples"] if s["member"] == by["t16"]), 1.0)

    def test_two_stopwatches_fail_the_build_and_the_server_still_stops(self):
        _g, session, stop = self.graph([("tip/fast", "tip", "fixed", ["t8"], {"ms": 1.0}),
                                        ("tip/other", "tip", "other_clock", ["t16"], {})])
        out = self.go([stop])
        self.assertEqual((out["session"].status, out["timing-server-stop"].status), ("failed", "ran"))
        self.assertIn("one stopwatch", out["session"].detail)
        self.assertTrue(out["timing-server-stop"].output["stopped"])

    def test_a_clock_off_the_lock_fails_the_session(self):
        (self.root / "config" / "clock.json").write_text(json.dumps({"ghz": 1.665, "lock": ["true"], "unlock": ["true"]}))
        config.reload()
        os.environ["FAKE_CLOCK_GHZ"] = "1.2"
        _g, session, stop = self.graph([("tip/fast", "tip", "fixed", ["t8"], {"ms": 1.0})])
        out = self.go([stop])
        self.assertEqual(out["session"].status, "failed")
        self.assertIn("CLOCK", out["session"].detail)

    def test_memory_is_each_entry_alone_and_a_failed_entry_is_recorded(self):
        g = Graph()
        server = start_timing_server(g)
        reg = register_timing(g, "tip/fast", server=server, env=self.env("tip"), executor="fake_entries:fixed",
                              cells=["t8"], params={"ms": 1.0})
        bad = register_timing(g, "tip/unbuilt", server=server, env=self.env("tip"), executor="fake_entries:unbuilt",
                              cells=["t8"])
        memory = measure_memory(g, "memory", server=server, entries=[reg, bad])
        stop = stop_timing_server(g, server=server, after=[memory])
        out = self.go([stop])
        rows = json.loads((Path(out["memory"].dir) / "memory.json").read_text())["rows"]
        self.assertEqual([r["status"] for r in rows], ["ok", "failed"])
        self.assertEqual(rows[0]["calls"], 5)

    def test_registrations_in_one_environment_share_its_worker(self):
        _g, session, stop = self.graph([("tip/fast", "tip", "fixed", ["t8"], {"ms": 1.0}),
                                        ("tip/slow", "tip", "fixed", ["t16"], {"ms": 2.0})])
        doc = self.session(self.go([stop])["session"])
        self.assertEqual(len({m["built"]["pid"] for m in doc["members"]}), 1)

    def test_a_null_gate_trusts_an_unbiased_entry_and_finds_a_biased_one(self):
        os.environ["FAKE_BIAS_DIR"] = str(self.root)
        g = Graph()
        server = start_timing_server(g)
        clock = register_clock_reader(g, "clock", server=server, env=self.env("tip"), executor="fake_entries:clock")
        gates = [measure_null_gate(g, f"null/{name}", server=server, clock=clock, rounds=2, reps=3,
                                   entry=register_timing(g, name, server=server, env=self.env("tip"),
                                                         executor=f"fake_entries:{ex}", cells=["t8"], params=params))
                 for name, ex, params in (("fast", "fixed", {"ms": 1.0}), ("biased", "biased", {}))]
        out = self.go([stop_timing_server(g, server=server, after=gates)])
        fair, biased = out["null/fast"].output["cells"]["t8"], out["null/biased"].output["cells"]["t8"]
        self.assertEqual((fair["trusted"], fair["ratio_median"]), (True, 1.0))
        self.assertEqual((biased["trusted"], biased["ratio_median"]), (False, 0.5))
        doc = json.loads((Path(out["null/biased"].dir) / "session.json").read_text())
        self.assertEqual(len({m["built"]["pid"] for m in doc["members"]}), 2)

    def test_a_stored_session_appends_a_run_stamped_sample_to_its_configurations_record(self):
        from rola_devtools.store import store

        g, session, _stop = self.graph([("tip/fast", "tip", "fixed", ["t8"], {"ms": 1.0})])
        kept = store(g, "store", source=session, location="timing/session", root=str(self.root / "records"))
        stop = stop_timing_server(g, "stop-after-store", server=g.targets["timing-server"], after=[kept])
        first, second = self.go([stop]), self.go([stop])
        from rola_results import Store

        (record,) = Store("timing/session", self.root / "records").records()
        self.assertEqual([s["run"] for s in record["samples"]], [first["store"].output["run"], second["store"].output["run"]])
        self.assertIn("tip/fast", record["samples"][0]["provenance"]["checkouts"])


if __name__ == "__main__":
    unittest.main()
