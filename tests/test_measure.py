"""The measurement service: an owner's registry described in its own environment, its units composed with cells into
nodes (a node only where a unit accepts a cell, an arm's memory node, sessions of timed arms), one worker an instance
for the run, stored results skipped, a build re-run when its output is no longer on the machine, refusals and failures
stored, sessions interleaved and paired to their reference instance. `python -m unittest tests.test_measure`"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from fake_store import FakeStore

from rola_devtools.cells import Registry
from rola_devtools.locks.gpu import gpu_lock
from rola_devtools.measure.service import Instance, Service, Session

HERE = Path(__file__).resolve().parent
CELLS = {"schema": 1, "data": "fake_provider:tokens",
         "cells": [{"name": "t8", "tokens": 8}, {"name": "t16", "tokens": 16}, {"name": "big", "tokens": 1000}]}


class Measure(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "ledger").write_text("")
        (self.root / "cells.json").write_text(json.dumps(CELLS))
        self.cells = Registry.load([self.root / "cells.json"])
        self.env = {"PYTHONPATH": os.pathsep.join([str(HERE.parent), str(HERE)]),
                    "FAKE_UNITS_LEDGER": str(self.root / "ledger"), "FAKE_UNITS_BINARY": str(self.root / "binary")}
        self.services: list[Service] = []

    def tearDown(self):
        for service in self.services:
            service.close()
        self.tmp.cleanup()

    def instance(self, label="fake", role="subject", **env):
        return Instance(label, sys.executable, str(HERE), "fake_units:registry", {**self.env, **env}, role)

    def service(self, *instances):
        service = Service(list(instances) or [self.instance()], registry=self.cells, hold=contextlib.nullcontext,
                          provenance=lambda instance: {"label": instance.label}, log=lambda _line: None)
        self.services.append(service)
        return service

    def store(self, location):
        return FakeStore(self.root / "store", location)

    def go(self, service, cells=("t8",), **kw):
        run_kw = {k: kw.pop(k) for k in ("repeat", "force", "dry") if k in kw}
        nodes = service.nodes(cells, **{"memory": False, **kw})
        return {o.id: o for o in service.run(nodes, self.store, **run_kw)}

    def ledger(self):
        return (self.root / "ledger").read_text().splitlines()

    def test_a_unit_says_which_cells_it_accepts_and_a_refused_cell_makes_no_node(self):
        service = self.service()
        units = service.units(service.instances[0], ["t8", "big"])
        self.assertEqual(units["a"]["on"]["big"], {"refused": "big: more than 256 tokens"})
        self.assertEqual(units["a"]["on"]["t8"], {"identity": {"n": 1, "version": "1"}})
        ids = {n.id for n in service.nodes(["t8", "big"], memory=False)}
        self.assertIn("fake:a@t8", ids)
        self.assertNotIn("fake:a@big", ids)
        self.assertIn("fake:build", ids)

    def test_a_run_stores_what_it_ran_and_a_second_run_skips_it(self):
        first = self.go(self.service())
        self.assertEqual((first["fake:build"].status, first["fake:a@t8"].status, first["fake:b@t8"].status),
                         ("ran", "ran", "ran"))
        st = self.store("fake/count")
        output = json.loads(st.output(st.get(first["fake:b@t8"].key)).read_text())
        self.assertEqual(output["value"], 16)
        self.assertEqual(set(output["deps"]), {"a", "build"})
        self.assertEqual(self.go(self.service())["fake:b@t8"].status, "complete")
        self.assertEqual(self.ledger().count("build"), 1)

    def test_a_label_never_enters_a_key_and_an_identity_change_rekeys_downstream(self):
        first = self.go(self.service(self.instance("one")))
        same = self.go(self.service(self.instance("two")))
        self.assertEqual(first["one:b@t8"].key, same["two:b@t8"].key)
        self.assertEqual(same["two:b@t8"].status, "complete")
        moved = self.go(self.service(self.instance("three", FAKE_UNITS_VERSION="2")))
        self.assertNotEqual(first["one:a@t8"].key, moved["three:a@t8"].key)
        self.assertNotEqual(first["one:b@t8"].key, moved["three:b@t8"].key)
        self.assertEqual(moved["three:build"].status, "complete")

    def test_a_refusal_is_stored_and_its_dependent_blocked_and_a_failure_too(self):
        out = self.go(self.service())
        self.assertEqual((out["fake:no@t8"].status, out["fake:after-no@t8"].status), ("refused", "blocked"))
        self.assertEqual((out["fake:boom@t8"].status, out["fake:after-boom@t8"].status), ("failed", "blocked"))
        self.assertIn("failed on purpose", out["fake:boom@t8"].detail)
        again = self.go(self.service())
        self.assertEqual((again["fake:no@t8"].status, again["fake:boom@t8"].status), ("refused", "failed"))

    def test_a_build_runs_again_when_its_output_left_the_machine(self):
        self.go(self.service())
        (self.root / "binary").unlink()
        out = self.go(self.service())
        self.assertEqual((out["fake:build"].status, out["fake:a@t8"].status), ("ran", "complete"))
        self.assertEqual(self.ledger().count("build"), 2)

    def test_select_takes_dependencies_along_and_a_plan_runs_nothing(self):
        service = self.service()
        out = self.go(service, select=lambda instance, unit: unit["name"] == "b", dry=True)
        self.assertEqual(set(out), {"fake:build", "fake:a@t8", "fake:b@t8"})
        self.assertEqual({o.status for o in out.values()}, {"pending"})
        self.assertEqual(self.ledger(), [])

    def test_a_session_interleaves_its_members_on_every_accepted_cell_and_records_a_refusal(self):
        session = Session("fixed", "fake/sessions", ("fast", "slow", "unfit"), ("t8", "t16"), rounds=2, reps=3,
                          relation={"holds": "two fake cells"})
        service = self.service()
        out = self.go(service, select=lambda instance, unit: False, sessions=[session])
        self.assertEqual(out["session fixed"].status, "ran")
        st = self.store("fake/sessions")
        doc = json.loads(st.output(st.get(out["session fixed"].key)).read_text())
        self.assertEqual([m["member"] for m in doc["members"]], ["fake:fast@t8", "fake:fast@t16", "fake:slow@t8",
                                                                 "fake:slow@t16"])
        self.assertEqual(set(doc["refusals"]), {"fake:unfit@t8", "fake:unfit@t16"})
        self.assertEqual(len(doc["order"]), 4 * 2 * 3)
        slow = doc["members"][2]
        self.assertEqual((slow["median_ms"], slow["paired"], slow["post"], slow["built"]),
                         (3.0, [], {"calls": 6}, {"ms": 3.0, "tokens": 8}))
        self.assertEqual(doc["relation"], {"holds": "two fake cells"})
        self.assertEqual(self.go(self.service(), select=lambda i, u: False, sessions=[session])["session fixed"].status,
                         "complete")

    def test_a_member_pairs_with_the_reference_instances_arm_on_its_cell(self):
        session = Session("checkouts", "fake/sessions", ("fast", "slow"), ("t8",), reference="base", rounds=2, reps=3)
        service = self.service(self.instance("tip"), self.instance("base", role="reference"))
        out = self.go(service, select=lambda i, u: False, sessions=[session])
        st = self.store("fake/sessions")
        members = {m["member"]: m for m in json.loads(st.output(st.get(out["session checkouts"].key)).read_text())["members"]}
        self.assertEqual([p["reference"] for p in members["tip:fast@t8"]["paired"]], ["base:fast@t8"])
        self.assertEqual(members["tip:slow@t8"]["paired"][0]["ratio_median"], 1.0)
        self.assertEqual((members["base:fast@t8"]["paired"], members["tip:fast@t8"]["role"]), ([], "subject"))

    def test_a_session_every_member_refuses_is_refused(self):
        session = Session("unfit", "fake/sessions", ("unfit",), ("t8",), rounds=1, reps=1)
        out = self.go(self.service(), select=lambda i, u: False, sessions=[session])
        self.assertEqual(out["session unfit"].status, "refused")

    def test_builds_run_before_the_units_are_described_on_cells(self):
        service = self.service()
        built = {o.id: o.status for o in service.build(self.store)}
        self.assertEqual(built, {"fake:build": "ran"})
        self.assertEqual(self.ledger(), ["build"])
        out = self.go(service, select=lambda i, u: u["name"] == "a")
        self.assertEqual((out["fake:build"].status, out["fake:a@t8"].status), ("complete", "ran"))

    def test_one_worker_serves_an_instance_for_the_whole_run(self):
        service = self.service()
        self.go(service, select=lambda i, u: u["name"] in ("a", "no"))
        pid = service.worker(service.instances[0]).process.pid
        self.go(service, cells=("t16",), select=lambda i, u: u["name"] == "a")
        self.assertEqual(service.worker(service.instances[0]).process.pid, pid)

    def test_a_memory_node_is_the_arm_alone(self):
        service = self.service()
        with gpu_lock(mode="shared"):
            out = self.go(service, cells=("t16",), select=lambda i, u: u["name"] == "holding", memory=True)
        memory = out["memory fake:holding@t16"]
        self.assertEqual(memory.status, "ran")
        st = self.store("fake/memory")
        doc = json.loads(st.output(st.get(memory.key)).read_text())
        self.assertGreaterEqual(doc["peak_allocated_bytes"] - doc["allocated_before_build_bytes"], 16 * 1024 + 16 * 4096)
        self.assertEqual((doc["outside_allocator_bytes"], doc["calls"], doc["built"]), (7, 5, {"held": 16 * 1024}))


if __name__ == "__main__":
    unittest.main()
