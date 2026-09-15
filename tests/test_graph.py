"""The build system: a graph described in its own environment, nodes keyed by what they depend on, stored results
skipped, refusals and failures recorded, dependents of either blocked, timed members interleaved in one session."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from rola_devtools.graph import Env
from rola_devtools.graph.engine import Session, key, load, run

HERE = Path(__file__).resolve().parent


class FakeStore:
    """The store's interface over a directory: `get`, `put`, `output`, with the store's key."""

    def __init__(self, root: Path, location: str) -> None:
        self.dir = root / location

    def get(self, k):
        path = self.dir / f"{k}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def put(self, semantics, *, output=None, error=None, wall_s=None, provenance=None):
        k = key(semantics)
        self.dir.mkdir(parents=True, exist_ok=True)
        record = self.get(k) or {"key": k, "semantics": semantics, "samples": []}
        n = len(record["samples"])
        sample = {"n": n, "ok": error is None, "provenance": provenance}
        if error is None:
            sample["output"] = f"{k}.{n}.out.json"
            (self.dir / sample["output"]).write_text(json.dumps(output))
        else:
            sample["error"] = error
        record["samples"].append(sample)
        (self.dir / f"{k}.json").write_text(json.dumps(record))
        return sample

    def output(self, record):
        return self.dir / [s for s in record["samples"] if s["ok"]][-1]["output"]


class Graph(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ledger = self.root / "ledger"
        self.ledger.write_text("")
        self.env = Env("fake", sys.executable, str(HERE), "fake_graph:graph",
                       {"PYTHONPATH": os.pathsep.join([str(HERE.parent), str(HERE)]),
                        "FAKE_GRAPH_LEDGER": str(self.ledger)})
        self.instance = load(self.env)

    def tearDown(self):
        self.tmp.cleanup()

    def store(self, location):
        return FakeStore(self.root / "store", location)

    def go(self, **kw):
        return {o.name: o for o in run([self.instance], self.store, log=lambda _line: None, **kw)}

    def executed(self):
        return self.ledger.read_text().split()

    def test_the_graph_describes_itself_in_its_own_environment(self):
        nodes = {n["name"]: n for n in self.instance.nodes}
        self.assertEqual(nodes["b"]["deps"], ["a"])
        self.assertEqual((nodes["a"]["location"], nodes["fast"]["timed"]), ("fake/count", True))
        self.assertEqual(nodes["a"]["identity"], {"n": 1, "version": "1"})

    def test_a_run_stores_what_it_ran_and_a_second_run_skips_it(self):
        first = self.go()
        self.assertEqual({first[q].status for q in ("fake:a", "fake:b")}, {"ran"})
        self.assertEqual(sorted(self.executed()), ["1", "2"])
        record = self.store("fake/count").get(first["fake:b"].key)
        output = json.loads(self.store("fake/count").output(record).read_text())
        self.assertEqual(output, {"value": 20, "deps": {"a": {"value": 10, "deps": {}}}})
        second = self.go()
        self.assertEqual({second[q].status for q in ("fake:a", "fake:b")}, {"complete"})
        self.assertEqual(sorted(self.executed()), ["1", "2"])

    def test_a_refusal_is_stored_and_its_dependent_is_blocked(self):
        out = self.go()
        self.assertEqual((out["fake:no"].status, out["fake:after-no"].status), ("refused", "blocked"))
        again = self.go()
        self.assertEqual(again["fake:no"].status, "refused")
        self.assertNotIn("3", self.executed())

    def test_a_failure_is_recorded_retried_and_blocks_its_dependent(self):
        out = self.go()
        self.assertEqual((out["fake:boom"].status, out["fake:after-boom"].status), ("failed", "blocked"))
        self.assertIn("failed on purpose", out["fake:boom"].error)
        self.assertEqual(self.go()["fake:boom"].status, "failed")
        record = self.store("fake/count").get(out["fake:boom"].key)
        self.assertEqual([s["ok"] for s in record["samples"]], [False, False])

    def test_an_identity_change_rekeys_the_node_and_everything_downstream(self):
        first = self.go(select={"fake:b"})
        self.env = Env(self.env.label, self.env.python, self.env.cwd, self.env.graph,
                       {**self.env.env, "FAKE_GRAPH_VERSION": "2"})
        self.instance = load(self.env)
        second = self.go(select={"fake:b"})
        self.assertNotEqual(first["fake:a"].key, second["fake:a"].key)
        self.assertNotEqual(first["fake:b"].key, second["fake:b"].key)
        self.assertEqual(second["fake:b"].status, "ran")

    def test_select_takes_the_dependencies_and_nothing_else(self):
        out = self.go(select={"fake:b"})
        self.assertEqual(set(out), {"fake:a", "fake:b"})

    def test_a_plan_runs_nothing(self):
        out = self.go(dry=True)
        self.assertEqual(out["fake:a"].status, "pending")
        self.assertEqual(self.executed(), [])

    def test_a_session_interleaves_its_members_and_records_the_refusal(self):
        session = Session("fast-vs-slow", "fake/sessions", ("fake:fast", "fake:slow", "fake:unbuilt"), rounds=2, reps=3,
                          relation={"holds": "one fake cell"})
        out = self.go(sessions=[session], select={"fake:fast"})
        self.assertEqual(out["fast-vs-slow"].status, "ran")
        st = self.store("fake/sessions")
        doc = json.loads(st.output(st.get(out["fast-vs-slow"].key)).read_text())
        self.assertEqual([m["member"] for m in doc["members"]], ["fake:fast", "fake:slow"])
        self.assertEqual(doc["refused"], {"fake:unbuilt": "this arm is not built"})
        self.assertEqual(len(doc["order"]), 2 * 2 * 3)
        slow = doc["members"][1]
        self.assertEqual((slow["median_ms"], slow["paired"], slow["post"]), (3.0, [], {"samples": 6}))
        self.assertEqual(doc["relation"], {"holds": "one fake cell"})
        self.assertEqual(self.go(sessions=[session], select={"fake:fast"})["fast-vs-slow"].status, "complete")

    def test_a_member_pairs_with_the_reference_instances_node_of_its_own_name(self):
        other = load(Env("other", self.env.python, self.env.cwd, self.env.graph, self.env.env))
        session = Session("two-checkouts", "fake/sessions", ("fake:fast", "other:fast", "other:slow"), reference="fake",
                          rounds=2, reps=3)
        out = {o.name: o for o in run([self.instance, other], self.store, [session], select={"fake:fast"},
                                      log=lambda _line: None)}
        st = self.store("fake/sessions")
        members = {m["member"]: m for m in json.loads(st.output(st.get(out["two-checkouts"].key)).read_text())["members"]}
        self.assertEqual([p["reference"] for p in members["other:fast"]["paired"]], ["fake:fast"])
        self.assertEqual((members["fake:fast"]["paired"], members["other:slow"]["paired"]), ([], []))

    def test_a_session_every_member_refuses_is_refused(self):
        session = Session("nothing-built", "fake/sessions", ("fake:unbuilt",), rounds=1, reps=1)
        self.assertEqual(self.go(sessions=[session], select={"fake:unbuilt"})["nothing-built"].status, "refused")


class Identity(unittest.TestCase):
    def test_a_code_key_follows_imports_inside_the_root_and_moves_with_their_bytes(self):
        from rola_devtools.graph import identity

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
