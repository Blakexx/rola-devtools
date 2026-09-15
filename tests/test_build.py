"""The build system alone: nodes in dependency order, keys from content and never from ids, stored results skipped, a
local result re-run when it is no longer present, refusals and failures stored, dependents of either blocked.
`python -m unittest tests.test_build`"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fake_store import FakeStore

from rola_devtools.build import Executed, Node, order, run


class Build(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ran: list[str] = []
        self.results: dict[str, Executed] = {}

    def tearDown(self):
        self.tmp.cleanup()

    def store(self, location):
        return FakeStore(self.root, location)

    def execute(self, node, deps):
        self.ran.append(node.id)
        if node.id in self.results:
            return self.results[node.id]
        return Executed(output={"n": node.semantics["n"], "read": {r: json.loads(p.read_text()) for r, p in deps.items()}})

    def go(self, nodes, **kw):
        return {o.id: o for o in run(nodes, self.store, self.execute, log=lambda _line: None, **kw)}

    @staticmethod
    def chain(prefix="", n=1):
        return [Node(f"{prefix}b", {"n": n + 1}, "loc", (("a", f"{prefix}a"),)), Node(f"{prefix}a", {"n": n}, "loc")]

    def test_dependencies_run_first_and_a_cycle_or_an_unknown_dependency_is_refused(self):
        self.assertEqual([n.id for n in order(self.chain())], ["a", "b"])
        with self.assertRaises(ValueError):
            order([Node("x", {}, "l", (("y", "y"),)), Node("y", {}, "l", (("x", "x"),))])
        with self.assertRaises(KeyError):
            order([Node("x", {}, "l", (("y", "nowhere"),))])
        with self.assertRaises(ValueError):
            order([Node("x", {}, "l"), Node("x", {}, "l")])

    def test_a_run_stores_and_a_second_skips_and_an_id_never_enters_a_key(self):
        first = self.go(self.chain())
        self.assertEqual((first["a"].status, first["b"].status, self.ran), ("ran", "ran", ["a", "b"]))
        again = self.go(self.chain(prefix="renamed-"))
        self.assertEqual({o.status for o in again.values()}, {"complete"})
        self.assertEqual(again["renamed-b"].key, first["b"].key)
        self.assertEqual(self.ran, ["a", "b"])

    def test_a_dependency_whose_output_changes_rekeys_its_dependent(self):
        first = self.go(self.chain())
        self.results["a"] = Executed(output={"different": True})
        second = self.go(self.chain(), force=True)
        self.assertEqual(second["a"].key, first["a"].key)
        self.assertNotEqual(second["b"].key, first["b"].key)

    def test_a_refusal_and_a_failure_are_stored_and_block_their_dependents(self):
        self.results = {"a": Executed(refused="not this one"), "c": Executed(error="broke")}
        nodes = self.chain() + [Node("c", {"n": 3}, "loc"), Node("d", {"n": 4}, "loc", (("c", "c"),))]
        out = self.go(nodes)
        self.assertEqual([out[i].status for i in "abcd"], ["refused", "blocked", "failed", "blocked"])
        self.ran.clear()
        out = self.go(nodes)
        self.assertEqual((out["a"].status, out["c"].status, self.ran), ("refused", "failed", ["c"]))

    def test_repeat_adds_to_a_repeatable_node_and_a_plan_runs_nothing(self):
        nodes = [Node("r", {"n": 1}, "loc", repeatable=True), Node("s", {"n": 2}, "loc")]
        self.assertEqual({o.status for o in self.go(nodes, dry=True).values()}, {"pending"})
        self.assertEqual(self.ran, [])
        self.go(nodes)
        self.go(nodes, repeat=True)
        self.assertEqual(self.ran, ["r", "s", "r"])

    def test_a_local_result_counts_only_while_present(self):
        here = {"present": True}
        nodes = [Node("binary", {"n": 1}, "loc", local=True)]
        self.go(nodes)
        self.assertEqual(run(nodes, self.store, self.execute, present=lambda node, path: here["present"],
                             log=lambda _l: None)[0].status, "complete")
        here["present"] = False
        self.assertEqual(run(nodes, self.store, self.execute, present=lambda node, path: here["present"],
                             log=lambda _l: None)[0].status, "ran")

    def test_an_execution_that_is_not_exactly_one_result_fails(self):
        self.results["a"] = Executed()
        self.assertEqual(self.go([Node("a", {"n": 1}, "loc")])["a"].status, "failed")


if __name__ == "__main__":
    unittest.main()
