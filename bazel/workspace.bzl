"""rola-devtools' own graph (when it is the root workspace): the module's generic rules handed this workspace's
environment and tree. Never loaded by a workspace depending on the module, which wraps the rules with its own."""

load("@devtools_env//:flags.bzl", "VENV")
load("@devtools_tree//:tree.bzl", "TRACKED")
load(":rules.bzl", _tree_part = "tree_part", _tree_test = "tree_test")

def tree_test(**kwargs):
    """A check over the tree's tracked files, the environment's tools on PATH."""
    _tree_test(venv = VENV, **kwargs)

def tree_part(parts = {}):
    """This package's part of the tree: `tracked` and `tracked_python`."""
    _tree_part(TRACKED, parts)
