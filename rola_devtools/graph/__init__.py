"""THE BUILD GRAPH'S SHARED MACHINERY, every RoLA repository's: the front door each repository's `bazel/bazel` stub runs
(`front_door`), the environment's store builder the `rola_devtools` Bazel module's environment extension runs
(`make_env`), and the test tiers' sharding plugin (`pytest_shards`). Standard library only: the front door runs under
the machine's `python3` before any environment exists. Docs: README.md, the build graph.
"""
