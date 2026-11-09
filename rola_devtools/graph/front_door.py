"""THE GRAPH'S FRONT DOOR: bazelisk runs a repository's `bazel/bazel` -- the wrapper its `.bazeliskrc` names -- in place
of the Bazel it downloaded (`BAZEL_REAL`), and that stub runs `main` here, so every Bazel command in a checkout goes
through one implementation, typed or run by a tool.

It hands Bazel the machine's settings from the dev config (the shared disk and repository caches under
`host.build_cache`, as an rc file read after the workspace's), and, where the workspace depends on the `rola_devtools`
module, the machine's rola-devtools checkout as that module's override (a second rc file, the workspace's own), so a
change made there is the one the graph runs and nothing names a machine path. It runs every BUILD under the host
budget (`rola_devtools.locks.host`): a command's build phase holds the slots free when it starts, at least one, and
runs one compile job a `host.nvcc_threads` of them; its test or run phase holds none, so a test or a measurement takes
its own locks as it does outside the graph -- a GPU test under a held budget is the deadlock the host's lock order
exists to prevent.
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
from pathlib import Path

from rola_devtools import config
from rola_devtools.locks import host

#: the commands whose whole run is a build, and those whose build comes before a phase that holds no budget
BUILDS = ("build",)
TESTS = ("test",)
#: the test command's options the build command refuses: they shape only the test summary
TEST_SUMMARY_ONLY = ("--test_verbose_timeout_warnings", "--notest_verbose_timeout_warnings",
                     "--print_relative_test_log_paths", "--noprint_relative_test_log_paths")
#: the module this checkout is, overriding a workspace's pin of it
MODULE = "rola_devtools"
CHECKOUT = Path(__file__).resolve().parents[2]


def _split(argv: list[str]) -> tuple[list[str], str | None, list[str]]:
    """(startup options, the command, its arguments): the command is the first word that is not an option, so a
    startup option is written `--name=value`."""
    for i, word in enumerate(argv):
        if not word.startswith("-"):
            return argv[:i], word, argv[i + 1:]
    return argv, None, []


def _cache() -> Path:
    return Path(config.machine("host.build_cache")) / "bazel"


def _write(path: Path, text: str) -> Path:
    """`text` at `path`, replaced through a rename only when it differs, so a concurrent reader sees a whole file."""
    if not path.is_file() or path.read_text() != text:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        with os.fdopen(fd, "w") as out:
            out.write(text)
        os.replace(tmp, path)
    return path


def machine_rc() -> Path:
    """The machine's settings as an rc file, regenerated from the dev config on every run."""
    cache = _cache()
    return _write(cache / "machine.bazelrc", (
        "# written by rola_devtools.graph.front_door from the dev config's host.build_cache; regenerated every run\n"
        f"common --disk_cache={cache / 'disk'}\n"
        f"common --repository_cache={cache / 'repository'}\n"
        f"common --repo_env=ROLA_GRAPH_STORE={cache / 'store'}\n"
        f"common --repo_env=ROLA_NVCC_THREADS={config.machine('host.nvcc_threads')}\n"
        #: a check of the commit reads the machine's dev config; a check over files unsets it
        f"test --test_env=ROLA_DEV_CONFIG={config.directory()}\n"))


def depends_on_module(workspace: Path) -> bool:
    """Whether the workspace's module file depends on this module (`bazel_dep(name = "rola_devtools", ...)`)."""
    text = (workspace / "MODULE.bazel").read_text() if (workspace / "MODULE.bazel").is_file() else ""
    return re.search(rf'bazel_dep\(\s*name\s*=\s*"{MODULE}"', text) is not None


def workspace_rc(workspace: Path) -> Path | None:
    """The workspace's own settings: this checkout as the override of its `rola_devtools` pin, where it has one."""
    if not depends_on_module(workspace):
        return None
    key = hashlib.sha256(str(workspace.resolve()).encode()).hexdigest()[:16]
    return _write(_cache() / "workspaces" / f"{key}.bazelrc", (
        f"# written by rola_devtools.graph.front_door for {workspace.name}: the machine's rola-devtools checkout\n"
        f"common --override_module={MODULE}={CHECKOUT}\n"))


def _wait(argv: list[str]) -> int:
    """Run Bazel's client to its end: an interrupt reaches it too (one process group), and it stops its server's work."""
    proc = subprocess.Popen(argv)
    while True:
        try:
            return proc.wait()
        except KeyboardInterrupt:
            continue


def under_budget(head: list[str], command: str, rest: list[str]) -> int:
    """The command under the host budget: the slots free now (at least one), one compile job a `host.nvcc_threads` of
    them -- nvcc runs that many cicc at once, one an architecture."""
    with host.acquire(host.budget_slots(), label=f"bazel {command}") as held:
        jobs = max(1, held // config.machine("host.nvcc_threads"))
        return _wait([*head, command, f"--jobs={jobs}", *rest])


def main(argv: list[str], workspace: Path) -> int:
    """The front door for the workspace at `workspace` (the stub's repository)."""
    real = os.environ.get("BAZEL_REAL")
    if not real:
        raise SystemExit("bazel/bazel is the graph's front door, which bazelisk runs in Bazel's place: run `bazel`, the "
                         "pinned bazelisk")
    startup, command, rest = _split(argv)
    if command == "coverage":
        raise SystemExit("bazel coverage is refused: its instrumented build differs from the budgeted one and would run "
                         "outside the host budget, and the graph has no coverage target")
    own = workspace_rc(workspace)
    head = [real, *startup, f"--bazelrc={machine_rc()}", *([f"--bazelrc={own}"] if own else [])]
    if command in BUILDS:
        return under_budget(head, command, rest)
    if command in TESTS:
        built = under_budget(head, "build", [w for w in rest if w.split("=")[0] not in TEST_SUMMARY_ONLY])
        if built:
            return built
    elif command == "run" and not any(w.startswith("--script_path") for w in rest):
        #: the run's build under the budget writes the launch script, which then runs holding nothing
        scratch = Path(config.machine("host.scratch"))
        scratch.mkdir(parents=True, exist_ok=True)
        fd, script = tempfile.mkstemp(prefix="bazel_run_", suffix=".sh", dir=scratch)
        os.close(fd)
        built = under_budget(head, command, [f"--script_path={script}", *rest])
        text = Path(script).read_text() if not built else ""
        os.unlink(script)
        if built:
            return built
        os.execv("/bin/bash", ["/bin/bash", "-c", text])
    elif command == "run":
        return under_budget(head, command, rest)
    os.execv(real, [*head, *([command] if command else []), *rest])
