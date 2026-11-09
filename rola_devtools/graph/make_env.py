"""THE ENVIRONMENT'S STORE: a pinned Python and a lock's packages, built once a key into the store every checkout's graph
on the machine shares, under a file lock -- a second builder of the same key waits and reuses it.

The lock's packages are installed hash-checked (`--require-hashes`); then each `--package`, without its dependencies
(the lock carries them): a local directory, keyed by the content of its files, or a requirement pinned by commit
(`name @ git+URL@COMMIT`, which has no hash to require), keyed by its text. A package builds apart from the
environment unless `--no-build-isolation` builds it against it (a build that imports a locked package; the lock then
carries the build backend). The key hashes the lock, every package, the Python version, uv's version, the extra index
and the isolation, so a change to any is a new environment and an unchanged one is never rebuilt. uv's cache in the
store hardlinks package files, so environments share their bytes.

Run by the `rola_devtools` module's environment extension (bazel/environment.bzl) under the machine's `python3`;
prints the environment's paths as JSON on its last line.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

#: a local package's files that are a build's or a tool's leftovers, never its content
SKIPPED = {"__pycache__", ".git", ".pytest_cache", ".ruff_cache"}


def _leftover(name: str, at_root: bool) -> bool:
    """A leftover anywhere, or a build's own folder or a Bazel output link at the package's root."""
    return name in SKIPPED or name.endswith(".egg-info") or (at_root and (name == "build" or name.startswith("bazel-")))


def _content(path: Path) -> bytes:
    """A local package directory's identity: every file's relative path and bytes, in order, leftovers skipped."""
    digest = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = file.relative_to(path)
        if any(_leftover(part, i == 0) for i, part in enumerate(rel.parts)):
            continue
        digest.update(str(rel).encode() + b"\0" + file.read_bytes() + b"\0")
    return digest.hexdigest().encode()


def _identity(package: str) -> bytes:
    path = Path(package)
    return _content(path) if path.is_dir() else package.encode()


def _key(lock: bytes, packages: list[str], python: str, uv_version: str, index: str, isolated: bool) -> str:
    """The environment's key: everything it is built from."""
    parts = [lock, *map(_identity, packages), python.encode(), uv_version.encode(), index.encode()]
    return hashlib.sha256(b"\0".join(parts + ([] if isolated else [b"no-build-isolation"]))).hexdigest()[:16]


def _install(uv: str, python: Path, package: str, uv_env: dict, scratch: Path, isolated: bool) -> None:
    """One package without its dependencies; a local directory from a copy of its content, so its build writes nothing
    into the checkout it came from."""
    if Path(package).is_dir():
        root = str(Path(package))

        def ignored(directory: str, names: list[str]) -> list[str]:
            return [n for n in names if _leftover(n, directory == root)]

        copy = scratch / Path(package).name
        shutil.copytree(package, copy, ignore=ignored, symlinks=True)
        package = str(copy)
    isolation = [] if isolated else ["--no-build-isolation"]
    subprocess.run([uv, "pip", "install", "--python", str(python), "--no-deps", *isolation, package], env=uv_env,
                   check=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--uv", required=True)
    ap.add_argument("--python", required=True, help="the interpreter version uv installs")
    ap.add_argument("--lock", type=Path, required=True, help="the hash-pinned lock (`uv pip compile --generate-hashes`)")
    ap.add_argument("--index", default="", help="an extra index the lock's packages come from (a torch index)")
    ap.add_argument("--no-build-isolation", action="store_true",
                    help="build each package against the environment (its build imports a locked package)")
    ap.add_argument("--package", action="append", default=[],
                    help="installed after the lock without its dependencies: a local directory, or a requirement "
                         "pinned by commit")
    a = ap.parse_args(argv)

    uv_version = subprocess.run([a.uv, "--version"], capture_output=True, text=True, check=True).stdout.strip()
    key = _key(a.lock.read_bytes(), a.package, a.python, uv_version, a.index, not a.no_build_isolation)
    a.store.mkdir(parents=True, exist_ok=True)
    env = a.store / "venvs" / key
    python = env / "bin" / "python3"
    #: uv's own state lives in the store: its interpreters and its cache, never the user's
    uv_env = {**os.environ, "UV_PYTHON_INSTALL_DIR": str(a.store / "python"), "UV_CACHE_DIR": str(a.store / "uv-cache"),
              "UV_PYTHON_PREFERENCE": "only-managed", "UV_NO_CONFIG": "1"}
    index = ["--extra-index-url", a.index, "--index-strategy", "unsafe-best-match"] if a.index else []
    with open(a.store / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (env / ".complete").is_file():
            #: an environment without its completion mark is an interrupted build's
            shutil.rmtree(env, ignore_errors=True)
            #: the interpreter in the store alone: no `python3.X` link in the user's `~/.local/bin`
            subprocess.run([a.uv, "python", "install", "--no-bin", a.python], env=uv_env, check=True)
            subprocess.run([a.uv, "venv", "--python", a.python, str(env)], env=uv_env, check=True)
            subprocess.run([a.uv, "pip", "sync", "--python", str(python), "--require-hashes", str(a.lock), *index],
                           env=uv_env, check=True)
            with tempfile.TemporaryDirectory(dir=a.store) as scratch:
                for package in a.package:
                    _install(a.uv, python, package, uv_env, Path(scratch), not a.no_build_isolation)
            (env / ".complete").write_text(f"{uv_version}\n")
    probe = ("import json, sys, sysconfig; print(json.dumps({'site': sysconfig.get_paths()['purelib'], "
             "'include': sysconfig.get_paths()['include'], 'base': sys.base_prefix}))")
    paths = json.loads(subprocess.run([str(python), "-c", probe], capture_output=True, text=True, check=True).stdout)
    print(json.dumps({"venv": str(env), "python": str(python), **paths}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
