"""IDENTITIES: what a result depends on, reduced to hashes a unit returns from `identity()`.

    files(root, paths)             the named files and directories, path and bytes each
    code(root, entry, roots, data) a module and every repository file it imports, transitively, plus data files

Paths enter a hash relative to `root`, so the same checkout at two places hashes the same.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path


def digest(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _hash(root: Path, paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()


def _expand(root: Path, names) -> list[Path]:
    out: list[Path] = []
    for name in names:
        path = root / name
        if not path.exists():
            raise FileNotFoundError(f"{name} is not in {root.name}")
        out += sorted(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts) if path.is_dir() \
            else [path]
    return out


def files(root: Path | str, paths) -> str:
    """sha256 over the named files and directories under `root`."""
    root = Path(root)
    return _hash(root, _expand(root, paths))


def code(root: Path | str, entry: str, roots=(".",), data=()) -> str:
    """sha256 over `entry` and every file it imports that resolves under one of `roots` (inside `root`), found by
    walking the imports, plus the `data` files and directories."""
    root = Path(root)
    seen: set[Path] = set()
    todo = [root / entry]
    while todo:
        path = todo.pop()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module]
            else:
                continue
            for name in names:
                rel = name.replace(".", "/")
                for base in roots:
                    for candidate in (root / base / f"{rel}.py", root / base / rel / "__init__.py"):
                        if candidate.exists():
                            todo.append(candidate)
    return _hash(root, sorted(seen) + _expand(root, data))
