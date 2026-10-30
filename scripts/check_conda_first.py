#!/usr/bin/env python3
"""Conda-first dependency gate.

Policy: every dependency comes from conda-forge. The only PyPI-sourced packages
allowed are MementoRC hb-* git dependencies and this package's own editable
install. Checks, all against the repo root (cwd):

1. pixi.lock exists and contains no `- pypi:` package other than hb-* git or a
   local path.
2. [tool.pixi.pypi-dependencies] / [tool.pixi.feature.*.pypi-dependencies]
   contain nothing but hb-* dependencies.
3. Every non-hb-* requirement in [project.dependencies] and
   [project.optional-dependencies] is declared on the conda side with an
   identical version range (runtime -> [tool.pixi.dependencies]; an extra ->
   the pixi feature of the same name, else base or any feature).

PyPI and conda names are compared case- and separator-insensitively. A real
rename goes in [tool.conda-first.names] as {pypi-name = "conda-name"}.
Stdlib only, so it runs before any environment is solved.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

HB_GIT = re.compile(r"git\+https://github\.com/MementoRC/hb-[\w.-]+")
LOCK_PYPI = re.compile(r"^\s*-\s+pypi:\s+(\S+)")
REQ = re.compile(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*(.*)$")


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def norm_spec(spec: str) -> str:
    spec = spec.replace(" ", "")
    if spec in ("", "*"):
        return "*"
    parts = [f"={p}" if re.fullmatch(r"=[^=<>!~].*", p) else p for p in spec.split(",")]
    return ",".join(sorted(parts))


def conda_spec(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("version", "*"))
    return str(value)


def is_hb(name: str) -> bool:
    return norm(name).startswith("hb-")


def check_lock(root: Path) -> list[str]:
    lock = root / "pixi.lock"
    if not lock.exists():
        return ["pixi.lock is missing (solve the environment first)"]
    errors = []
    for lineno, line in enumerate(lock.read_text().splitlines(), 1):
        m = LOCK_PYPI.match(line)
        if m is None:
            continue
        src = m.group(1)
        if HB_GIT.match(src) or "://" not in src:
            continue
        errors.append(f"pixi.lock:{lineno}: PyPI-sourced package {src}")
    return errors


def check_pyproject(root: Path) -> list[str]:
    data = tomllib.loads((root / "pyproject.toml").read_text())
    tool = data.get("tool", {})
    pixi = tool.get("pixi", {})
    renames = {norm(k): norm(v) for k, v in tool.get("conda-first", {}).get("names", {}).items()}
    errors = []

    base = {norm(k): norm_spec(conda_spec(v)) for k, v in pixi.get("dependencies", {}).items()}
    features = {
        name: {norm(k): norm_spec(conda_spec(v)) for k, v in feat.get("dependencies", {}).items()}
        for name, feat in pixi.get("feature", {}).items()
    }

    pypi_tables = {"[tool.pixi.pypi-dependencies]": pixi.get("pypi-dependencies", {})}
    for name, feat in pixi.get("feature", {}).items():
        pypi_tables[f"[tool.pixi.feature.{name}.pypi-dependencies]"] = feat.get(
            "pypi-dependencies", {}
        )
    for table, deps in pypi_tables.items():
        for dep in deps:
            if not is_hb(dep):
                errors.append(f"{table}: {dep} must come from conda-forge")

    def check_req(req: str, where: str, candidates: list[dict[str, str]]) -> None:
        m = REQ.match(req)
        if is_hb(m.group(1)):
            return
        name = norm(m.group(1))
        conda_name = renames.get(name, name)
        spec = norm_spec(m.group(2).split(";")[0])
        found = [c[conda_name] for c in candidates if conda_name in c]
        if not found:
            errors.append(f"{where}: {req!r} has no conda-side declaration ({conda_name})")
        elif spec not in found:
            errors.append(f"{where}: {req!r} range differs from conda side {sorted(set(found))}")

    project = data.get("project", {})
    for req in project.get("dependencies", []):
        check_req(req, "[project.dependencies]", [base])
    for extra, reqs in project.get("optional-dependencies", {}).items():
        scope = [features[extra]] if extra in features else [base, *features.values()]
        for req in reqs:
            check_req(req, f"[project.optional-dependencies.{extra}]", scope)
    return errors


def main() -> int:
    root = Path.cwd()
    errors = check_pyproject(root) + check_lock(root)
    for err in errors:
        print(f"conda-first: {err}", file=sys.stderr)
    if errors:
        print(f"conda-first: FAILED ({len(errors)} violation(s))", file=sys.stderr)
        return 1
    print("conda-first: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
