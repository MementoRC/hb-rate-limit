"""AST guard: rate_limit must not reference hummingbot (ADR 0001).

Covers string imports (importlib.import_module / __import__) that import-linter
cannot see. Static imports are also checked, including under hb_compat.
"""

import ast
from pathlib import Path

import pytest

_PKG_DIR = Path(__file__).resolve().parent.parent / "rate_limit"
_FILES = sorted(_PKG_DIR.rglob("*.py"))
_DYNAMIC_IMPORTERS = {"import_module", "__import__"}
FORBIDDEN_ROOTS = (
    "hummingbot",
    "data_type_primitives",
    "event_bus",
    "logger",
    "async_utils",
    "connector_utils",
    "candles_feed",
    "liquidations_feed",
    "rate_oracle",
    "market_data",
    "market_connector",
    "market_simulator",
    "web_assistant",
    "remote_iface",
    "strategy_framework",
    "order_book",
    "user_stream_tracker",
)

# relpath (relative to the package dir) -> issue URL tracking the violation
_KNOWN_VIOLATIONS: dict[str, str] = {}


def _is_forbidden(name: str) -> bool:
    return any(name == root or name.startswith(root + ".") for root in FORBIDDEN_ROOTS)


def _callee_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _find_violations(tree: ast.AST) -> list[str]:
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            violations.extend(
                f"line {node.lineno}: import {a.name}" for a in node.names if _is_forbidden(a.name)
            )
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module and _is_forbidden(node.module):
                violations.append(f"line {node.lineno}: from {node.module} import ...")
        elif (
            isinstance(node, ast.Call)
            and _callee_name(node.func) in _DYNAMIC_IMPORTERS
            and node.args
        ):
            first = node.args[0]
            if (
                isinstance(first, ast.Constant)
                and isinstance(first.value, str)
                and _is_forbidden(first.value)
            ):
                violations.append(f"line {node.lineno}: dynamic import of {first.value!r}")
    return violations


def test_package_files_discovered() -> None:
    assert _FILES, f"no python files found under {_PKG_DIR}"


@pytest.mark.unit
@pytest.mark.parametrize("path", _FILES, ids=lambda p: str(p.relative_to(_PKG_DIR)))
def test_no_forbidden_references(path: Path) -> None:
    rel = path.relative_to(_PKG_DIR).as_posix()
    if rel in _KNOWN_VIOLATIONS:
        pytest.skip(f"known violation: {_KNOWN_VIOLATIONS[rel]}")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assert _find_violations(tree) == []


@pytest.mark.unit
def test_guard_detects_violations() -> None:
    src = (
        "import hummingbot.core\n"
        "from hummingbot.x import y\n"
        "import importlib\n"
        "importlib.import_module('hummingbot.a')\n"
        "__import__('hummingbot')\n"
        "# import hummingbot\n"
        "'''from hummingbot import z'''\n"
        "importlib.import_module('os')\n"
    )
    assert len(_find_violations(ast.parse(src))) == 4
