"""The `ingest` component must depend on nothing outside the standard library.

This is an architectural constraint, not a style preference. The collector runs
unattended on a schedule; a numpy or pyarrow upgrade must never be able to stop
a capture. `pyproject.toml` declares no runtime dependencies for exactly this
reason, and this test is what keeps that declaration honest.

Enforced by parsing the source rather than by importing it, so the test still
fails correctly when the offending dependency happens to be installed in the
developer's venv -- which it always is, via the `analysis` extra.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

INGEST_DIR = Path(__file__).resolve().parents[1] / "src" / "spxrnd" / "ingest"

# Imports of our own package are fine as long as they stay inside `ingest`.
OWN_PACKAGE_PREFIX = "spxrnd.ingest"


def ingest_sources() -> list[Path]:
    return sorted(INGEST_DIR.rglob("*.py"))


def top_level_imports(tree: ast.AST) -> set[str]:
    """Every module name imported by this file, reduced to its top-level name.

    Relative imports (`from .osi import ...`) carry no module name to check and
    are excluded -- they cannot reach outside the package by construction.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative: `from . import x`
                continue
            if node.module:
                names.add(node.module.split(".")[0])
    return names


def test_ingest_directory_exists() -> None:
    assert INGEST_DIR.is_dir(), f"missing component directory: {INGEST_DIR}"


@pytest.mark.parametrize("path", ingest_sources(), ids=lambda p: p.name)
def test_module_imports_only_stdlib(path: Path) -> None:
    tree = ast.parse(path.read_text(), filename=str(path))
    offenders = sorted(
        name
        for name in top_level_imports(tree)
        if name not in sys.stdlib_module_names and name != "spxrnd"
    )
    assert not offenders, (
        f"{path.relative_to(INGEST_DIR.parents[2])} imports non-stdlib modules: "
        f"{offenders}. The ingest component must stay dependency-free so the "
        f"collector cannot be broken by a dependency upgrade. Move the code that "
        f"needs {offenders} into `store` or `curate`."
    )


@pytest.mark.parametrize("path", ingest_sources(), ids=lambda p: p.name)
def test_module_does_not_import_sibling_components(path: Path) -> None:
    """Dependencies flow one way: ingest -> store -> curate -> analytics -> cli.

    `ingest` sits at the head of the chain, so it may import nothing from its
    siblings.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    offenders = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("spxrnd.") and not node.module.startswith(
                OWN_PACKAGE_PREFIX
            ):
                offenders.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("spxrnd.") and not alias.name.startswith(
                    OWN_PACKAGE_PREFIX
                ):
                    offenders.add(alias.name)
    assert not offenders, (
        f"{path.name} imports from a downstream component: {sorted(offenders)}. "
        f"ingest is the head of the dependency chain and must not depend on "
        f"store, curate, analytics, or cli."
    )
