# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
"""Static check of the tutorial and example notebooks against the current API.

The notebooks are not executed (some take several minutes and need optional
dependencies or downloads). Their code cells are parsed, and every ReservoirPy
name they import or read from a module alias must exist.
"""

import ast
import glob
import importlib
import json
import os
from pathlib import Path

import pytest

import reservoirpy

REPO_DIR = Path(reservoirpy.__file__).parent.parent
NOTEBOOKS = sorted(
    glob.glob(str(REPO_DIR / "tutorials" / "**" / "*.ipynb"), recursive=True)
    + glob.glob(str(REPO_DIR / "examples" / "**" / "*.ipynb"), recursive=True)
)


def _code_cells(path):
    with open(path, encoding="utf-8") as f:
        notebook = json.load(f)
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        lines = "".join(cell["source"]).split("\n")
        # IPython magics and shell escapes are not Python
        lines = [line for line in lines if not line.lstrip().startswith(("%", "!"))]
        yield "\n".join(lines)


def _resolves(dotted_name):
    """True if `dotted_name` (e.g. ``reservoirpy.nodes.Ridge``) can be
    imported or reached by attribute access."""
    parts = dotted_name.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        try:
            for attr in parts[i:]:
                obj = getattr(obj, attr)
        except Exception:
            return False
        return True
    return False


def _reservoirpy_references(source):
    """Names of every ReservoirPy object a code cell imports or accesses
    through a module alias."""
    tree = ast.parse(source)
    aliases = {}
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "reservoirpy":
                    names.add(alias.name)
                    aliases[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else "reservoirpy"
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            if node.module.split(".")[0] == "reservoirpy":
                for alias in node.names:
                    if alias.name != "*":
                        names.add(f"{node.module}.{alias.name}")
    return names, aliases


def _attribute_chain(node):
    chain = []
    while isinstance(node, ast.Attribute):
        chain.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return node.id, list(reversed(chain))
    return None, []


@pytest.mark.skipif(len(NOTEBOOKS) == 0, reason="tutorials and examples are not installed")
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: os.path.relpath(p, REPO_DIR))
def test_notebook_uses_existing_api(path):
    """Every ReservoirPy name a notebook imports, and the first attribute it
    reads from a ReservoirPy module alias, must exist in this version."""
    aliases = {}
    missing = set()
    for source in _code_cells(path):
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        names, new_aliases = _reservoirpy_references(source)
        aliases.update(new_aliases)
        for name in names:
            if not _resolves(name):
                missing.add(name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                root, chain = _attribute_chain(node)
                if root in aliases and chain:
                    name = f"{aliases[root]}.{chain[0]}"
                    if not _resolves(name):
                        missing.add(name)
    assert missing == set()
