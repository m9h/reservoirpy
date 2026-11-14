# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
"""Package-level checks: clean import, no side effects."""

import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

import reservoirpy

PACKAGE_DIR = Path(reservoirpy.__file__).parent
REPO_DIR = PACKAGE_DIR.parent
SOURCE_FILES = sorted(str(p) for p in PACKAGE_DIR.rglob("*.py"))


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: os.path.relpath(p, PACKAGE_DIR))
def test_source_compiles_without_warning(path):
    """Invalid escape sequences are a SyntaxWarning since Python 3.12 and
    will become an error."""
    with open(path, encoding="utf-8") as f:
        source = f.read()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compile(source, path, "exec")


def _run_python(code, tmp_path):
    env = dict(os.environ, TMPDIR=str(tmp_path), TEMP=str(tmp_path), TMP=str(tmp_path))
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=str(REPO_DIR))
    assert result.returncode == 0, result.stderr
    return result


def test_import_leaves_no_temporary_directory(tmp_path):
    _run_python("import reservoirpy", tmp_path)
    assert os.listdir(tmp_path) == []


def test_datasets_cache_is_removed_at_exit(tmp_path):
    code = "from reservoirpy.datasets import mackey_glass; mackey_glass(100)"
    _run_python(code, tmp_path)
    assert os.listdir(tmp_path) == []


def test_datasets_is_reachable_as_attribute(tmp_path):
    _run_python("import reservoirpy as rpy; rpy.datasets.mackey_glass(10)", tmp_path)
