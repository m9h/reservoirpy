# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
"""Differences between the Jax and NumPy backends that the user guide
describes (docs/source/user_guide/jax_backend_differences.rst).

If one of these tests fails because the backend improved, update the guide.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse
from numpy.testing import assert_allclose

import reservoirpy
import reservoirpy.jax as rjx
import reservoirpy.jax.nodes as jax_nodes
import reservoirpy.nodes as numpy_nodes

REPO_DIR = Path(reservoirpy.__file__).parent.parent
GUIDE = REPO_DIR / "docs" / "source" / "user_guide" / "jax_backend_differences.rst"

X = np.random.default_rng(0).uniform(size=(200, 2))


def test_single_precision_by_default():
    if jax.config.jax_enable_x64:  # read at run time: other tests may have enabled it
        pytest.skip("64-bit mode is enabled")
    states = jax_nodes.Reservoir(20, seed=0).run(X)
    assert states.dtype == jnp.float32
    assert numpy_nodes.Reservoir(20, seed=0).run(X).dtype == np.float64


ILL_CONDITIONED_FIT = """
import json
import numpy as np
from reservoirpy.datasets import lorenz
from reservoirpy.nodes import NVAR, Ridge
from reservoirpy.jax.nodes import Ridge as JaxRidge

# NVAR features of the Lorenz attractor: values up to ~2000, nearly collinear
X = lorenz(1000, x0=[17.677, 12.931, 43.914], h=0.025, method="RK23")
features = np.asarray(NVAR(2, 2, strides=1).run(X[:600]))
targets = X[1:601] - X[:600]

numpy_ridge = Ridge(ridge=2.5e-6).fit(features, targets)
jax_ridge = JaxRidge(ridge=2.5e-6).fit(features, targets)
difference = np.abs(np.asarray(jax_ridge.Wout) - np.asarray(numpy_ridge.Wout)).max()
print(json.dumps(dict(difference=float(difference), dtype=str(np.asarray(jax_ridge.Wout).dtype))))
"""


def _ill_conditioned_fit(x64: bool) -> dict:
    env = dict(os.environ, JAX_ENABLE_X64="1" if x64 else "0")
    result = subprocess.run(
        [sys.executable, "-c", ILL_CONDITIONED_FIT], env=env, capture_output=True, text=True, cwd=str(REPO_DIR)
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_64_bit_mode_restores_numpy_precision():
    single = _ill_conditioned_fit(x64=False)
    double = _ill_conditioned_fit(x64=True)

    assert single["dtype"] == "float32"
    assert double["dtype"] == "float64"
    # readout weights are of order 1
    assert double["difference"] < 1e-4
    assert single["difference"] > 100 * double["difference"]


def test_storage_of_weights_follows_their_density():
    # (details in test_mat_gen_storage.py)
    default = jax_nodes.Reservoir(1000, seed=0)
    very_sparse = jax_nodes.Reservoir(1000, rc_connectivity=5 / 1000, seed=0)
    default.run(X)
    very_sparse.run(X)

    assert isinstance(default.W, jax.Array)
    assert isinstance(very_sparse.W, sparse.BCOO)


def _numpy_only_nodes():
    return sorted(set(numpy_nodes.__all__) - set(jax_nodes.__all__))


def test_every_jax_node_has_a_numpy_version():
    assert set(jax_nodes.__all__) <= set(numpy_nodes.__all__)


@pytest.mark.skipif(not (REPO_DIR / "docs").exists(), reason="documentation sources are not installed")
def test_guide_lists_the_numpy_only_nodes():
    text = GUIDE.read_text(encoding="utf-8")
    section = text[text.index(".. _jax_numpy_only_nodes:") :]
    section = section[: section.index("\n.. _", 10)] if "\n.. _" in section[10:] else section
    documented = sorted(set(re.findall(r":py:class:`~?reservoirpy\.nodes\.(\w+)`", section)))

    assert documented == _numpy_only_nodes()
