# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
"""Storage of the matrices generated for the Jax backend.

A Jax sparse (BCOO) product is only faster than a dense one for very sparse
matrices. Unless a storage is asked for, the initializers pick it from the
size and density of the generated matrix.
"""

import jax
import numpy as np
import pytest
from jax.experimental import sparse
from numpy.testing import assert_allclose

import reservoirpy.jax.mat_gen as jax_mat_gen
import reservoirpy.mat_gen as numpy_mat_gen
from reservoirpy.jax.nodes import Reservoir
from reservoirpy.observables import spectral_radius


def is_sparse(matrix):
    return isinstance(matrix, sparse.BCOO)


def is_dense(matrix):
    return isinstance(matrix, jax.Array)


def test_large_and_not_very_sparse_is_dense():
    assert is_dense(jax_mat_gen.normal(1000, 1000, connectivity=0.1, seed=0))


def test_large_and_very_sparse_is_sparse():
    assert is_sparse(jax_mat_gen.normal(1000, 1000, connectivity=0.005, seed=0))


def test_small_is_dense_whatever_its_density():
    assert is_dense(jax_mat_gen.normal(100, 100, connectivity=0.02, seed=0))


@pytest.mark.parametrize("connectivity", [0.005, 0.5])
def test_requested_storage_is_respected(connectivity):
    assert is_sparse(jax_mat_gen.normal(1000, 1000, connectivity=connectivity, sparsity_type="csr", seed=0))
    assert is_dense(jax_mat_gen.normal(1000, 1000, connectivity=connectivity, sparsity_type="dense", seed=0))


def test_requested_storage_is_respected_when_set_beforehand():
    initializer = jax_mat_gen.normal(sparsity_type="csr")

    assert is_sparse(initializer(1000, 1000, connectivity=0.5, seed=0))


def test_storage_does_not_change_the_values():
    chosen = jax_mat_gen.normal(400, 400, connectivity=0.1, sr=0.9, seed=0)
    as_sparse = jax_mat_gen.normal(400, 400, connectivity=0.1, sr=0.9, sparsity_type="csr", seed=0)
    reference = numpy_mat_gen.normal(400, 400, connectivity=0.1, sr=0.9, seed=0)

    assert is_dense(chosen)
    assert_allclose(np.asarray(chosen), reference.toarray(), rtol=1e-6)
    assert_allclose(np.asarray(as_sparse.todense()), reference.toarray(), rtol=1e-6)


def test_spectral_radius_is_set_on_the_chosen_storage():
    matrix = jax_mat_gen.normal(400, 400, connectivity=0.1, sr=0.7, seed=0)

    assert_allclose(spectral_radius(matrix), 0.7, rtol=1e-4)


def test_threshold_is_documented_and_used():
    density = jax_mat_gen.DENSE_ABOVE_DENSITY
    assert 0.0 < density < 0.1
    assert is_dense(jax_mat_gen.normal(1000, 1000, connectivity=2 * density, seed=0))
    assert is_sparse(jax_mat_gen.normal(1000, 1000, connectivity=density / 3, seed=0))


def test_reservoir_default_connectivity_is_dense_and_tutorial_connectivity_is_sparse():
    X = np.zeros((5, 1))
    default = Reservoir(1000, seed=0)
    very_sparse = Reservoir(1000, rc_connectivity=5 / 1000, seed=0)
    default.run(X)
    very_sparse.run(X)

    assert is_dense(default.W)
    assert is_sparse(very_sparse.W)


def test_reservoir_output_does_not_depend_on_storage():
    X = np.random.default_rng(0).uniform(size=(100, 2))
    chosen = Reservoir(400, seed=0)
    as_sparse = Reservoir(400, W=jax_mat_gen.normal(sparsity_type="csr"), seed=0)

    assert_allclose(chosen.run(X), as_sparse.run(X), atol=1e-5)
    assert is_dense(chosen.W) and is_sparse(as_sparse.W)
