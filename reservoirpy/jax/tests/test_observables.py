# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
"""reservoirpy.observables has no Jax version: it must accept Jax models, matrices and arrays."""

import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse
from numpy.testing import assert_allclose

import reservoirpy.jax as rjx
from reservoirpy import observables
from reservoirpy.jax.nodes import Reservoir
from reservoirpy.nodes import Reservoir as NumpyReservoir

X = np.random.default_rng(0).uniform(size=(300, 1))


def test_spectral_radius_of_a_sparse_jax_matrix():
    reservoir = Reservoir(200, sr=0.7, W=rjx.mat_gen.normal(sparsity_type="csr"), seed=0)
    reservoir.run(X)

    assert isinstance(reservoir.W, sparse.BCOO)
    assert_allclose(observables.spectral_radius(reservoir.W), 0.7, rtol=1e-4)


def test_spectral_radius_of_a_dense_jax_matrix():
    reservoir = Reservoir(50, sr=0.7, seed=0)  # stored dense: see reservoirpy.jax.mat_gen
    reservoir.run(X)

    assert_allclose(observables.spectral_radius(reservoir.W), 0.7, rtol=1e-4)


def test_effective_spectral_radius_matches_numpy_backend():
    jax_reservoir = Reservoir(100, sr=0.9, lr=0.5, seed=0)
    numpy_reservoir = NumpyReservoir(100, sr=0.9, lr=0.5, seed=0)
    jax_reservoir.run(X)
    numpy_reservoir.run(X)

    expected = observables.effective_spectral_radius(W=numpy_reservoir.W, lr=0.5)
    assert_allclose(observables.effective_spectral_radius(W=jax_reservoir.W, lr=0.5), expected, rtol=1e-4)


def test_memory_capacity_of_a_jax_model():
    model = rjx.ESN(units=50, sr=0.9, input_scaling=0.1, ridge=1e-6, seed=0)
    capacity = observables.memory_capacity(model, k_max=30, seed=0)

    assert 1.0 < capacity <= 30.0


@pytest.mark.parametrize("metric", ["mse", "rmse", "nrmse", "rsquare", "mae"])
def test_metrics_accept_jax_arrays(metric):
    y_true, y_pred = X, X + 0.1
    expected = getattr(observables, metric)(y_true, y_pred)

    assert_allclose(getattr(observables, metric)(jnp.asarray(y_true), jnp.asarray(y_pred)), expected, rtol=1e-4)
