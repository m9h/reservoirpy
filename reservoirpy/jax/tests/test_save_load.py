# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import pickle

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

import reservoirpy as rpy
from reservoirpy.jax import ESN
from reservoirpy.jax import activationsfunc as F
from reservoirpy.jax.nodes import ES2N, LIF, NVAR, IPReservoir, Reservoir, Ridge

X = np.sin(np.linspace(0, 20, 200)).reshape(-1, 1)
Y = np.cos(np.linspace(0, 20, 200)).reshape(-1, 1)


@pytest.mark.parametrize("name", ["tanh", "sigmoid", "relu", "softmax", "softplus", "identity"])
def test_activation_functions_can_be_pickled(name):
    f = F.get_function(name)
    g = pickle.loads(pickle.dumps(f))
    x = jnp.linspace(-1.0, 1.0, 5)
    assert_allclose(np.asarray(g(x)), np.asarray(f(x)))


@pytest.mark.parametrize(
    "node",
    [
        Reservoir(20, seed=1),
        Reservoir(20, activation="sigmoid", seed=1),
        IPReservoir(20, seed=1),
        ES2N(20, seed=1),
        LIF(20, seed=1),
        NVAR(delay=2, order=2),
    ],
    ids=lambda node: type(node).__name__,
)
def test_node_pickle_round_trip(node):
    expected = np.asarray(node.run(X))
    node.reset()
    clone = pickle.loads(pickle.dumps(node))
    assert_allclose(np.asarray(clone.run(X)), expected)


def test_trained_model_save_and_load(tmp_path):
    model = Reservoir(30, seed=1) >> Ridge(ridge=1e-4)
    model.fit(X, Y, warmup=10)
    model.reset()
    expected = np.asarray(model.run(X))

    path = rpy.save(model, tmp_path / "model")
    loaded = rpy.load(path)
    loaded.reset()

    assert_allclose(np.asarray(loaded.run(X)), expected, rtol=1e-5, atol=1e-6)


def test_trained_esn_save_and_load(tmp_path):
    esn = ESN(units=30, ridge=1e-4, seed=1)
    esn.fit(X, Y, warmup=10)
    esn.reset()
    expected = np.asarray(esn.run(X))

    path = rpy.save(esn, tmp_path / "esn.rpy")
    loaded = rpy.load(path)
    loaded.reset()

    assert_allclose(np.asarray(loaded.run(X)), expected, rtol=1e-5, atol=1e-6)
