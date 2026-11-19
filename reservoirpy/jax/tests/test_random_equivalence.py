# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
"""The Jax and NumPy backends must agree on randomly drawn configurations.

``nodes/tests/test_numpy_equivalence.py`` compares one configuration per node.
Here the hyperparameters, the shapes and the way the node is called are drawn
at random (from a fixed seed, so that failures can be reproduced), and models
are compared as well as single nodes.
"""

import jax
import numpy as np
import pytest
from numpy.testing import assert_allclose

import reservoirpy.jax as rjx
import reservoirpy.jax.mat_gen as jax_mat_gen
import reservoirpy.jax.nodes as jax_nodes
import reservoirpy.mat_gen as numpy_mat_gen
import reservoirpy.nodes as numpy_nodes
from reservoirpy import ESN as NumpyESN

N_DRAWS = 12
ACTIVATIONS = ["tanh", "sigmoid", "relu", "softplus", "identity"]


def tolerance():
    return 1e-9 if jax.config.jax_enable_x64 else 2e-4


def close(jax_result, numpy_result):
    assert np.shape(jax_result) == np.shape(numpy_result)
    assert_allclose(np.asarray(jax_result), np.asarray(numpy_result), atol=tolerance(), rtol=tolerance())


def series(rng, dim, steps=60):
    return rng.uniform(-1, 1, (steps, dim))


def draw_reservoir(rng):
    units = int(rng.integers(5, 60))
    dim = int(rng.integers(1, 5))
    params = dict(
        units=units,
        sr=float(rng.uniform(0.1, 1.3)),
        lr=float(rng.uniform(0.05, 1.0)) if rng.random() < 0.7 else rng.uniform(0.05, 1.0, units),
        input_scaling=float(rng.uniform(0.1, 2.0)) if rng.random() < 0.7 else list(rng.uniform(0.1, 2.0, dim)),
        input_connectivity=float(rng.uniform(0.1, 1.0)),
        rc_connectivity=float(rng.uniform(0.05, 1.0)),
        activation=str(rng.choice(ACTIVATIONS)),
        seed=int(rng.integers(1000)),
    )
    if rng.random() < 0.5:
        params["bias"] = float(rng.uniform(-0.5, 0.5))
    return params, dim


def draws(draw, salt):
    rng = np.random.default_rng(salt)
    return [draw(rng) for _ in range(N_DRAWS)]


@pytest.mark.parametrize("params, dim", draws(draw_reservoir, 1), ids=range(N_DRAWS))
def test_reservoir(params, dim):
    rng = np.random.default_rng(params["seed"])
    x = series(rng, dim)
    numpy_node, jax_node = numpy_nodes.Reservoir(**params), jax_nodes.Reservoir(**params)

    close(jax_node.run(x), numpy_node.run(x))
    # the state carries over from run to step, and from step to step
    for row in x[:3]:
        close(jax_node.step(row), numpy_node.step(row))
    # several series at once, as an array and as a list of different lengths
    batch = np.stack([series(rng, dim, 20) for _ in range(3)])
    close(jax_node.run(batch), numpy_node.run(batch))
    ragged = [series(rng, dim, 15), series(rng, dim, 25)]
    for jax_out, numpy_out in zip(jax_node.run(ragged), numpy_node.run(ragged)):
        close(jax_out, numpy_out)
    # after a reset both start again from the same state
    numpy_node.reset(), jax_node.reset()
    close(jax_node.run(x[:10]), numpy_node.run(x[:10]))


@pytest.mark.parametrize(
    "initializer", ["uniform", "normal", "bernoulli", "ring", "line", "orthogonal", "cluster", "small_world"]
)
@pytest.mark.parametrize("draw", range(4))
def test_recurrent_initializers(initializer, draw):
    rng = np.random.default_rng([draw, len(initializer)])
    units = int(rng.integers(2, 9)) * 6  # cluster needs a multiple of 3, small_world an even size
    params = dict(units=units, sr=float(rng.uniform(0.2, 1.2)), seed=int(rng.integers(1000)))
    x = series(rng, 2)
    numpy_node = numpy_nodes.Reservoir(W=getattr(numpy_mat_gen, initializer), **params)
    jax_node = jax_nodes.Reservoir(W=getattr(jax_mat_gen, initializer), **params)

    close(jax_node.run(x), numpy_node.run(x))


def draw_es2n(rng):
    params, dim = draw_reservoir(rng)
    del params["lr"]
    params["proximity"] = float(rng.uniform(0.05, 1.0))
    return params, dim


@pytest.mark.parametrize("params, dim", draws(draw_es2n, 2), ids=range(N_DRAWS))
def test_es2n(params, dim):
    x = series(np.random.default_rng(params["seed"]), dim)
    close(jax_nodes.ES2N(**params).run(x), numpy_nodes.ES2N(**params).run(x))


def draw_lif(rng):
    dim = int(rng.integers(1, 4))
    params = dict(
        units=int(rng.integers(5, 60)),
        sr=float(rng.uniform(0.1, 1.3)),
        lr=float(rng.uniform(0.0, 0.9)),
        inhibitory=float(rng.uniform(0.0, 0.5)),
        threshold=float(rng.uniform(0.5, 1.5)),
        input_scaling=float(rng.uniform(0.2, 2.0)),
        input_connectivity=float(rng.uniform(0.1, 1.0)),
        rc_connectivity=float(rng.uniform(0.05, 1.0)),
        seed=int(rng.integers(1000)),
    )
    if rng.random() < 0.5:
        params.update(adaptation=float(rng.uniform(0.05, 0.5)), adaptation_lr=float(rng.uniform(0.01, 0.5)))
    return params, dim


@pytest.mark.parametrize("params, dim", draws(draw_lif, 3), ids=range(N_DRAWS))
def test_lif(params, dim):
    x = np.abs(series(np.random.default_rng(params["seed"]), dim))
    close(jax_nodes.LIF(**params).run(x), numpy_nodes.LIF(**params).run(x))


def draw_ip(rng):
    dim = int(rng.integers(1, 4))
    activation = str(rng.choice(["tanh", "sigmoid"]))
    params = dict(
        units=int(rng.integers(5, 40)),
        sr=float(rng.uniform(0.3, 1.1)),
        lr=float(rng.uniform(0.2, 1.0)),
        activation=activation,
        mu=float(rng.uniform(0.1, 0.4)) if activation == "sigmoid" else float(rng.uniform(-0.2, 0.2)),
        sigma=float(rng.uniform(0.2, 1.0)),
        learning_rate=float(rng.uniform(1e-4, 1e-2)),
        epochs=int(rng.integers(1, 3)),
        seed=int(rng.integers(1000)),
    )
    return params, dim


@pytest.mark.parametrize("params, dim", draws(draw_ip, 4), ids=range(N_DRAWS))
def test_intrinsic_plasticity(params, dim):
    x = series(np.random.default_rng(params["seed"]), dim, steps=80)
    numpy_node, jax_node = numpy_nodes.IPReservoir(**params), jax_nodes.IPReservoir(**params)
    numpy_node.fit(x, warmup=5), jax_node.fit(x, warmup=5)

    close(jax_node.a, numpy_node.a)
    close(jax_node.b, numpy_node.b)
    close(jax_node.run(x), numpy_node.run(x))


@pytest.mark.parametrize("name", ["Tanh", "Sigmoid", "ReLU", "Softplus", "Softmax", "Identity", "Input", "Output"])
def test_stateless_nodes(name):
    x = series(np.random.default_rng(len(name)), 4) * 5
    close(getattr(jax_nodes, name)().run(x), getattr(numpy_nodes, name)().run(x))


def draw_nvar(rng):
    return dict(delay=int(rng.integers(1, 5)), order=int(rng.integers(1, 4)), strides=int(rng.integers(1, 4))), int(
        rng.integers(1, 4)
    )


@pytest.mark.parametrize("params, dim", draws(draw_nvar, 5), ids=range(N_DRAWS))
def test_nvar(params, dim):
    x = series(np.random.default_rng(dim), dim)
    close(jax_nodes.NVAR(**params).run(x), numpy_nodes.NVAR(**params).run(x))


def draw_readout(rng):
    kind = str(rng.choice(["Ridge", "RLS", "LMS"]))
    fit_bias = bool(rng.random() < 0.7)
    if kind == "Ridge":
        params = dict(ridge=float(10 ** rng.uniform(-4, 1)), fit_bias=fit_bias)
    elif kind == "RLS":
        params = dict(
            alpha=float(10 ** rng.uniform(-3, 0)), forgetting=float(rng.uniform(0.95, 1.0)), fit_bias=fit_bias
        )
    else:
        params = dict(learning_rate=float(10 ** rng.uniform(-4, -2)), fit_bias=fit_bias)
    return kind, params, int(rng.integers(1, 6)), int(rng.integers(1, 4)), int(rng.integers(1000))


@pytest.mark.parametrize("kind, params, dim, out_dim, seed", draws(draw_readout, 6), ids=range(N_DRAWS))
def test_readouts(kind, params, dim, out_dim, seed):
    rng = np.random.default_rng(seed)
    x, y = series(rng, dim, 120), series(rng, out_dim, 120)
    numpy_node, jax_node = getattr(numpy_nodes, kind)(**params), getattr(jax_nodes, kind)(**params)
    numpy_node.fit(x, y, warmup=5), jax_node.fit(x, y, warmup=5)

    close(jax_node.Wout, numpy_node.Wout)
    close(jax_node.bias, numpy_node.bias)
    close(jax_node.run(x), numpy_node.run(x))


@pytest.fixture
def double_precision():
    """A ridge regression in single precision can be off by several percent of
    the output (see test_backend_differences.py); models are compared in double."""
    before = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", before)


def draw_model(rng):
    params, dim = draw_reservoir(rng)
    params["sr"] = min(params["sr"], 0.95)
    shape = str(rng.choice(["chain", "input_to_readout", "deep", "esn", "esn_feedback", "esn_input_to_readout"]))
    return shape, params, dim, int(rng.integers(1, 3)), float(10 ** rng.uniform(-4, 0))


def build(nodes, esn, shape, params, ridge):
    reservoir, readout = nodes.Reservoir(**params), nodes.Ridge(ridge=ridge)
    if shape == "chain":
        return reservoir >> readout
    if shape == "input_to_readout":
        data = nodes.Input()
        return [data, data >> reservoir] >> readout
    if shape == "deep":
        second = nodes.Reservoir(**{**params, "seed": params["seed"] + 1, "input_scaling": 0.5})
        return reservoir >> second >> readout
    return esn(
        reservoir=reservoir,
        readout=readout,
        feedback=shape == "esn_feedback",
        input_to_readout=shape == "esn_input_to_readout",
    )


@pytest.mark.parametrize("shape, params, dim, out_dim, ridge", draws(draw_model, 7), ids=range(N_DRAWS))
def test_models(shape, params, dim, out_dim, ridge, double_precision):
    rng = np.random.default_rng(params["seed"])
    x, y = series(rng, dim, 150), series(rng, out_dim, 150)
    numpy_model = build(numpy_nodes, NumpyESN, shape, params, ridge)
    jax_model = build(jax_nodes, rjx.ESN, shape, params, ridge)
    numpy_model.fit(x[:100], y[:100], warmup=10), jax_model.fit(x[:100], y[:100], warmup=10)

    close(jax_model.run(x[100:]), numpy_model.run(x[100:]))
