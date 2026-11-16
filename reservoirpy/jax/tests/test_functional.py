# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
"""reservoirpy.jax.functional.pure_run: a node's run as a pure function."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

import reservoirpy.jax.nodes as nodes
from reservoirpy.jax.functional import pure_run

X = np.random.default_rng(0).uniform(-1, 1, (50, 2))
TOL = dict(atol=1e-5, rtol=1e-5)


def reservoir(**params):
    node = nodes.Reservoir(**{**dict(units=20, sr=0.9, lr=0.5, seed=0), **params})
    node.initialize(X)
    return node


def test_same_result_as_run_and_node_untouched():
    node = reservoir()
    run, params, state = pure_run(node)

    final_state, outputs = run(params, state, X)

    assert params == {}
    assert_allclose(node.state["out"], 0.0)  # the node did not move
    assert_allclose(outputs, node.run(X), **TOL)
    assert_allclose(final_state["out"], node.state["out"], **TOL)


def test_starts_from_the_given_state():
    node = reservoir()
    run, params, state = pure_run(node)
    start = {**state, "out": jnp.full(20, 0.3)}

    _, outputs = run(params, start, X)

    node.state = start
    assert_allclose(outputs, node.run(X), **TOL)


def test_continuing_from_final_state_equals_one_long_run():
    run, params, state = pure_run(reservoir())

    middle, first = run(params, state, X[:20])
    _, second = run(params, middle, X[20:])

    assert_allclose(jnp.concatenate([first, second]), run(params, state, X)[1], **TOL)


def test_jit_and_vmap_over_states_and_inputs():
    run, params, state = pure_run(reservoir())
    starts = jnp.linspace(-0.5, 0.5, 4)[:, None] * jnp.ones(20)
    many_states = {**{k: jnp.stack([v] * 4) for k, v in state.items()}, "out": starts}

    _, by_state = jax.jit(jax.vmap(run, in_axes=(None, 0, None)))(params, many_states, X)
    inputs = jnp.stack([X, 2 * X, 3 * X])
    _, by_input = jax.jit(jax.vmap(run, in_axes=(None, None, 0)))(params, state, inputs)

    assert by_state.shape == (4, 50, 20)
    assert_allclose(by_state[1], run(params, {**state, "out": starts[1]}, X)[1], **TOL)
    assert by_input.shape == (3, 50, 20)
    assert_allclose(by_input[2], run(params, state, 3 * X)[1], **TOL)


def test_parameters_can_be_overridden_without_changing_the_node():
    node = reservoir(lr=0.5)
    run, params, state = pure_run(node, "lr")

    assert set(params) == {"lr"} and float(params["lr"]) == 0.5
    _, outputs = run({"lr": 0.2}, state, X)

    assert node.lr == 0.5
    assert_allclose(outputs, reservoir(lr=0.2).run(X), **TOL)


def test_vmap_over_a_parameter():
    run, params, state = pure_run(reservoir(), "lr")
    rates = jnp.array([0.1, 0.5, 1.0])

    _, outputs = jax.vmap(run, in_axes=({"lr": 0}, None, None))({"lr": rates}, state, X)

    for rate, result in zip(rates, outputs):
        assert_allclose(result, reservoir(lr=float(rate)).run(X), **TOL)


@pytest.mark.parametrize("storage", ["sparse", "dense"])
def test_scaling_the_recurrent_matrix_is_scaling_the_spectral_radius(storage):
    node = reservoir(sr=0.6)
    if storage == "dense" and hasattr(node.W, "todense"):
        node.W = node.W.todense()
    run, params, state = pure_run(node, "W")

    _, outputs = run({"W": params["W"] * 1.5}, state, X)

    assert_allclose(outputs, reservoir(sr=0.9).run(X), atol=1e-4, rtol=1e-4)


def test_gradient_with_respect_to_a_parameter():
    run, params, state = pure_run(reservoir(), "lr")

    def loss(rate):
        return jnp.sum(run({"lr": rate}, state, X)[1] ** 2)

    gradient = jax.grad(loss)(0.5)
    numerical = (loss(0.5 + 1e-2) - loss(0.5 - 1e-2)) / 2e-2

    assert jnp.isfinite(gradient)
    assert_allclose(gradient, numerical, rtol=2e-2)


def test_gradient_with_respect_to_the_input():
    run, params, state = pure_run(reservoir())

    gradient = jax.grad(lambda x: jnp.sum(run(params, state, x)[1][-1]))(jnp.asarray(X))

    assert gradient.shape == X.shape
    assert jnp.all(jnp.isfinite(gradient)) and jnp.any(gradient != 0)


def test_noise_follows_the_key_in_the_state():
    node = reservoir(noise_rc=0.1)
    run, params, state = pure_run(node)
    other = {**state, "noise_key": jax.random.split(state["noise_key"])[0]}

    _, first = run(params, state, X)
    _, again = run(params, state, X)
    _, different = run(params, other, X)

    assert_allclose(first, again)
    assert not np.allclose(first, different)


@pytest.mark.parametrize(
    "make",
    [
        lambda: nodes.ES2N(20, sr=0.9, proximity=0.5, seed=0),
        lambda: nodes.LIF(20, sr=0.9, lr=0.1, seed=0),
        lambda: nodes.NVAR(delay=2, order=2),
        lambda: nodes.Tanh(),
        lambda: nodes.Ridge(ridge=1e-3).fit(X, X[:, :1]),
    ],
    ids=["ES2N", "LIF", "NVAR", "Tanh", "Ridge"],
)
def test_other_nodes(make):
    node = make()
    if not node.initialized:
        node.initialize(X)
    run, params, state = pure_run(node)

    _, outputs = jax.jit(run)(params, state, X)

    node.reset()
    assert_allclose(outputs, node.run(X), **TOL)


def test_readout_weights_as_parameter():
    readout = nodes.Ridge(ridge=1e-3).fit(X, X[:, :1])
    run, params, state = pure_run(readout, "Wout", "bias")

    _, outputs = run({"Wout": 2 * params["Wout"], "bias": 0 * params["bias"]}, state, X)

    assert_allclose(outputs, 2 * (readout.run(X) - readout.bias), atol=1e-4)


def test_errors():
    with pytest.raises(RuntimeError, match="initialize"):
        pure_run(nodes.Reservoir(20))
    with pytest.raises(AttributeError, match="no parameter 'leak'"):
        pure_run(reservoir(), "leak")
    with pytest.raises(TypeError, match="Model"):
        pure_run(reservoir() >> nodes.Ridge())
    run, params, state = pure_run(reservoir(), "lr")
    with pytest.raises(KeyError, match="lr"):
        run({}, state, X)
    with pytest.raises(KeyError, match="sr"):
        run({"lr": 0.5, "sr": 1.0}, state, X)


def test_user_guide_examples():
    """Run the code blocks of docs/source/user_guide/jax_transformations.rst."""
    import re
    from pathlib import Path

    import reservoirpy

    guide = Path(reservoirpy.__file__).parent.parent / "docs" / "source" / "user_guide" / "jax_transformations.rst"
    if not guide.exists():
        pytest.skip("documentation sources are not available")
    blocks = re.findall(r"\.\. code-block:: python\n\n((?:    .*\n|\n)+)", guide.read_text())
    assert len(blocks) == 4
    scope = {}

    def run_block(block):
        exec("\n".join(line[4:] for line in block.splitlines()), scope)

    run_block(blocks[0])
    assert_allclose(scope["outputs"], scope["reservoir"].run(scope["x"]), **TOL)
    scope["reservoir"].reset()
    for block in blocks[1:]:
        run_block(block)

    assert float(scope["spread"]) < 1e-5
    spreads = np.asarray(scope["spreads"])
    assert spreads[0] < 1e-5 and spreads[1] < 1e-5 and spreads[2] > 0.1
    assert np.isfinite(float(scope["slope"])) and float(scope["slope"]) != 0


def test_module_docstring_example():
    import re

    import reservoirpy.jax.functional as functional

    block = re.search(r"\.\. code-block:: python\n\n((?:    .*\n|\n)+)", functional.__doc__).group(1)
    scope = {}
    exec("\n".join(line[4:] for line in block.splitlines()), scope)

    assert float(scope["spread"]) < 1e-5
