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
    assert len(blocks) == 6
    scope = {}

    def run_block(block):
        exec("\n".join(line[4:] for line in block.splitlines()), scope)

    run_block(blocks[0])
    assert_allclose(scope["outputs"], scope["reservoir"].run(scope["x"]), **TOL)
    scope["reservoir"].reset()
    for block in blocks[1:]:
        run_block(block)
    # FORCE learning: the three trained models generate their targets alone
    assert scope["generated"].shape == (3, 500, 1)
    assert float(scope["errors"].max()) < 0.1

    assert float(scope["spread"]) < 1e-5
    spreads = np.asarray(scope["spreads"])
    assert spreads[0] < 1e-5 and spreads[1] < 1e-5 and spreads[2] > 0.1
    assert np.isfinite(float(scope["slope"])) and float(scope["slope"]) != 0
    esn_state = scope["final_state"]
    assert_allclose(scope["predictions"], scope["esn"].run(scope["x"][400:]), **TOL)
    assert set(esn_state["nodes"]) == {"Reservoir", "readout"}
    changes = np.asarray(scope["changes"])
    assert changes[0] < 1e-10 < changes[1] < changes[2]  # batched and single runs differ by rounding


def test_module_docstring_example():
    import re

    import reservoirpy.jax.functional as functional

    block = re.search(r"\.\. code-block:: python\n\n((?:    .*\n|\n)+)", functional.__doc__).group(1)
    scope = {}
    exec("\n".join(line[4:] for line in block.splitlines()), scope)

    assert float(scope["spread"]) < 1e-5


# Models
Y = np.random.default_rng(1).uniform(-1, 1, (50, 1))


def fitted(model):
    model.fit(X, Y, warmup=5)
    for node in model.nodes:
        node.reset() if hasattr(node, "reset") else None
    model.reset()
    return model


def chain():
    return fitted(nodes.Reservoir(20, sr=0.9, lr=0.5, seed=0) >> nodes.Ridge(ridge=1e-2))


def test_model_same_result_as_run_and_model_untouched():
    model = chain()
    run, params, state = pure_run(model)

    final_state, outputs = jax.jit(run)(params, state, X)

    assert set(state["nodes"]) == {"Reservoir", "Ridge"}
    assert_allclose(model.nodes[0].state["out"], 0.0)
    assert_allclose(outputs, model.run(X), **TOL)
    assert_allclose(final_state["nodes"]["Reservoir"]["out"], model.nodes[0].state["out"], **TOL)


@pytest.mark.parametrize("feedback, input_to_readout", [(True, False), (False, True), (True, True)])
def test_esn_with_feedback_and_input_to_readout(feedback, input_to_readout):
    import reservoirpy.jax as rjx

    model = fitted(rjx.ESN(units=20, sr=0.9, ridge=1e-2, seed=0, feedback=feedback, input_to_readout=input_to_readout))
    run, params, state = pure_run(model)

    _, outputs = run(params, state, X)

    assert_allclose(outputs, model.run(X), **TOL)
    assert len(state["feedback"]) == int(feedback)


def test_model_continuing_from_final_state_with_feedback():
    import reservoirpy.jax as rjx

    run, params, state = pure_run(fitted(rjx.ESN(units=20, sr=0.9, ridge=1e-2, seed=0, feedback=True)))

    middle, first = run(params, state, X[:20])
    _, second = run(params, middle, X[20:])

    assert_allclose(jnp.concatenate([first, second]), run(params, state, X)[1], **TOL)


def test_model_parameters_are_named_by_node():
    model = chain()
    run, params, state = pure_run(model, "Reservoir.lr", "Ridge.Wout")

    assert set(params) == {"Reservoir.lr", "Ridge.Wout"}
    _, doubled = run({**params, "Ridge.Wout": 2 * params["Ridge.Wout"]}, state, X)
    _, plain = run(params, state, X)
    bias = model.nodes[1].bias

    assert_allclose(doubled - bias, 2 * (plain - bias), atol=1e-4)
    assert model.nodes[0].lr == 0.5


def test_model_vmap_over_starting_states_and_gradient():
    run, params, state = pure_run(chain(), "Ridge.Wout")
    starts = jnp.linspace(-0.5, 0.5, 4)[:, None] * jnp.ones(20)
    many = jax.tree_util.tree_map(lambda leaf: jnp.stack([leaf] * 4), state)
    many["nodes"]["Reservoir"]["out"] = starts

    _, outputs = jax.vmap(run, in_axes=(None, 0, None))(params, many, X)
    gradient = jax.grad(lambda p: jnp.sum(run(p, state, X)[1] ** 2))(params)

    assert outputs.shape == (4, 50, 1)
    assert not np.allclose(outputs[0, 0], outputs[3, 0])
    assert gradient["Ridge.Wout"].shape == params["Ridge.Wout"].shape
    assert jnp.all(jnp.isfinite(gradient["Ridge.Wout"]))


def test_model_nodes_of_the_same_class_need_names():
    def deep(**names):
        first = nodes.Reservoir(20, seed=0, name=names.get("first"))
        second = nodes.Reservoir(15, seed=1, name=names.get("second"))
        return fitted(first >> second >> nodes.Ridge(ridge=1e-2))

    with pytest.raises(ValueError, match="name"):
        pure_run(deep())
    model = deep(first="fast", second="slow")
    run, params, state = pure_run(model, "slow.lr")

    assert set(state["nodes"]) == {"fast", "slow", "Ridge"}
    assert_allclose(run(params, state, X)[1], model.run(X), **TOL)


def test_model_with_several_outputs():
    reservoir = nodes.Reservoir(20, seed=0)
    model = fitted(reservoir >> [nodes.Ridge(ridge=1e-2, name="a"), nodes.Ridge(ridge=1.0, name="b")])
    run, params, state = pure_run(model)

    _, outputs = run(params, state, X)
    expected = model.run(X)

    assert set(outputs) == {"a", "b"}
    assert_allclose(outputs["a"], expected["a"], **TOL)
    assert_allclose(outputs["b"], expected["b"], **TOL)


def test_model_errors():
    with pytest.raises(RuntimeError, match="initialize"):
        pure_run(nodes.Reservoir(20) >> nodes.Ridge())
    with pytest.raises(AttributeError, match="Reservoir.leak"):
        pure_run(chain(), "Reservoir.leak")
    with pytest.raises(ValueError, match="'node.parameter'"):
        pure_run(chain(), "lr")
    with pytest.raises(ValueError, match="no node 'Tanh'"):
        pure_run(chain(), "Tanh.lr")
