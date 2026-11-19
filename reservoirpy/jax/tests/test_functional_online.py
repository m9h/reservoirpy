# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
"""reservoirpy.jax.functional.pure_partial_fit: online training as a pure function."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

import reservoirpy.jax.nodes as nodes
from reservoirpy.jax.functional import pure_partial_fit, pure_run

RNG = np.random.default_rng(0)
X = RNG.uniform(-1, 1, (80, 3))
Y = X @ RNG.uniform(-1, 1, (3, 2)) + 0.5
TIME = np.arange(300) * 0.1
TARGET, NOTHING = np.sin(TIME)[:, None], np.zeros((300, 0))
TOL = dict(atol=2e-4, rtol=2e-4)


def force(seed=1, initialized=True):
    reservoir, readout = nodes.Reservoir(40, sr=1.2, lr=0.3, seed=seed), nodes.RLS(alpha=1.0)
    model = (reservoir >> readout) & (reservoir << readout)
    if initialized:
        model.initialize(NOTHING, TARGET)
    return model, readout


@pytest.mark.parametrize(
    "make", [lambda: nodes.RLS(alpha=0.1), lambda: nodes.LMS(learning_rate=0.05)], ids=["RLS", "LMS"]
)
def test_node_same_result_as_partial_fit_and_node_untouched(make):
    reference, node = make(), make()
    expected = reference.partial_fit(X, Y)
    node.initialize(X, Y)

    fit, learned, state = pure_partial_fit(node)
    learned_after, state_after, outputs = fit(learned, state, X, Y)

    assert_allclose(outputs, expected, **TOL)
    assert_allclose(learned_after["Wout"], reference.Wout, **TOL)
    assert_allclose(learned_after["bias"], reference.bias, **TOL)
    assert_allclose(state_after["out"], expected[-1], **TOL)
    assert_allclose(node.Wout, 0.0)  # the node itself has learned nothing


def test_node_two_halves_are_one_fit():
    node = nodes.RLS(alpha=0.1)
    node.initialize(X, Y)
    fit, learned, state = pure_partial_fit(node)

    whole, _, _ = fit(learned, state, X, Y)
    half, state_half, _ = fit(learned, state, X[:40], Y[:40])
    halves, _, _ = fit(half, state_half, X[40:], Y[40:])

    assert_allclose(halves["Wout"], whole["Wout"], **TOL)


def test_model_with_feedback_same_result_as_partial_fit_and_model_untouched():
    reference, reference_readout = force()
    expected = reference.partial_fit(NOTHING[:200], TARGET[:200])
    model, readout = force()

    fit, learned, state = pure_partial_fit(model)
    learned_after, state_after, outputs = jax.jit(fit)(learned, state, NOTHING[:200], TARGET[:200])

    assert set(learned) == {"RLS.Wout", "RLS.bias", "RLS.P", "RLS.S"}
    assert_allclose(outputs, expected, atol=2e-3)
    assert_allclose(learned_after["RLS.Wout"], reference_readout.Wout, atol=2e-3)
    assert_allclose(readout.Wout, 0.0)
    assert set(state_after) == {"nodes", "feedback"}


def test_model_then_runs_alone_through_pure_run():
    reference, _ = force()
    reference.partial_fit(NOTHING[:200], TARGET[:200])
    expected = reference.run(NOTHING[200:250])
    model, _ = force()

    fit, learned, state = pure_partial_fit(model)
    learned, state, _ = fit(learned, state, NOTHING[:200], TARGET[:200])
    run, _, _ = pure_run(model, "RLS.Wout", "RLS.bias")
    _, outputs = run({"RLS.Wout": learned["RLS.Wout"], "RLS.bias": learned["RLS.bias"]}, state, NOTHING[200:250])

    assert_allclose(outputs, expected, atol=5e-3)


def test_model_with_an_input_and_a_feedback():
    def build():
        source, reservoir, readout = nodes.Input(), nodes.Reservoir(30, sr=0.9, seed=2), nodes.RLS(alpha=0.5)
        return (source >> reservoir >> readout) & (reservoir << readout)

    target = Y[:, :1]
    expected = build().partial_fit(X, target)
    model = build()
    model.initialize(X, target)

    fit, learned, state = pure_partial_fit(model)
    _, _, outputs = fit(learned, state, X, target)

    assert_allclose(outputs, expected, atol=2e-3)


def test_vmap_over_targets():
    model, _ = force()
    fit, learned, state = pure_partial_fit(model)
    targets = jnp.stack([jnp.sin(f * TIME[:150])[:, None] for f in (0.5, 1.0, 2.0)])

    batched, _, outputs = jax.jit(jax.vmap(fit, in_axes=(None, None, None, 0)))(learned, state, NOTHING[:150], targets)
    single, _, alone = fit(learned, state, NOTHING[:150], targets[2])

    assert outputs.shape == (3, 150, 1)
    assert batched["RLS.Wout"].shape == (3, 40, 1)
    assert_allclose(outputs[2], alone, atol=2e-3)
    assert_allclose(batched["RLS.Wout"][2], single["RLS.Wout"], atol=2e-3)


def test_errors():
    model, _ = force(initialized=False)
    with pytest.raises(RuntimeError, match="initialize"):
        pure_partial_fit(model)
    with pytest.raises(RuntimeError, match="initialize"):
        pure_partial_fit(nodes.RLS())

    ridge = nodes.Ridge(ridge=1e-3).fit(X, Y)
    with pytest.raises(TypeError, match="online"):
        pure_partial_fit(ridge)

    reservoir = nodes.Reservoir(10, seed=0)
    offline = reservoir >> nodes.Ridge(ridge=1e-3)
    offline.fit(X, Y)
    with pytest.raises(TypeError, match="online"):
        pure_partial_fit(offline)

    two = nodes.Reservoir(10, seed=0) >> nodes.RLS(name="first") >> nodes.LMS(name="second")
    two.initialize(X, {"first": Y, "second": Y})
    with pytest.raises(TypeError, match="one"):
        pure_partial_fit(two)
