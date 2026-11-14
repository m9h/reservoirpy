# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from reservoirpy import ESN
from reservoirpy.nodes import Reservoir, Ridge

rng = np.random.default_rng(0)
X = rng.normal(size=(300, 3))


def silent_reservoir(units=50, **kwargs):
    """A reservoir whose state is its noise: no recurrence, input or bias."""
    return Reservoir(W=np.zeros((units, units)), Win=np.zeros((units, 3)), bias=0.0, **kwargs)


def relay_reservoir(**kwargs):
    """A reservoir that copies its 3 inputs: x[t] = u[t]."""
    return Reservoir(W=np.zeros((3, 3)), Win=np.eye(3), bias=0.0, activation="identity", **kwargs)


def test_no_noise_by_default():
    res = Reservoir(20, seed=1)
    assert res.noise_rc == 0.0
    assert res.noise_in == 0.0
    assert res.noise_type == "normal"

    out = res.run(X)
    res.reset()
    assert_array_equal(res.run(X), out)


def test_noise_does_not_change_the_weights():
    plain = Reservoir(20, seed=1)
    noisy = Reservoir(20, seed=1, noise_rc=0.1, noise_in=0.1)
    plain.run(X)
    noisy.run(X)

    assert_array_equal(noisy.W.toarray(), plain.W.toarray())
    assert_array_equal(noisy.Win.toarray(), plain.Win.toarray())


def test_noise_rc_is_added_to_the_state():
    res = silent_reservoir(noise_rc=0.2, seed=1)
    states = res.run(np.zeros((2000, 3)))

    assert_allclose(states.mean(), 0.0, atol=0.01)
    assert_allclose(states.std(), 0.2, rtol=0.05)


def test_noise_rc_is_outside_the_leaky_integration():
    # x[t+1] = (1 - lr) * x[t] + lr * f(...) + noise: the noise gain does not depend on lr
    res = silent_reservoir(noise_rc=0.2, lr=0.5, seed=1)
    states = res.run(np.zeros((4000, 3)))
    increments = states[1:] - 0.5 * states[:-1]

    assert_allclose(increments.std(), 0.2, rtol=0.05)


def test_noise_in_is_added_to_the_input():
    res = relay_reservoir(noise_in=0.3, seed=1)
    states = res.run(X.repeat(10, axis=0))
    difference = states - X.repeat(10, axis=0)

    assert_allclose(difference.mean(), 0.0, atol=0.02)
    assert_allclose(difference.std(), 0.3, rtol=0.05)


def test_noise_is_scaled_by_input_weights():
    res = Reservoir(W=np.zeros((3, 3)), Win=2.0 * np.eye(3), bias=0.0, activation="identity", noise_in=0.3, seed=1)
    states = res.run(np.zeros((3000, 3)))

    assert_allclose(states.std(), 0.6, rtol=0.05)


def test_noise_is_reproducible_with_a_seed():
    a = Reservoir(20, noise_rc=0.1, noise_in=0.1, seed=3).run(X)
    b = Reservoir(20, noise_rc=0.1, noise_in=0.1, seed=3).run(X)
    c = Reservoir(20, noise_rc=0.1, noise_in=0.1, seed=4).run(X)

    assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_noise_differs_between_runs():
    res = Reservoir(20, noise_rc=0.1, seed=3)
    first = res.run(X)
    res.reset()
    second = res.run(X)

    assert not np.allclose(first, second)


def test_noise_differs_between_series():
    res = silent_reservoir(noise_rc=0.1, seed=3)
    states = res.run(np.zeros((4, 50, 3)))

    assert not np.allclose(states[0], states[1])


def test_step_uses_noise():
    res = silent_reservoir(noise_rc=0.1, seed=3)
    assert np.abs(res.step(np.zeros(3))).sum() > 0


def test_uniform_noise_is_bounded():
    res = silent_reservoir(noise_rc=0.2, noise_type="uniform", seed=1)
    states = res.run(np.zeros((2000, 3)))

    assert np.abs(states).max() <= 0.2
    assert np.abs(states).max() > 0.19
    # variance of a uniform distribution on [-1, 1] is 1/3
    assert_allclose(states.std(), 0.2 / np.sqrt(3), rtol=0.05)


def test_noise_kwargs():
    res = silent_reservoir(noise_rc=1.0, noise_type="uniform", noise_kwargs=dict(low=2.0, high=3.0), seed=1)
    states = res.run(np.zeros((200, 3)))

    assert states.min() >= 2.0
    assert states.max() <= 3.0


@pytest.mark.parametrize("params", [dict(noise_rc=-0.1), dict(noise_in=-1.0), dict(noise_type="not_a_distribution")])
def test_invalid_noise_parameters(params):
    with pytest.raises(ValueError):
        Reservoir(10, **params)


def test_noise_in_a_model():
    model = Reservoir(30, noise_rc=0.01, noise_in=0.01, seed=1) >> Ridge(ridge=1e-3)
    model.fit(X[:200], X[1:201], warmup=10)
    assert model.run(X[200:]).shape == (100, 3)

    esn = ESN(units=30, noise_rc=0.01, ridge=1e-3, seed=1)
    esn.fit(X[:200], X[1:201], warmup=10)
    assert esn.reservoir.noise_rc == 0.01
    assert esn.run(X[200:]).shape == (100, 3)


def test_noise_regularises_the_readout():
    # State noise during training acts like a ridge penalty: readout weights get smaller.
    def weight_norm(noise_rc):
        model = Reservoir(50, sr=0.9, noise_rc=noise_rc, seed=1) >> Ridge(ridge=1e-9)
        model.fit(X[:-1], X[1:], warmup=10)
        return np.linalg.norm(model.nodes[1].Wout)

    assert weight_norm(0.1) < weight_norm(0.0)
