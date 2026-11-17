# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
"""Noise in the LIF node: membrane noise gives spontaneous activity."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from reservoirpy.nodes import LIF

SILENCE = np.zeros((300, 2))
DRIVE = np.random.default_rng(0).uniform(0, 1, (300, 2))


def dense(weights):
    return np.asarray(weights.todense() if hasattr(weights, "todense") else weights)


def lif(**params):
    return LIF(**{**dict(units=30, sr=0.5, lr=0.2, inhibitory=0.2, seed=0), **params})


def test_no_noise_by_default():
    node = lif()

    assert node.noise_rc == 0.0 and node.noise_in == 0.0
    assert_array_equal(node.run(SILENCE), 0.0)  # nothing happens without input


def test_noise_does_not_change_the_weights():
    quiet, noisy = lif(), lif(noise_rc=0.3)
    quiet.initialize(DRIVE), noisy.initialize(DRIVE)

    assert_array_equal(dense(noisy.W), dense(quiet.W))
    assert_array_equal(dense(noisy.Win), dense(quiet.Win))


def test_membrane_noise_gives_spontaneous_spikes():
    spikes = lif(noise_rc=0.4).run(SILENCE)

    assert set(np.unique(spikes)) == {0.0, 1.0}
    assert 0.01 < spikes.mean() < 0.9


def test_more_noise_more_spikes():
    assert lif(noise_rc=0.6).run(SILENCE).mean() > lif(noise_rc=0.3).run(SILENCE).mean()


def test_input_noise_goes_through_the_input_weights():
    spikes = lif(noise_in=2.0, input_connectivity=1.0).run(SILENCE)
    unconnected = LIF(30, Win=np.zeros((30, 2)), noise_in=2.0, seed=0).run(SILENCE)

    assert spikes.mean() > 0.0
    assert_array_equal(unconnected, 0.0)


def test_noise_is_reproducible_with_a_seed_and_differs_between_runs():
    first, second = lif(noise_rc=0.4), lif(noise_rc=0.4)
    a = first.run(DRIVE)

    assert_array_equal(a, second.run(DRIVE))
    first.reset()
    assert not np.array_equal(a, first.run(DRIVE))  # a new run draws new noise
    assert not np.array_equal(a, lif(noise_rc=0.4, seed=1).run(DRIVE))


def test_noise_differs_between_series():
    node = lif(noise_rc=0.4)

    batch = node.run(np.stack([DRIVE, DRIVE]))

    assert not np.array_equal(batch[0], batch[1])


def test_uniform_noise():
    spikes = lif(noise_rc=1.5, noise_type="uniform").run(SILENCE)

    assert 0.0 < spikes.mean() < 1.0


@pytest.mark.parametrize("params", [dict(noise_rc=-0.1), dict(noise_in=-1.0), dict(noise_rc=0.1, noise_type="pink")])
def test_invalid_noise_parameters(params):
    with pytest.raises(ValueError, match="[Nn]oise"):
        lif(**params)
