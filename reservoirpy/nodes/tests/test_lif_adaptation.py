# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
"""Spike-frequency adaptation in the LIF node: every spike of a neuron adds to
a current that holds it back and decays slowly. With noise and strong
recurrent excitation, this turns saturation into population bursts."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from reservoirpy.nodes import LIF

SILENCE = np.zeros((1500, 1))
CONSTANT = np.full((200, 1), 0.6)


def neuron(**params):
    """One neuron without recurrence, which receives its input unchanged."""
    return LIF(W=np.zeros((1, 1)), Win=np.ones((1, 1)), **params)


def culture(**params):
    defaults = dict(units=200, inhibitory=0.2, sr=8.0, lr=0.1, noise_rc=0.2, seed=0)
    return LIF(**{**defaults, **params})


def burst_starts(spikes, level=0.5):
    """Timesteps at which more than half of the neurons start firing together."""
    active = spikes.mean(axis=1) > level
    return np.flatnonzero(active[1:] & ~active[:-1])


def test_no_adaptation_by_default():
    node = neuron()
    node.run(CONSTANT)

    assert node.adaptation == 0.0
    assert "adaptation" not in node.state


@pytest.mark.parametrize("params", [dict(adaptation=-0.1), dict(adaptation_lr=-0.1), dict(adaptation_lr=1.5)])
def test_invalid_values_are_refused(params):
    with pytest.raises(ValueError):
        neuron(**params)


def test_matches_a_hand_written_neuron():
    increment, decay, leak = 0.3, 0.1, 0.05
    spikes = neuron(lr=leak, adaptation=increment, adaptation_lr=decay).run(CONSTANT)

    v, a, expected = 0.0, 0.0, []
    for x in CONSTANT[:, 0]:
        v *= 1 - leak
        spike = float(v > 1.0)
        v = 0.0 if spike else v
        a = (1 - decay) * a + increment * spike
        v += x - a
        expected.append(spike)

    assert 0 < sum(expected) < len(expected)
    assert_array_equal(spikes[:, 0], expected)


def test_a_neuron_slows_down_under_constant_input():
    plain = neuron().run(CONSTANT)[:, 0]
    adapting = neuron(adaptation=0.2, adaptation_lr=0.02).run(CONSTANT)[:, 0]
    intervals = np.diff(np.flatnonzero(adapting))

    assert adapting.sum() < plain.sum()
    assert intervals[-1] > intervals[0]


def test_the_current_jumps_at_a_spike_and_decays_after():
    node = neuron(adaptation=0.5, adaptation_lr=0.25)
    node.run(np.array([[2.0], [0.0]]))  # the spike is emitted on the second step
    assert_allclose(node.state["adaptation"], [0.5])

    node.run(np.zeros((2, 1)))
    assert_allclose(node.state["adaptation"], [0.5 * 0.75**2])


def test_reset_clears_the_current():
    node = neuron(adaptation=0.5)
    node.run(CONSTANT)
    assert node.state["adaptation"][0] > 0.0

    node.reset()
    assert_array_equal(node.state["adaptation"], 0.0)


def test_adaptation_does_not_change_the_weights():
    plain, adapting = culture(), culture(adaptation=0.1)
    plain.initialize(SILENCE), adapting.initialize(SILENCE)

    assert (plain.W != adapting.W).nnz == 0


def test_without_adaptation_strong_recurrence_saturates():
    spikes = culture().run(SILENCE)

    assert spikes[-500:].mean() > 0.8
    assert len(burst_starts(spikes)) == 1


def test_with_adaptation_the_network_bursts_without_input():
    spikes = culture(adaptation=0.1, adaptation_lr=0.02).run(SILENCE)
    rate = spikes.mean(axis=1)

    assert len(burst_starts(spikes)) >= 3  # the whole network fires together, again and again
    assert np.mean(rate < 0.05) > 0.5  # and is nearly silent in between
