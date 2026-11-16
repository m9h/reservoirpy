# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
"""Measures of a reservoir that need no task: ranks of its states and
sensitivity to its initial state."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from reservoirpy.nodes import Reservoir
from reservoirpy.observables import (
    echo_state_distance,
    effective_rank,
    generalization_rank,
    kernel_rank,
)


def relay(units, input_dim, seed=0):
    """Linear reservoir without memory: x[t] = Win u[t]."""
    Win = np.random.default_rng(seed).normal(size=(units, input_dim))
    return Reservoir(W=np.zeros((units, units)), Win=Win, bias=0.0, activation="identity")


# --- effective_rank


def test_effective_rank_of_equal_directions():
    rng = np.random.default_rng(0)
    basis, _ = np.linalg.qr(rng.normal(size=(200, 3)))
    mixing, _ = np.linalg.qr(rng.normal(size=(20, 3)))
    states = basis @ mixing.T  # three singular values of 1, the others null

    assert effective_rank(states) == 3


def test_effective_rank_follows_the_threshold():
    states = np.diag([100.0, 1.0, 1.0])

    assert effective_rank(states, threshold=0.95) == 1  # 100 / 102 > 0.95
    assert effective_rank(states, threshold=0.99) == 2  # 101 / 102 > 0.99
    assert effective_rank(states, threshold=1.0) == 3


def test_effective_rank_of_null_states():
    assert effective_rank(np.zeros((50, 10))) == 0


def test_effective_rank_rejects_invalid_threshold():
    with pytest.raises(ValueError):
        effective_rank(np.eye(3), threshold=1.5)


# --- kernel_rank


@pytest.mark.parametrize("input_dim", [1, 2, 3])
def test_kernel_rank_of_a_linear_relay_is_its_input_dimension(input_dim):
    # all final states lie in the span of the columns of Win
    assert kernel_rank(relay(30, input_dim), input_dim=input_dim, seed=0) == input_dim


def test_kernel_rank_of_a_nonlinear_reservoir():
    reservoir = Reservoir(50, sr=0.9, input_scaling=1.0, seed=0)
    rank = kernel_rank(reservoir, seed=0)

    assert 10 < rank <= 50


def test_kernel_rank_is_bounded_by_the_number_of_streams():
    reservoir = Reservoir(50, sr=0.9, input_scaling=1.0, seed=0)

    assert kernel_rank(reservoir, n_streams=7, seed=0) <= 7


def test_kernel_rank_decreases_with_tolerance():
    reservoir = Reservoir(50, sr=0.9, input_scaling=1.0, seed=0)
    ranks = [kernel_rank(reservoir, rtol=rtol, seed=0) for rtol in (1e-12, 1e-6, 1e-3, 1e-1)]

    assert ranks == sorted(ranks, reverse=True)
    assert ranks[-1] < ranks[0]


def test_kernel_rank_is_reproducible_and_leaves_the_node_untouched():
    reservoir = Reservoir(50, sr=0.9, seed=0)
    reservoir.run(np.ones((10, 1)))
    state = reservoir.state["out"].copy()

    assert kernel_rank(reservoir, seed=3) == kernel_rank(reservoir, seed=3)
    assert_array_equal(reservoir.state["out"], state)


def test_kernel_rank_accepts_given_streams():
    streams = np.random.default_rng(0).uniform(-1, 1, size=(12, 40, 2))

    assert kernel_rank(relay(30, 2), series=streams) == 2


def test_kernel_rank_works_on_a_node_never_run():
    assert kernel_rank(Reservoir(20, seed=0), seed=0) > 0


# --- generalization_rank


def test_generalization_rank_without_noise_is_one():
    # every stream is the same: every final state is the same
    reservoir = Reservoir(50, sr=0.9, seed=0)

    assert generalization_rank(reservoir, noise=0.0, seed=0) == 1


def test_generalization_rank_grows_with_noise():
    reservoir = Reservoir(50, sr=0.9, input_scaling=1.0, seed=0)

    assert generalization_rank(reservoir, noise=0.5, rtol=1e-3, seed=0) > 1


def test_generalization_rank_is_larger_for_a_chaotic_reservoir():
    ordered = Reservoir(50, sr=0.5, input_scaling=1.0, seed=0)
    chaotic = Reservoir(50, sr=4.0, input_scaling=1.0, seed=0)

    ordered_rank = generalization_rank(ordered, noise=0.01, rtol=1e-2, seed=0)
    chaotic_rank = generalization_rank(chaotic, noise=0.01, rtol=1e-2, seed=0)
    assert ordered_rank < chaotic_rank


def test_generalization_rank_is_at_most_kernel_rank_plus_one_for_small_noise():
    reservoir = Reservoir(50, sr=0.5, input_scaling=1.0, seed=0)

    assert generalization_rank(reservoir, noise=1e-3, rtol=1e-2, seed=0) <= kernel_rank(reservoir, rtol=1e-2, seed=0)


# --- echo_state_distance


def test_echo_state_distance_vanishes_for_a_contracting_reservoir():
    reservoir = Reservoir(50, sr=0.5, seed=0)

    assert echo_state_distance(reservoir, seed=0) < 1e-8


def test_echo_state_distance_remains_for_a_chaotic_reservoir():
    reservoir = Reservoir(50, sr=4.0, input_scaling=0.01, seed=0)

    assert echo_state_distance(reservoir, seed=0) > 0.1


def test_echo_state_distance_without_time_is_the_distance_between_starts():
    reservoir = Reservoir(50, sr=0.5, seed=0)
    series = np.zeros((0, 1))

    with pytest.raises(ValueError):
        echo_state_distance(reservoir, series=series, seed=0)


def test_echo_state_distance_is_reproducible_and_leaves_the_node_untouched():
    reservoir = Reservoir(50, sr=1.5, seed=0)
    reservoir.run(np.ones((10, 1)))
    state = reservoir.state["out"].copy()

    assert echo_state_distance(reservoir, seed=1) == echo_state_distance(reservoir, seed=1)
    assert_array_equal(reservoir.state["out"], state)
