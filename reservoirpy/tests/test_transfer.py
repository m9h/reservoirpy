# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
"""reservoirpy.transfer: carry a trained readout from one reservoir to another."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.transfer import align_states, transplant_readout

rng = np.random.default_rng(0)
EXPERT = rng.normal(size=(300, 20))
TARGETS = np.tanh(EXPERT @ rng.normal(size=(20, 2))) + 0.5
MIX, OFFSET = rng.normal(size=(20, 20)), rng.normal(size=20)
STUDENT = EXPERT @ MIX + OFFSET  # the same activity seen through another basis


def expert_readout():
    return Ridge(ridge=1e-6).fit(EXPERT, TARGETS)


def test_align_states_recovers_a_linear_map():
    matrix, offset = align_states(STUDENT, EXPERT, ridge=1e-10)

    assert matrix.shape == (20, 20) and offset.shape == (20,)
    assert_allclose(STUDENT @ matrix + offset, EXPERT, atol=1e-6)


def test_align_states_between_different_sizes():
    wide = np.hstack([STUDENT, rng.normal(size=(300, 15))])

    matrix, offset = align_states(wide, EXPERT, ridge=1e-8)

    assert matrix.shape == (35, 20)
    assert_allclose(wide @ matrix + offset, EXPERT, atol=1e-4)


def test_ridge_shrinks_the_map():
    loose, _ = align_states(STUDENT, EXPERT, ridge=1e-10)
    tight, _ = align_states(STUDENT, EXPERT, ridge=1e4)

    assert np.linalg.norm(tight) < 0.5 * np.linalg.norm(loose)


def test_transplanted_readout_reproduces_the_expert():
    expert = expert_readout()
    weights_before = expert.Wout.copy()

    student = transplant_readout(expert, STUDENT, EXPERT, ridge=1e-10)

    assert type(student) is Ridge and student is not expert
    assert student.initialized and student.input_dim == 20 and student.output_dim == 2
    assert_allclose(student.run(STUDENT), expert.run(EXPERT), atol=1e-5)
    assert_allclose(expert.Wout, weights_before)  # the expert is left as it was


def test_student_of_another_size():
    wide = np.hstack([STUDENT, rng.normal(size=(300, 15))])
    expert = expert_readout()

    student = transplant_readout(expert, wide, EXPERT, ridge=1e-8)

    assert student.Wout.shape == (35, 2) and student.input_dim == 35
    assert_allclose(student.run(wide), expert.run(EXPERT), atol=1e-3)


def test_lists_of_series_are_concatenated():
    expert = expert_readout()
    whole = transplant_readout(expert, STUDENT, EXPERT, ridge=1e-6)

    pieces = transplant_readout(expert, [STUDENT[:100], STUDENT[100:]], [EXPERT[:100], EXPERT[100:]], ridge=1e-6)
    stacked = transplant_readout(expert, STUDENT.reshape(3, 100, 20), EXPERT.reshape(3, 100, 20), ridge=1e-6)

    assert_allclose(pieces.Wout, whole.Wout)
    assert_allclose(stacked.Wout, whole.Wout)


def test_refinement_moves_between_the_transplant_and_a_fresh_fit():
    expert = expert_readout()
    student_states = STUDENT + 0.3 * rng.normal(size=STUDENT.shape)  # an imperfect copy
    transplanted = transplant_readout(expert, student_states, EXPERT, ridge=1e-3)
    fresh = Ridge(ridge=1e-9).fit(student_states, TARGETS)

    held = transplant_readout(expert, student_states, EXPERT, ridge=1e-3, targets=TARGETS, prior_ridge=1e12)
    free = transplant_readout(expert, student_states, EXPERT, ridge=1e-3, targets=TARGETS, prior_ridge=1e-9)

    assert_allclose(held.Wout, transplanted.Wout, atol=1e-6)
    assert_allclose(free.Wout, fresh.Wout, atol=1e-5)
    assert_allclose(free.bias, fresh.bias, atol=1e-5)


def paired_reservoirs():
    """Two different reservoirs given the same input, and a task with memory and a nonlinearity."""
    u = np.random.default_rng(1).uniform(0, 0.5, (3100, 1))
    y = (np.roll(u, 3, axis=0) * np.roll(u, 1, axis=0))[100:]
    expert_states = Reservoir(50, sr=0.8, lr=0.5, seed=11).run(u)[100:]
    student_states = Reservoir(50, sr=0.8, lr=0.5, seed=21).run(u)[100:]
    expert = Ridge(ridge=1e-6).fit(expert_states[:1500], y[:1500])
    test = slice(2600, 3000)

    def error(readout, states=student_states):
        return float(np.sqrt(np.mean((readout.run(states[test]) - y[test]) ** 2)))

    return expert, expert_states, student_states, y, error


def test_transplant_needs_no_labels():
    """1000 paired steps without targets beat 40 steps with targets."""
    expert, expert_states, student_states, y, error = paired_reservoirs()
    paired, labelled = slice(1500, 2500), slice(1500, 1540)

    transplanted = transplant_readout(expert, student_states[paired], expert_states[paired])
    scratch = Ridge(ridge=1e-6).fit(student_states[labelled], y[labelled])

    assert error(transplanted) < 1.1 * error(expert, expert_states)  # the student does as well as the expert
    assert error(transplanted) < 0.8 * error(scratch)


def test_transplant_is_no_shortcut_when_paired_steps_are_few():
    """The map between states has as many unknowns as a readout: 40 paired steps do not determine it."""
    expert, expert_states, student_states, y, error = paired_reservoirs()
    few = slice(1500, 1540)

    transplanted = transplant_readout(expert, student_states[few], expert_states[few], ridge=1e-4)

    assert error(transplanted) > 1.2 * error(expert, expert_states)


def test_jax_readout():
    jax_nodes = pytest.importorskip("reservoirpy.jax.nodes")
    expert = jax_nodes.Ridge(ridge=1e-3).fit(EXPERT, TARGETS)

    student = transplant_readout(expert, STUDENT, EXPERT, ridge=1e-6)

    assert type(student) is jax_nodes.Ridge
    assert type(student.Wout) is type(expert.Wout)
    assert_allclose(student.run(STUDENT), expert.run(EXPERT), atol=5e-2)


def test_errors():
    expert = expert_readout()
    with pytest.raises(ValueError, match="same number of timesteps"):
        transplant_readout(expert, STUDENT[:200], EXPERT)
    with pytest.raises(ValueError, match="20 features"):
        transplant_readout(expert, STUDENT, EXPERT[:, :10])
    with pytest.raises(ValueError, match="not fitted"):
        transplant_readout(Ridge(), STUDENT, EXPERT)
    with pytest.raises(TypeError, match="Wout"):
        transplant_readout(Reservoir(20), STUDENT, EXPERT)
    with pytest.raises(ValueError, match="targets"):
        transplant_readout(expert, STUDENT, EXPERT, targets=TARGETS[:100])
    with pytest.raises(ValueError, match="positive"):
        align_states(STUDENT, EXPERT, ridge=-1.0)


def test_module_docstring_example():
    import re

    import reservoirpy.transfer as transfer

    block = re.search(r"\.\. code-block:: python\n\n((?:    .*\n|\n)+)", transfer.__doc__).group(1)
    scope = {}
    exec("\n".join(line[4:] for line in block.splitlines()), scope)

    assert scope["error"] < 0.1 * float(scope["y"].std())
