# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
"""
==============================================
Readout transfer (:mod:`reservoirpy.transfer`)
==============================================

Carry a trained readout from one reservoir (the *expert*) to another (the
*student*) that received the same inputs.

Two reservoirs driven by the same input do not have the same states: their
units are not the same units. When both carry the same information, a linear
map takes the states of one onto the states of the other. With that map, a
readout trained on the expert can be used on the student without training it
again. This is the "knowledge transplant" of Kim et al. (2026) [1]_, used
there to move a readout between living neuronal cultures, and the functions
follow the ``KnowledgeTransfer`` operator of their MiV-OS library.

.. autosummary::
   :toctree: generated/

    align_states
    transplant_readout

Example
=======

.. code-block:: python

    import numpy as np
    from reservoirpy.nodes import Reservoir, Ridge
    from reservoirpy.transfer import transplant_readout

    u = np.random.default_rng(0).uniform(-1, 1, (1200, 1))
    y = np.roll(u, 2, axis=0)                      # task: recall the input two steps back
    expert_states = Reservoir(50, sr=0.8, seed=1).run(u)
    student_states = Reservoir(50, sr=0.8, seed=2).run(u)

    expert = Ridge(ridge=1e-6).fit(expert_states[:1000], y[:1000], warmup=50)
    # 100 steps recorded from both reservoirs are enough to move the readout
    student = transplant_readout(expert, student_states[1000:1100], expert_states[1000:1100], ridge=1e-6)
    error = np.sqrt(np.mean((student.run(student_states[1100:]) - y[1100:]) ** 2))

References
==========

.. [1] Kim et al. (2026). Computing with living neurons: chaos-controlled
       reservoir computing with knowledge transplant. arXiv:2604.02552.
"""

import copy
from typing import Optional, Sequence, Union

import numpy as np

__all__ = ["align_states", "transplant_readout"]

States = Union[np.ndarray, Sequence[np.ndarray]]


def _stack(states: States, name: str) -> np.ndarray:
    """Timeseries, or several of them, as one array of shape (timesteps, features)."""
    if isinstance(states, (list, tuple)):
        states = np.concatenate([np.asarray(series, dtype=float) for series in states], axis=0)
    states = np.asarray(states, dtype=float)
    if states.ndim == 3:
        states = states.reshape(-1, states.shape[-1])
    if states.ndim != 2:
        raise ValueError(f"'{name}' must have shape ([series,] timesteps, features), got {states.shape}.")
    return states


def _ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    """Weights and intercept of a ridge regression from x to y. The intercept is not penalized."""
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x - x_mean, y - y_mean
    weights = np.linalg.solve(xc.T @ xc + ridge * np.eye(x.shape[1]), xc.T @ yc)
    return weights, y_mean - x_mean @ weights


def align_states(source: States, target: States, ridge: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Linear map from the states of one reservoir to those of another.

    Fits ``matrix`` and ``offset`` such that ``source @ matrix + offset``
    is close to ``target``, by ridge regression. Row ``t`` of ``source`` and
    row ``t`` of ``target`` must come from the same moment of the same input.

    Parameters
    ----------
    source : array of shape ([series,] timesteps, source_features) or list of arrays
        States to map from.
    target : array of shape ([series,] timesteps, target_features) or list of arrays
        States to map onto, paired with ``source``.
    ridge : float, default to 1e-6
        Regularization of the map. Raise it when there are few paired
        timesteps compared with the number of features.

    Returns
    -------
    matrix : array of shape (source_features, target_features)
    offset : array of shape (target_features,)
    """
    if ridge < 0:
        raise ValueError(f"'ridge' must be positive, got {ridge}.")
    source, target = _stack(source, "source"), _stack(target, "target")
    if len(source) != len(target):
        raise ValueError(f"Paired states must have the same number of timesteps, got {len(source)} and {len(target)}.")
    return _ridge(source, target, ridge)


def transplant_readout(
    readout,
    student_states: States,
    expert_states: States,
    ridge: float = 1e-6,
    targets: Optional[States] = None,
    prior_ridge: float = 1.0,
    target_states: Optional[States] = None,
):
    """Readout for a student reservoir, made from the readout of an expert.

    The student states are mapped onto the expert states with
    :py:func:`align_states`, and the expert readout is composed with that
    map. No target is needed for this: only states of the two reservoirs
    recorded under the same input.

    If ``targets`` are given, the transplanted readout is then refined on
    them. The refinement is a ridge regression that shrinks the weights
    towards the transplanted ones instead of towards zero: with few labelled
    timesteps it stays close to the transplant, with many it approaches a
    readout trained on the student alone.

    Parameters
    ----------
    readout : Node
        A fitted linear readout of the expert, with ``Wout`` and ``bias``
        (:py:class:`~reservoirpy.nodes.Ridge`, :py:class:`~reservoirpy.nodes.RLS`,
        :py:class:`~reservoirpy.nodes.LMS`, or their Jax versions). It is not modified.
    student_states : array of shape ([series,] timesteps, student_features) or list of arrays
        States of the student.
    expert_states : array of shape ([series,] timesteps, expert_features) or list of arrays
        States of the expert at the same timesteps of the same input.
    ridge : float, default to 1e-6
        Regularization of the map between states.
    targets : array of shape ([series,] timesteps, output_dim) or list of arrays, optional
        Target outputs for the student states. If given, the readout is refined.
    prior_ridge : float, default to 1.0
        Strength of the pull towards the transplanted weights during
        refinement. Very large: the transplant is kept as it is. Close to
        zero: the result is a readout fitted on the student from scratch.

    target_states : array of shape ([series,] timesteps, student_features) or list of arrays, optional
        Student states that ``targets`` belong to, when they are not
        ``student_states``. Typical use: many paired timesteps without
        targets for the map, and a few timesteps with targets to refine.

    Returns
    -------
    Node
        A new readout of the same class as ``readout``, taking student states.

    Notes
    -----
    The map between states has ``student_features * expert_features``
    unknowns. It needs more paired timesteps than the student has features,
    or a larger ``ridge``, or states reduced to a few components first. What
    the transplant saves is labels, not recordings: the paired states need no
    target. With few paired timesteps a transplanted readout is no better
    than one fitted on the student from scratch.
    """
    if not (hasattr(readout, "Wout") and hasattr(readout, "bias")):
        raise TypeError(f"{type(readout).__name__} is not a linear readout: it has no 'Wout' and 'bias'.")
    if not getattr(readout, "initialized", False) or callable(readout.Wout):
        raise ValueError("The readout is not fitted.")
    expert_weights = np.asarray(readout.Wout, dtype=float)
    expert_bias = np.asarray(readout.bias, dtype=float)
    student_states, expert_states = _stack(student_states, "student_states"), _stack(expert_states, "expert_states")
    if expert_states.shape[1] != expert_weights.shape[0]:
        raise ValueError(
            f"The readout takes {expert_weights.shape[0]} features, 'expert_states' has {expert_states.shape[1]}."
        )

    matrix, offset = align_states(student_states, expert_states, ridge=ridge)
    weights = matrix @ expert_weights
    bias = offset @ expert_weights + expert_bias

    if target_states is not None and targets is None:
        raise ValueError("'target_states' was given without 'targets'.")
    if targets is not None:
        targets = _stack(targets, "targets")
        labelled, name = student_states, "student_states"
        if target_states is not None:
            labelled, name = _stack(target_states, "target_states"), "target_states"
        if len(targets) != len(labelled):
            raise ValueError(
                f"'targets' must have one row per timestep of '{name}', got {len(targets)} and {len(labelled)}."
            )
        # ridge regression on what the transplant leaves unexplained
        residual = targets - (labelled @ weights + bias)
        correction, intercept = _ridge(labelled, residual, prior_ridge)
        weights, bias = weights + correction, bias + intercept

    student = copy.copy(readout)
    like = type(readout.Wout)
    as_backend = (lambda a: a) if like is np.ndarray else _to_backend(readout.Wout)
    student.Wout, student.bias = as_backend(weights), as_backend(bias)
    student.input_dim = weights.shape[0]
    student.state = {"out": as_backend(np.zeros(weights.shape[1]))}
    return student


def _to_backend(example):
    """Converter from NumPy arrays to the array type and precision of ``example``."""
    import jax.numpy as jnp

    return lambda a: jnp.asarray(a, dtype=example.dtype)
