# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
"""
============================================================
Nodes as pure functions (:mod:`reservoirpy.jax.functional`)
============================================================

:py:meth:`~reservoirpy.jax.node.Node.run` reads and updates the state stored
in the node. Jax transformations (``jax.jit``, ``jax.vmap``, ``jax.grad``)
need a function whose result depends only on its arguments.
:py:func:`pure_run` gives that function for an initialized node.

.. autosummary::
   :toctree: generated/

    pure_run

Example
=======

Run one reservoir from eight starting states in a single compiled call, to
see whether it forgets where it started:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    from reservoirpy.jax.functional import pure_run
    from reservoirpy.jax.nodes import Reservoir

    x = jax.random.uniform(jax.random.PRNGKey(0), (500, 1))
    reservoir = Reservoir(100, sr=0.9, seed=0)
    reservoir.initialize(x)

    run, params, state = pure_run(reservoir)
    starts = jax.random.uniform(jax.random.PRNGKey(1), (8, 100), minval=-1, maxval=1)
    states = {**{k: jnp.stack([v] * 8) for k, v in state.items()}, "out": starts}
    _, outputs = jax.jit(jax.vmap(run, in_axes=(None, 0, None)))(params, states, x)
    spread = outputs[:, -1].std(axis=0).max()  # close to 0: the start is forgotten
"""

import copy
from typing import Callable

import jax.numpy as jnp
from jax.experimental import sparse

from .model import Model
from .node import Node
from .type import State, Timeseries

__all__ = ["pure_run"]


def pure_run(node: Node, *param_names: str) -> tuple[Callable, dict, State]:
    """The run of an initialized node, as a pure function.

    Parameters
    ----------
    node : :py:class:`~reservoirpy.jax.node.Node`
        An initialized node. It is never modified, neither here nor by the
        returned function.
    *param_names : str
        Names of attributes of the node (such as ``"lr"``, ``"W"`` or
        ``"Wout"``) that become arguments of the returned function. All
        others keep the value they have when ``pure_run`` is called.

    Returns
    -------
    run : callable
        ``run(params, state, x) -> (final_state, outputs)``, where ``x`` is
        one timeseries of shape ``(timesteps, input_dim)`` and ``outputs``
        has shape ``(timesteps, output_dim)``. It can be passed to
        ``jax.jit``, ``jax.vmap`` and ``jax.grad``.
    params : dict
        Current values of the parameters named in ``param_names``.
    state : dict
        Current state of the node. Its ``"out"`` entry is the last output.
        A noisy :py:class:`~reservoirpy.jax.nodes.Reservoir` also carries a
        ``"noise_key"``: two runs from the same state draw the same noise.

    Raises
    ------
    RuntimeError
        If the node is not initialized.
    AttributeError
        If the node has no attribute with one of the given names.
    TypeError
        If ``node`` is a Model. Use the nodes of the model one by one.

    Notes
    -----
    To change the spectral radius, pass ``"W"`` and scale the matrix: the
    spectral radius is only used when the matrix is generated.

    Gradients through a long run of a reservoir close to or beyond the edge
    of chaos grow exponentially with the length of the run.
    """
    if isinstance(node, Model):
        raise TypeError("pure_run takes a Node, not a Model. Call it on the nodes of the model.")
    if not node.initialized:
        raise RuntimeError(f"{type(node).__name__} is not initialized: call node.initialize(x) before pure_run.")
    for name in param_names:
        if not hasattr(node, name):
            raise AttributeError(f"{type(node).__name__} has no parameter '{name}'.")

    names = frozenset(param_names)
    frozen = copy.copy(node)  # later changes to the node do not reach the function
    params = {name: _as_array(getattr(node, name)) for name in param_names}
    state = dict(node.state)

    def run(params: dict, state: State, x: Timeseries) -> tuple[State, Timeseries]:
        if set(params) != names:
            missing, unknown = sorted(names - set(params)), sorted(set(params) - names)
            raise KeyError(f"params must have the keys {sorted(names)}: missing {missing}, unexpected {unknown}.")
        local = copy.copy(frozen)
        for name, value in params.items():
            setattr(local, name, value)
        return local._run(state, jnp.asarray(x))

    return run, params, state


def _as_array(value):
    if isinstance(value, sparse.JAXSparse):
        return value
    return jnp.asarray(value)
