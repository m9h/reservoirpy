# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
"""
============================================================
Nodes as pure functions (:mod:`reservoirpy.jax.functional`)
============================================================

:py:meth:`~reservoirpy.jax.node.Node.run` reads and updates the state stored
in the node. Jax transformations (``jax.jit``, ``jax.vmap``, ``jax.grad``)
need a function whose result depends only on its arguments.
:py:func:`pure_run` gives that function for an initialized node or model.

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

import jax
import jax.numpy as jnp
from jax.experimental import sparse

from .model import Model
from .node import Node
from .type import State, Timeseries

__all__ = ["pure_run"]


def pure_run(node: Node, *param_names: str) -> tuple[Callable, dict, State]:
    """The run of an initialized node or model, as a pure function.

    Parameters
    ----------
    node : :py:class:`~reservoirpy.jax.node.Node` or :py:class:`~reservoirpy.jax.Model`
        An initialized node, or an initialized model with one input node.
        It is never modified, neither here nor by the returned function.
    *param_names : str
        Names of attributes of the node (such as ``"lr"``, ``"W"`` or
        ``"Wout"``) that become arguments of the returned function. All
        others keep the value they have when ``pure_run`` is called.
        For a model, write ``"node.parameter"``, where ``node`` is the name
        of the node or, if it has none, the name of its class:
        ``"Reservoir.W"``, ``"Ridge.Wout"``.

    Returns
    -------
    run : callable
        ``run(params, state, x) -> (final_state, outputs)``, where ``x`` is
        one timeseries of shape ``(timesteps, input_dim)`` and ``outputs``
        has shape ``(timesteps, output_dim)``. It can be passed to
        ``jax.jit``, ``jax.vmap`` and ``jax.grad``. For a model with several
        output nodes, ``outputs`` is a dictionary with one entry per node.
    params : dict
        Current values of the parameters named in ``param_names``.
    state : dict
        Current state of the node. Its ``"out"`` entry is the last output.
        A noisy :py:class:`~reservoirpy.jax.nodes.Reservoir` also carries a
        ``"noise_key"``: two runs from the same state draw the same noise.
        For a model, ``state["nodes"]`` holds the state of each node under
        its name, and ``state["feedback"]`` the contents of the feedback
        connections.

    Raises
    ------
    RuntimeError
        If the node is not initialized.
    AttributeError
        If the node has no attribute with one of the given names.
    ValueError
        If two nodes of a model have no name and the same class: they
        could not be told apart.
    TypeError
        If the model has several input nodes.

    Notes
    -----
    To change the spectral radius, pass ``"W"`` and scale the matrix: the
    spectral radius is only used when the matrix is generated.

    Gradients through a long run of a reservoir close to or beyond the edge
    of chaos grow exponentially with the length of the run.
    """
    if isinstance(node, Model):
        return _pure_model_run(node, param_names)
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


def _labels(model: Model) -> dict:
    """A unique label for each node of the model: its name, or the name of its class."""
    labels = {node: node.name if node.name is not None else type(node).__name__ for node in model.nodes}
    names = list(labels.values())
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(
            f"Several nodes of the model would be called {repeated}. Give them a name: Reservoir(..., name='first')."
        )
    return labels


def _pure_model_run(model: Model, param_names: tuple) -> tuple[Callable, dict, dict]:
    if not model.initialized:
        raise RuntimeError("The model is not initialized: fit it or call model.initialize(x) before pure_run.")
    if len(model.inputs) != 1:
        raise TypeError(f"pure_run takes a Model with one input node, this one has {len(model.inputs)}.")
    labels = _labels(model)
    by_label = {label: node for node, label in labels.items()}

    targets = {}
    for name in param_names:
        label, _, attribute = name.rpartition(".")
        if not label:
            raise ValueError(f"Parameters of a Model are written 'node.parameter', got '{name}'.")
        if label not in by_label:
            raise ValueError(f"The model has no node '{label}'. Its nodes are {sorted(by_label)}.")
        if not hasattr(by_label[label], attribute):
            raise AttributeError(f"{label} has no parameter '{attribute}' ('{name}').")
        targets[name] = (label, attribute)

    names = frozenset(param_names)
    frozen = {label: copy.copy(node) for label, node in by_label.items()}
    order = [labels[node] for node in model.execution_order]
    parents = {labels[node]: [labels[parent] for parent in model.parents[node]] for node in model.nodes}
    edges = [(labels[parent], labels[child]) for (parent, _delay, child) in model.feedback_buffers]
    input_label = labels[model.inputs[0]]
    output_labels = [labels[node] for node in model.outputs]

    params = {name: _as_array(getattr(by_label[label], attribute)) for name, (label, attribute) in targets.items()}
    state = {
        "nodes": {label: dict(node.state) for label, node in by_label.items()},
        "feedback": [jnp.asarray(buffer) for buffer in model.feedback_buffers.values()],
    }

    def run(params: dict, state: dict, x: Timeseries):
        if set(params) != names:
            missing, unknown = sorted(names - set(params)), sorted(set(params) - names)
            raise KeyError(f"params must have the keys {sorted(names)}: missing {missing}, unexpected {unknown}.")
        local = {label: copy.copy(node) for label, node in frozen.items()}
        for name, value in params.items():
            label, attribute = targets[name]
            setattr(local[label], attribute, value)

        def step(carry, x_t):
            buffers, node_states = carry
            new_states = {}
            # same order of concatenation as Model._step: input, parents, feedback
            for label in order:
                inputs = [x_t] if label == input_label else []
                inputs += [new_states[parent]["out"] for parent in parents[label]]
                inputs += [buffer[-1] for (_parent, child), buffer in zip(edges, buffers) if child == label]
                new_states[label] = local[label]._step(node_states[label], jnp.concatenate(inputs, axis=-1))
            new_buffers = [
                jnp.roll(buffer.at[-1].set(new_states[parent]["out"]), 1, axis=0)
                for (parent, _child), buffer in zip(edges, buffers)
            ]
            return (new_buffers, new_states), {label: new_states[label]["out"] for label in output_labels}

        (buffers, node_states), outputs = jax.lax.scan(step, (state["feedback"], state["nodes"]), jnp.asarray(x))
        if len(output_labels) == 1:
            outputs = outputs[output_labels[0]]
        return {"nodes": node_states, "feedback": buffers}, outputs

    return run, params, state


def _as_array(value):
    if isinstance(value, sparse.JAXSparse):
        return value
    return jnp.asarray(value)
