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
    pure_partial_fit

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
from .node import Node, OnlineNode
from .type import State, Timeseries

__all__ = ["pure_run", "pure_partial_fit"]


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


def _targets(labels: dict, param_names: tuple) -> dict:
    """The node label and attribute behind each name written 'node.parameter'."""
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
    return targets


def _pure_model_run(model: Model, param_names: tuple) -> tuple[Callable, dict, dict]:
    if not model.initialized:
        raise RuntimeError("The model is not initialized: fit it or call model.initialize(x) before pure_run.")
    if len(model.inputs) != 1:
        raise TypeError(f"pure_run takes a Model with one input node, this one has {len(model.inputs)}.")
    labels = _labels(model)
    by_label = {label: node for node, label in labels.items()}
    targets = _targets(labels, param_names)

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


def pure_partial_fit(node: Node, *param_names: str) -> tuple[Callable, dict, State]:
    """The online training of an initialized node or model, as a pure function.

    :py:meth:`~reservoirpy.jax.node.OnlineNode.partial_fit` runs a Python
    loop and stores what it learns in the node. The function returned here
    does the same computation in one ``jax.lax.scan``: it can be compiled,
    and applied to many targets, starting states or networks at once.

    Parameters
    ----------
    node : :py:class:`~reservoirpy.jax.nodes.RLS`, :py:class:`~reservoirpy.jax.nodes.LMS` or :py:class:`~reservoirpy.jax.Model`
        An initialized online node, or an initialized model with one input
        node and one online node, with or without feedback connections.
        It is never modified, neither here nor by the returned function.
    *param_names : str
        For a model, names of parameters that training does not change and
        that become a fifth argument of the returned function,
        ``fit(learned, state, x, y, params)``. They are written as in
        :py:func:`pure_run`, which gives their current values:
        ``pure_run(model, "Reservoir.W")[1]``.

    Returns
    -------
    fit : callable
        ``fit(learned, state, x, y) -> (learned, state, outputs)``, where
        ``x`` has shape ``(timesteps, input_dim)``, ``y`` is the target of
        the online node, of shape ``(timesteps, output_dim)``, and
        ``outputs`` is what the node or model produced while it was learning.
        A model without input takes an ``x`` of shape ``(timesteps, 0)``.
    learned : dict
        What training changes, with its current value: for
        :py:class:`~reservoirpy.jax.nodes.RLS`, ``Wout``, ``bias``, ``P`` and
        ``S``. For a model the names are written ``"node.parameter"``, as
        in :py:func:`pure_run`.
    state : dict
        Current state of the node or model, as in :py:func:`pure_run`.

    Raises
    ------
    RuntimeError
        If the node or model is not initialized.
    TypeError
        If the node is not an online node, or if the model does not have
        exactly one online node, or has another trainable node.

    Example
    -------
    FORCE learning [1]_: the output of the readout is fed back to the
    reservoir while the readout is trained, then the model generates the
    target alone.

    >>> reservoir, readout = Reservoir(300, sr=1.2, lr=0.3), RLS(alpha=1.0)
    >>> model = (reservoir >> readout) & (reservoir << readout)
    >>> model.initialize(x, target)  # x of shape (timesteps, 0)
    >>> fit, learned, state = pure_partial_fit(model)
    >>> learned, state, during = jax.jit(fit)(learned, state, x, target)
    >>> run, _, _ = pure_run(model, "RLS.Wout", "RLS.bias")
    >>> weights = {"RLS.Wout": learned["RLS.Wout"], "RLS.bias": learned["RLS.bias"]}
    >>> state, after = run(weights, state, x)

    References
    ----------
    .. [1] Sussillo, D., & Abbott, L. F. (2009). Generating Coherent Patterns of
           Activity from Chaotic Neural Networks. Neuron, 63(4), 544–557.
           https://doi.org/10.1016/j.neuron.2009.07.018
    """
    if isinstance(node, Model):
        return _pure_model_partial_fit(node, param_names)
    if param_names:
        raise TypeError("Other parameters can only be passed for a Model: an online node has none that matter.")
    _check_online(node)
    if not node.initialized:
        raise RuntimeError(f"{type(node).__name__} is not initialized: call node.initialize(x, y) first.")

    frozen = copy.copy(node)
    learned = {name: jnp.asarray(getattr(node, name), dtype=float) for name in node._learned}
    state = dict(node.state)

    def fit(learned: dict, state: State, x: Timeseries, y: Timeseries):
        def step(learned, xy):
            learned, prediction = _learn(frozen, learned, *xy)
            return learned, prediction

        learned, outputs = jax.lax.scan(step, learned, (jnp.asarray(x), jnp.asarray(y)))
        return learned, {**state, "out": outputs[-1]}, outputs

    return fit, learned, state


def _check_online(node: Node):
    if not isinstance(node, OnlineNode):
        raise TypeError(f"{type(node).__name__} is not an online node: it has no partial_fit to make pure.")
    if not hasattr(node, "_learned"):
        raise TypeError(
            f"{type(node).__name__} does not say what its online training changes: give the class an attribute "
            "'_learned', the names of the attributes that '_learning_step' updates."
        )


def _learn(frozen: OnlineNode, learned: dict, x, y) -> tuple[dict, jax.Array]:
    """One learning step of a copy of the node that holds the values of ``learned``."""
    local = copy.copy(frozen)
    for name, value in learned.items():
        setattr(local, name, value)
    prediction = local._learning_step(x, y)
    return {name: jnp.asarray(getattr(local, name), dtype=float) for name in learned}, prediction


def _pure_model_partial_fit(model: Model, param_names: tuple = ()) -> tuple[Callable, dict, dict]:
    if not model.initialized:
        raise RuntimeError("The model is not initialized: call model.initialize(x, y) before pure_partial_fit.")
    if len(model.inputs) != 1:
        raise TypeError(f"pure_partial_fit takes a Model with one input node, this one has {len(model.inputs)}.")
    online = [node for node in model.trainable_nodes if isinstance(node, OnlineNode)]
    if len(online) != 1 or len(model.trainable_nodes) != 1:
        raise TypeError(
            "pure_partial_fit takes a Model with one online node and no other trainable node, this one has "
            f"{len(online)} online node(s) among {len(model.trainable_nodes)} trainable node(s)."
        )
    [trained] = online
    _check_online(trained)

    labels = _labels(model)
    targets = _targets(labels, param_names)
    names = frozenset(param_names)
    frozen = {label: copy.copy(node) for node, label in labels.items()}
    order = [labels[node] for node in model.execution_order]
    parents = {labels[node]: [labels[parent] for parent in model.parents[node]] for node in model.nodes}
    edges = [(labels[parent], labels[child]) for (parent, _delay, child) in model.feedback_buffers]
    input_label, trained_label = labels[model.inputs[0]], labels[trained]
    output_labels = [labels[node] for node in model.outputs]

    learned = {f"{trained_label}.{name}": jnp.asarray(getattr(trained, name), dtype=float) for name in trained._learned}
    state = {
        "nodes": {label: dict(node.state) for node, label in labels.items()},
        "feedback": [jnp.asarray(buffer) for buffer in model.feedback_buffers.values()],
    }

    def fit(learned: dict, state: dict, x: Timeseries, y: Timeseries, params: dict = {}):
        if set(params) != names:
            missing, unknown = sorted(names - set(params)), sorted(set(params) - names)
            raise KeyError(f"params must have the keys {sorted(names)}: missing {missing}, unexpected {unknown}.")
        local = {label: copy.copy(node) for label, node in frozen.items()}
        for name, value in params.items():
            label, attribute = targets[name]
            setattr(local[label], attribute, value)

        def step(carry, xy):
            (buffers, node_states, values), (x_t, y_t) = carry, xy
            new_states = {}
            # same order of concatenation as Model._step: input, parents, feedback
            for label in order:
                inputs = [x_t] if label == input_label else []
                inputs += [new_states[parent]["out"] for parent in parents[label]]
                inputs += [buffer[-1] for (_parent, child), buffer in zip(edges, buffers) if child == label]
                node_input = jnp.concatenate(inputs, axis=-1)
                if label == trained_label:
                    values, prediction = _learn(local[label], values, node_input, y_t)
                    new_states[label] = {"out": prediction}
                else:
                    new_states[label] = local[label]._step(node_states[label], node_input)
            new_buffers = [
                jnp.roll(buffer.at[-1].set(new_states[parent]["out"]), 1, axis=0)
                for (parent, _child), buffer in zip(edges, buffers)
            ]
            return (new_buffers, new_states, values), {label: new_states[label]["out"] for label in output_labels}

        values = {name.rpartition(".")[2]: value for name, value in learned.items()}
        start = (state["feedback"], state["nodes"], values)
        (buffers, node_states, values), outputs = jax.lax.scan(step, start, (jnp.asarray(x), jnp.asarray(y)))
        if len(output_labels) == 1:
            outputs = outputs[output_labels[0]]
        learned = {f"{trained_label}.{name}": value for name, value in values.items()}
        return learned, {"nodes": node_states, "feedback": buffers}, outputs

    return fit, learned, state


def _as_array(value):
    if isinstance(value, sparse.JAXSparse):
        return value
    return jnp.asarray(value)
