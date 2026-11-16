.. _jax_transformations:

===========================================
Jax backend: ``jit``, ``vmap`` and ``grad``
===========================================

The methods of a node (``run``, ``step``, ``fit``) read and update the state stored in the node. That is convenient,
but the transformations that make Jax worth using need *pure* functions, whose result depends only on their
arguments. :py:func:`reservoirpy.jax.functional.pure_run` returns such a function for any initialized node:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    from reservoirpy.jax.functional import pure_run
    from reservoirpy.jax.nodes import Reservoir

    x = jax.random.uniform(jax.random.PRNGKey(0), (500, 1), minval=-1, maxval=1)
    reservoir = Reservoir(100, sr=0.9, lr=0.5, seed=0)
    reservoir.initialize(x)

    run, params, state = pure_run(reservoir)
    final_state, outputs = run(params, state, x)   # same values as reservoir.run(x)

``state`` is the state of the node when ``pure_run`` was called (``state["out"]`` is its last output) and ``params``
holds the parameters you asked to expose, none here. The node itself is never modified. The code blocks of this page
follow one another, and are run by ``reservoirpy/jax/tests/test_functional.py``.

Many starting states at once
============================

Does the reservoir forget where it started? Run it from eight random states on the same input, in one compiled call,
and look at what is left of the differences. This is the echo state property, measured rather than assumed:

.. code-block:: python

    starts = jax.random.uniform(jax.random.PRNGKey(1), (8, 100), minval=-1, maxval=1)
    states = {**{key: jnp.stack([value] * 8) for key, value in state.items()}, "out": starts}

    run_many = jax.jit(jax.vmap(run, in_axes=(None, 0, None)))
    _, outputs = run_many(params, states, x)        # shape (8, 500, 100)
    spread = outputs[:, -1].std(axis=0).max()       # about 0: the start is forgotten

Many parameter values at once
=============================

Name the attributes that should become arguments. To change the spectral radius, expose ``W`` and scale it: the
spectral radius is only used when the matrix is generated.

.. code-block:: python

    run, params, state = pure_run(reservoir, "W", "lr")

    def spread_for(scale):
        scaled = {"W": params["W"] * scale, "lr": params["lr"]}
        _, outputs = jax.vmap(run, in_axes=(None, 0, None))(scaled, states, x)
        return outputs[:, -1].std(axis=0).max()

    scales = jnp.array([0.5, 1.0, 3.0])             # spectral radius 0.45, 0.9 and 2.7
    spreads = jax.jit(jax.vmap(spread_for))(scales)  # the last one does not forget its start

Gradients
=========

The function can be differentiated with respect to the exposed parameters, the starting state or the input:

.. code-block:: python

    def mean_activity(leak_rate):
        _, outputs = run({"W": params["W"], "lr": leak_rate}, state, x)
        return jnp.mean(outputs**2)

    slope = jax.grad(mean_activity)(0.5)

Gradients through a long run are only well behaved when the reservoir forgets its past. Close to or beyond the edge of
chaos they grow exponentially with the length of the run.

What it does not do
===================

- ``pure_run`` takes a node, not a :py:class:`~reservoirpy.jax.Model`. Chain the functions of the nodes yourself.
- Training (``fit``, ``partial_fit``) is not covered. Fit the node first: its learned parameters (``Wout`` and
  ``bias`` for a readout) can then be exposed like any other.
- A noisy reservoir keeps its random key in the state (``state["noise_key"]``). Two runs from the same state draw the
  same noise; give each run its own key to get independent noise.
