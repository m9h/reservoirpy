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

Models
======

``pure_run`` also takes a fitted :py:class:`~reservoirpy.jax.Model`, feedback connections included. Parameters are
written ``"node.parameter"`` and the state has one entry per node. A node is called by its name or, if it has none,
by the name of its class:

.. code-block:: python

    from reservoirpy.jax import ESN

    target = jnp.roll(x, 5, axis=0)                     # recall the input five steps back
    esn = ESN(units=100, sr=0.9, ridge=1e-4, seed=0, feedback=True)
    esn.fit(x[:400], target[:400], warmup=50)

    run, params, state = pure_run(esn, "readout.Wout")
    final_state, predictions = run(params, state, x[400:])  # same values as esn.run(x[400:])

    # how much does the output change when the trained weights are disturbed?
    def output_change(scale):
        disturbed = {"readout.Wout": params["readout.Wout"] * scale}
        return jnp.mean((run(disturbed, state, x[400:])[1] - predictions) ** 2)

    changes = jax.vmap(output_change)(jnp.array([1.0, 1.01, 1.1]))

Because the readout feeds back into the reservoir, the disturbance changes the states too, not only the output.

Online training
===============

:py:func:`~reservoirpy.jax.functional.pure_partial_fit` does for ``partial_fit`` what ``pure_run`` does for ``run``.
What training changes (for :py:class:`~reservoirpy.jax.nodes.RLS`: ``Wout``, ``bias`` and the matrix ``P``) is
passed in and returned, and the whole training is one compiled loop.

The example is FORCE learning (Sussillo and Abbott, 2009). The output of the readout is fed back to the reservoir
while the readout is being trained, so that the reservoir is driven by an output that is already nearly right. The
model then generates the target alone. Here three targets are learned at once from the same reservoir:

.. code-block:: python

    from reservoirpy.jax.functional import pure_partial_fit
    from reservoirpy.jax.nodes import RLS

    time = jnp.arange(1500) * 0.1
    targets = jnp.stack([jnp.sin(f * time)[:, None] for f in (1.0, 1.5, 2.0)])
    nothing = jnp.zeros((1500, 0))                      # the model has no input

    generator, force = Reservoir(200, sr=1.2, lr=0.3, seed=1), RLS(alpha=1.0)
    model = (generator >> force) & (generator << force)
    model.initialize(nothing, targets[0])

    fit, learned, state = pure_partial_fit(model)
    train = jax.jit(jax.vmap(fit, in_axes=(None, None, None, 0)))
    learned, state, during = train(learned, state, nothing[:1000], targets[:, :1000])

    # each trained model continues alone, from the state training left it in
    run, _, _ = pure_run(model, "RLS.Wout", "RLS.bias")
    weights = {"RLS.Wout": learned["RLS.Wout"], "RLS.bias": learned["RLS.bias"]}
    _, generated = jax.vmap(run, in_axes=(0, 0, None))(weights, state, nothing[1000:])
    errors = jnp.abs(generated - targets[:, 1000:]).mean(axis=(1, 2))

On a laptop CPU, training a 300-unit model for 3000 steps took 0.13 s this way, 0.34 s with ``partial_fit`` on the
NumPy backend and 1.0 s with ``partial_fit`` on the Jax backend, which runs a Python loop.

What it does not do
===================

- A model must have a single input node, and two nodes of the same class must be given names.
- Offline training (``fit``) is not covered. Fit the node first: its learned parameters (``Wout`` and ``bias`` for
  a readout) can then be exposed like any other.
- ``pure_partial_fit`` takes :py:class:`~reservoirpy.jax.nodes.RLS` and :py:class:`~reservoirpy.jax.nodes.LMS`
  readouts, alone or as the only trainable node of a model.
- A noisy reservoir keeps its random key in the state (``state["noise_key"]``). Two runs from the same state draw the
  same noise; give each run its own key to get independent noise.
