.. _jax_backend_differences:

=================================================
Jax backend: differences with the NumPy backend
=================================================

:mod:`reservoirpy.jax` mirrors the API of :mod:`reservoirpy`: most programs switch backend by changing their imports
(see :doc:`jax_backend`). The two backends are tested against each other, and with the same seed they build the same
weight matrices and produce the same outputs, up to single precision.

They still differ in a few ways that are easy to miss. This page lists them. Each point below is checked by the test
suite (``reservoirpy/jax/tests/test_backend_differences.py`` unless stated otherwise).

.. contents::
    :local:
    :depth: 1

.. _jax_precision:

Single precision by default
===========================

Jax computes in 32-bit floating point unless told otherwise, whatever the ``dtype`` argument of the nodes says.
The NumPy backend computes in 64-bit.

This rarely matters when running a reservoir. It matters when fitting a readout on features that are
nearly collinear or have very different scales, with a small ``ridge``: the regression is ill-conditioned, and single
precision is not enough to solve it. No error or warning is raised.

For example, fitting a :py:class:`~reservoirpy.jax.nodes.Ridge` (``ridge=2.5e-6``) on the
:py:class:`~reservoirpy.nodes.NVAR` features of a Lorenz timeseries, whose values reach about 2000, gives readout
weights that differ from the NumPy ones by 4.4, for weights of order 1. In 64-bit mode the difference is
:math:`7 \cdot 10^{-7}`.

To compute in 64-bit, enable it **before any Jax computation**, at the top of your program:

.. code-block:: python

    import jax
    jax.config.update("jax_enable_x64", True)

or set the environment variable ``JAX_ENABLE_X64=1``. See `Double (64bit) precision
<https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision>`__ in the Jax
documentation. 64-bit computation uses twice the memory and can be slower.

If you stay in single precision, prefer a larger ``ridge``, scale your inputs, and compare a fit with the NumPy
backend once.

.. _jax_speed:

Speed depends on the connectivity of the reservoir
==================================================

The benchmark of :doc:`jax_backend` uses reservoirs with five connections per neuron (``rc_connectivity=5/units``).
With the default ``rc_connectivity=0.1``, the picture is different.

A Jax reservoir stores ``W`` as a sparse :py:class:`jax.experimental.sparse.BCOO` array, as the NumPy backend
stores it as a SciPy sparse array. Jax sparse products are slow when the matrix is not very sparse. For such
reservoirs, ask for a dense matrix:

.. code-block:: python

    import reservoirpy.jax as rjx

    reservoir = rjx.nodes.Reservoir(1000, W=rjx.mat_gen.normal(sparsity_type="dense"))

The matrix has the same values, and the reservoir gives the same output.

Time to run a reservoir on 5000 timesteps, in seconds, after compilation (Apple M5 Max, CPU, Jax 0.12; your numbers
will differ):

.. list-table::
    :header-rows: 1

    * - Units
      - ``rc_connectivity``
      - NumPy
      - Jax, sparse (default)
      - Jax, dense
    * - 300
      - 0.1
      - 0.12
      - 0.44
      - 0.13
    * - 1000
      - 0.1
      - 0.61
      - 3.2
      - 0.58
    * - 3000
      - 0.1
      - 6.1
      - 26
      - 8.9
    * - 300
      - 5 / units
      - 0.07
      - 0.15
      - 0.13
    * - 1000
      - 5 / units
      - 0.13
      - 0.25
      - 0.56
    * - 3000
      - 5 / units
      - 0.28
      - 0.62
      - 8.5

These timings only cover :py:meth:`~reservoirpy.jax.Node.run` on a CPU. The benchmark of :doc:`jax_backend` times
``fit`` and ``run`` of a whole model together, and a GPU changes the comparison again; neither was measured here. Time
your own model before choosing.

The first call of a Jax node also includes its compilation. Do not time it.

.. _jax_numpy_only_nodes:

Nodes without a Jax version
===========================

These nodes only exist in :mod:`reservoirpy.nodes`:

- :py:class:`~reservoirpy.nodes.ScikitLearnNode`, since scikit-learn computes with NumPy;
- :py:class:`~reservoirpy.nodes.LocalPlasticityReservoir`;
- :py:class:`~reservoirpy.nodes.HAGReservoir`.

NumPy nodes can be connected to Jax nodes in a model: :doc:`jax_backend` has an example with
:py:class:`~reservoirpy.nodes.ScikitLearnNode`.

:mod:`reservoirpy.observables`, :mod:`reservoirpy.datasets` and :mod:`reservoirpy.hyper` have no Jax version either.
The functions of :mod:`reservoirpy.observables` accept Jax models, Jax arrays and the weight matrices of Jax nodes
(``reservoirpy/jax/tests/test_observables.py``).

.. _jax_noise:

Noise
=====

:py:class:`reservoirpy.jax.nodes.Reservoir` accepts ``noise_rc`` and ``noise_in`` like
:py:class:`reservoirpy.nodes.Reservoir`, with these differences:

- ``noise_type`` is ``"normal"`` or ``"uniform"``, and there is no ``noise_kwargs``;
- the random key is part of the node state, under ``"noise_key"``. Code that builds a state by hand, for instance to
  call ``_step`` or to compute a Jacobian, must provide it. A reservoir without noise has no such key;
- with the same seed, the two backends draw different noise values from the same distribution;
- inside a :py:class:`~reservoirpy.jax.Model` run on several timeseries at once, all timeseries receive the same
  noise. A reservoir run on its own gives each timeseries its own noise.

These points are checked in ``reservoirpy/jax/nodes/tests/test_reservoir_noise.py``.

.. _jax_saving:

Saving and loading
==================

:py:func:`reservoirpy.save` and :py:func:`reservoirpy.load` work with Jax nodes and models
(``reservoirpy/jax/tests/test_save_load.py``). A node built with your own activation function can only be saved if
that function can be pickled: define it at the top level of a module, not as a ``lambda``, and wrap the functions of
:py:mod:`jax.nn` rather than passing them directly. The functions of :mod:`reservoirpy.jax.activationsfunc` can be
pickled.
