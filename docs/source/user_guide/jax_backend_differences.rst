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

Dense or sparse weights, and speed
==================================

The NumPy backend stores the recurrent matrix ``W`` as a SciPy sparse array. The Jax equivalent, a
:py:class:`jax.experimental.sparse.BCOO` array, is only faster than a dense array when the matrix is very sparse. The
Jax initializers therefore choose: a matrix is stored dense when the fraction of its elements that are non-zero
reaches 2% on a CPU or 5% on a GPU, or when it has at most 100 000 elements, and sparse otherwise
(:mod:`reservoirpy.jax.mat_gen`). The device is the one Jax uses when the matrix is generated. The values of the matrix are the
same either way, and it is always generated sparse, so that setting its spectral radius stays cheap.

With the default ``rc_connectivity=0.1`` a Jax reservoir is therefore dense. With the five connections per neuron of
the benchmark in :doc:`jax_backend` (``rc_connectivity=5/units``), it is sparse from 317 units up. To decide yourself:

.. code-block:: python

    import reservoirpy.jax as rjx

    reservoir = rjx.nodes.Reservoir(1000, W=rjx.mat_gen.normal(sparsity_type="csr"))    # always sparse
    reservoir = rjx.nodes.Reservoir(1000, W=rjx.mat_gen.normal(sparsity_type="dense"))  # always dense

Prefer the default to ``sparsity_type="dense"`` for large reservoirs: asking the initializer for a dense matrix also
makes it compute the spectral radius on a dense matrix, which takes minutes at 10 000 units.

The thresholds come from the timings below: microseconds per timestep of :py:meth:`~reservoirpy.jax.Node.run`,
after compilation, on 2000 timesteps. The fastest Jax storage is in bold.

Apple M5 Max, CPU, Jax 0.12:

.. list-table::
    :header-rows: 1

    * - Units
      - Density
      - NumPy (sparse)
      - Jax sparse
      - Jax dense
    * - 300
      - 0.5%
      - 13
      - 52
      - **32**
    * - 300
      - 10%
      - 20
      - 89
      - **48**
    * - 1000
      - 0.5%
      - 20
      - **57**
      - 84
    * - 1000
      - 1%
      - 24
      - 117
      - **89**
    * - 1000
      - 10%
      - 112
      - 652
      - **81**
    * - 3000
      - 1%
      - 113
      - **555**
      - 600
    * - 3000
      - 2%
      - 184
      - 989
      - **739**
    * - 3000
      - 10%
      - 1024
      - 4685
      - **620**

NVIDIA L4 GPU (cloud instance), one process per row:

.. list-table::
    :header-rows: 1

    * - Units
      - Density
      - Jax sparse
      - Jax dense
    * - 3000
      - 1%
      - **127**
      - 278
    * - 3000
      - 3%
      - **245**
      - 292
    * - 3000
      - 5%
      - 370
      - **256**
    * - 3000
      - 10%
      - 425
      - **256**
    * - 6000
      - 3%
      - **409**
      - 1472
    * - 6000
      - 5%
      - **765**
      - 1266
    * - 6000
      - 10%
      - 1808
      - **1328**
    * - 10 000
      - 3%
      - **1238**
      - 3458
    * - 10 000
      - 5%
      - **2479**
      - 3220
    * - 10 000
      - 10%
      - 6240
      - **3473**

What these numbers say, and do not say:

- The crossing point is between 1% and 2% of density on this CPU. On this GPU it is between 3% and 5% at 3000 units
  and between 5% and 10% at 6000 and 10 000 units. Near its threshold the automatic choice can be the slower one: by
  up to 1.3 times at 1% on the CPU, and by 1.7 times at 6000 units and 5% on the GPU.
- Small reservoirs gain nothing from Jax here. On the CPU no Jax step was faster than about 30 microseconds whatever
  the size, and at 300 units the NumPy backend ran the reservoir faster. At 1000 units it depended on the density,
  and at 3000 units and 10% density Jax was faster. On the GPU, smaller reservoirs (100 to 1000 units, in
  ``gpu_nvidia_l4.jsonl``) cost 65 to 85 microseconds per step at every density below 5%.
- These are timings of ``run`` for one reservoir on two machines. The benchmark of :doc:`jax_backend` times ``fit`` and
  ``run`` of a whole model. Time your own model before choosing.
- The scripts and the raw timings are in ``benchmarks/jax_storage`` in the repository.

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
