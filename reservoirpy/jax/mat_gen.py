"""
===========================================================
Jax weights initialization (:mod:`reservoirpy.jax.mat_gen`)
===========================================================

Jax versions of the :mod:`reservoirpy.mat_gen` initializers: the matrices are
generated as in :mod:`reservoirpy.mat_gen`, then converted to a dense
:py:class:`jax.Array` or to a sparse
:py:class:`jax.experimental.sparse.BCOO` array. ``random_sparse`` and
``fast_spectral_initialization`` have no Jax version.

**Dense or sparse.** A Jax sparse product is only faster than a dense one when
the matrix is very sparse. Unless you pass ``sparsity_type``, a matrix
generated sparse is stored dense when the fraction of its elements that are
non-zero reaches :py:data:`DENSE_ABOVE_DENSITY` (2% on a CPU, 5% on a GPU), or
when it has at most :py:data:`DENSE_BELOW_SIZE` (100 000) elements. The device
is the one Jax uses when the matrix is generated. Its values are the same
either way. Pass ``sparsity_type="csr"`` to always get a sparse array, and
``sparsity_type="dense"`` to always get a dense one. The thresholds come from
timings of :py:class:`~reservoirpy.jax.nodes.Reservoir` on one CPU and one GPU;
see :doc:`/user_guide/jax_backend_differences`.

.. currentmodule:: reservoirpy.jax.mat_gen

.. autosummary::
   :toctree: generated/

    uniform
    normal
    bernoulli
    ring
    line
    orthogonal
    cluster
    small_world
    zeros
    ones
    JaxInitializer
    dense_is_faster
"""

# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

import jax
import jax.numpy as jnp
import scipy.sparse
from jax.experimental import sparse as jsparse

from .. import mat_gen

#: Density (fraction of non-zero elements) from which a matrix generated
#: sparse is stored dense, unless ``sparsity_type`` is given, for each Jax
#: backend. Other accelerators use the GPU value.
DENSE_ABOVE_DENSITY = {"cpu": 0.02, "gpu": 0.05}
#: Number of elements up to which a matrix generated sparse is stored dense,
#: unless ``sparsity_type`` is given.
DENSE_BELOW_SIZE = 100_000


def dense_is_faster(shape, nnz, backend=None) -> bool:
    """Whether a matrix of this shape and number of non-zero elements is
    expected to be faster stored dense than as a sparse ``BCOO`` array.

    Parameters
    ----------
    shape : tuple of int
        Shape of the matrix.
    nnz : int
        Number of non-zero elements.
    backend : str, optional
        ``"cpu"`` or ``"gpu"``. By default, :py:func:`jax.default_backend`.

    Returns
    -------
    bool
    """
    if backend is None:
        backend = jax.default_backend()
    size = 1
    for dimension in shape:
        size *= dimension
    density = DENSE_ABOVE_DENSITY.get(backend, DENSE_ABOVE_DENSITY["gpu"])
    return size <= DENSE_BELOW_SIZE or nnz >= density * size


class JaxInitializer(mat_gen.Initializer):
    """Initializer for matrix generation for the Jax backend

    This inherits from the regular numpy/scipy Initializer class.
    The only difference is that the resulting array is converted to Jax
    (a regular :py:class:`jax.Array` or a :py:class:`jax.experimental.sparse.BCOO` array).

    A matrix is always generated, and its spectral radius set, in the format
    of the NumPy initializer. When that format is sparse and no
    ``sparsity_type`` was asked for, the storage of the Jax array is chosen
    from the density and size of the matrix: see :mod:`reservoirpy.jax.mat_gen`.
    """

    def __call__(self, *shape, **kwargs):
        storage_requested = "sparsity_type" in kwargs or "sparsity_type" in self._kwargs
        call_result = mat_gen.Initializer.__call__(self, *shape, **kwargs)

        # Not initialized yet, only kwargs set
        if isinstance(call_result, mat_gen.Initializer):
            return call_result
        # Initialized, call_result is a sparse or dense array to convert to Jax
        else:
            if scipy.sparse.issparse(call_result):
                if not storage_requested and dense_is_faster(call_result.shape, call_result.nnz):
                    return jnp.array(call_result.toarray())
                return jsparse.BCOO.from_scipy_sparse(call_result)
            else:
                return jnp.array(call_result)


bernoulli = JaxInitializer(mat_gen._bernoulli)
cluster = JaxInitializer(mat_gen._cluster)
line = JaxInitializer(mat_gen._line)
normal = JaxInitializer(mat_gen._normal)
orthogonal = JaxInitializer(mat_gen._orthogonal)
ring = JaxInitializer(mat_gen._ring)
small_world = JaxInitializer(mat_gen._small_world)
uniform = JaxInitializer(mat_gen._uniform)
zeros = JaxInitializer(mat_gen._zeros)
ones = JaxInitializer(mat_gen._ones)


def __getattr__(attr):
    if attr in ["random_sparse", "fast_spectral_initialization"]:
        raise AttributeError(f"{attr} does not have a Jax implementation. Please use reservoir.mat_gen.{attr} instead.")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {attr}")
