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
generated sparse is stored dense when at least :py:data:`DENSE_ABOVE_DENSITY`
(3%) of its elements are non-zero, or when it has at most
:py:data:`DENSE_BELOW_SIZE` (100 000) elements. Its values are the same either
way. Pass ``sparsity_type="csr"`` to always get a sparse array, and
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
"""

# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

import jax.numpy as jnp
import scipy.sparse
from jax.experimental import sparse as jsparse

from .. import mat_gen

#: Density (fraction of non-zero elements) from which a matrix generated
#: sparse is stored dense, unless ``sparsity_type`` is given.
DENSE_ABOVE_DENSITY = 0.03
#: Number of elements up to which a matrix generated sparse is stored dense,
#: unless ``sparsity_type`` is given.
DENSE_BELOW_SIZE = 100_000


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
                size = call_result.shape[0] * call_result.shape[1]
                dense_is_faster = size <= DENSE_BELOW_SIZE or call_result.nnz >= DENSE_ABOVE_DENSITY * size
                if dense_is_faster and not storage_requested:
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
