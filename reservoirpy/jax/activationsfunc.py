"""
====================================================================
Jax activation functions (:py:mod:`reservoirpy.jax.activationsfunc`)
====================================================================

Jax versions of the :mod:`reservoirpy.activationsfunc` activation functions
(those of :py:mod:`jax.nn`).

.. currentmodule:: reservoirpy.jax.activationsfunc

.. autosummary::
   :toctree: generated/

    get_function
    identity
    relu
    sigmoid
    softmax
    softplus
    tanh
"""

# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

from typing import Callable, Union

import jax.nn
import jax.numpy as jnp

# The functions below are thin wrappers around their :py:mod:`jax.nn`
# equivalents. They are defined here, rather than imported, so that they can be
# pickled: ``pickle`` looks functions up by module and name, and the objects
# exposed by ``jax.nn`` are not the ones found at their declared location.
# Without this, any node holding an activation function could not be saved.


def tanh(x):
    """Hyperbolic tangent. See :py:func:`jax.numpy.tanh`."""
    return jnp.tanh(x)


def sigmoid(x):
    """Logistic sigmoid. See :py:func:`jax.nn.sigmoid`."""
    return jax.nn.sigmoid(x)


def relu(x):
    """Rectified linear unit. See :py:func:`jax.nn.relu`."""
    return jax.nn.relu(x)


def softplus(x):
    """Softplus. See :py:func:`jax.nn.softplus`."""
    return jax.nn.softplus(x)


def softmax(x, axis=-1):
    """Softmax. See :py:func:`jax.nn.softmax`."""
    return jax.nn.softmax(x, axis=axis)


# jax.nn.identity was introduced in jax v0.6.0.
# To be able to use previous jax versions, we reimplement
# this one.
def identity(x):
    """Identity function."""
    return jnp.asarray(x)


def get_function(name: Union[Callable, str]) -> Callable:
    """Return a Jax activation function from name.

    Parameters
    ----------
    name : str, Callable
        Name of the activation function.
        Can be one of {'softmax', 'softplus',
        'sigmoid', 'tanh', 'identity', 'relu'} or
        their respective short names {'smax', 'sp',
        'sig', 'id', 're'}. If `name` is a Callable,
        simply returns `name`.

    Returns
    -------
    callable
        An activation function.
    """
    if isinstance(name, Callable):
        return name

    index = {
        "softmax": softmax,
        "softplus": softplus,
        "sigmoid": sigmoid,
        "tanh": tanh,
        "identity": identity,
        "relu": relu,
        "smax": softmax,
        "sp": softplus,
        "sig": sigmoid,
        "id": identity,
        "re": relu,
    }

    if not name in index:
        raise ValueError(f"Function name must be one of {[k for k in index.keys()]}")
    else:
        return index[name]
