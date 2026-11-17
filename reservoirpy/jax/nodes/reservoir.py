# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

from functools import partial
from typing import Callable, Optional, Sequence, Union

import jax
import jax.numpy as jnp
from numpy.random import Generator

from ...utils.data_validation import check_node_input
from ..activationsfunc import get_function, tanh
from ..mat_gen import bernoulli, normal
from ..node import Node
from ..type import NodeInput, State, Timestep, Weights, is_array, is_multiseries
from ..utils import rand_generator
from ._noise import NoisyNodeMixin


class Reservoir(NoisyNodeMixin, Node):
    """Pool of leaky-integrator neurons with random recurrent connexions.

    Reservoir neurons states, gathered in a vector :math:`\\mathbf{x}`, follow
    the update rule below:

    .. math::

        \\mathbf{x}[t+1] = (1 - \\mathrm{lr}) * \\mathbf{x}[t] + \\mathrm{lr}
         * f(\\mathbf{W}_{in} \\cdot (\\mathbf{u}[t+1] + c_{in} * \\xi)
          + \\mathbf{W} \\cdot \\mathbf{x}[t] + \\mathbf{b}) + c_{rc} * \\xi

    where:
        - :math:`\\mathbf{x}` is the output activation vector of the reservoir;
        - :math:`\\mathbf{u}` is the input timeseries;
        - :math:`f` is the activation function;
        - :math:`\\xi` is a random noise, drawn again at each timestep and for
          each term;
        - :math:`c_{in}` and :math:`c_{rc}` are the noise gains ``noise_in``
          and ``noise_rc``. Both are 0 by default: the reservoir is then
          deterministic.


    Parameters
    ----------
    units : int, optional
        Number of reservoir units. If None, the number of units will be inferred from
        the ``W`` matrix shape.
    lr : float or array-like of shape (units,), default to 1.0
        Neurons leak rate. Must be in :math:`[0, 1]`.
    sr : float, optional
        Spectral radius of recurrent weight matrix.
    input_scaling : float or array-like of shape (features,), default to 1.0.
        Input gain. An array of the same dimension as the inputs can be used to
        set up different input scaling for each feature.
    input_connectivity : float, default to 0.1
        Connectivity of input neurons, i.e. ratio of input neurons connected
        to reservoir neurons. Must be in :math:`]0, 1]`.
    rc_connectivity : float, default to 0.1
        Connectivity of recurrent weight matrix, i.e. ratio of reservoir
        neurons connected to other reservoir neurons, including themselves.
        Must be in :math:`]0, 1]`.
    Win : callable or array-like of shape (units, features), default to :py:func:`~reservoirpy.mat_gen.bernoulli`
        Input weights matrix or initializer. If a callable (like a function) is used,
        then this function should accept any keywords
        parameters and at least two parameters that will be used to define the shape of
        the returned weight matrix.
    W : callable or array-like of shape (units, units), default to :py:func:`~reservoirpy.mat_gen.normal`
        Recurrent weights matrix or initializer. If a callable (like a function) is
        used, then this function should accept any keywords
        parameters and at least two parameters that will be used to define the shape of
        the returned weight matrix.
    bias : callable or array-like of shape (units, 1), default to :py:func:`~reservoirpy.mat_gen.bernoulli`
        Bias weights vector or initializer. If a callable (like a function) is
        used, then this function should accept any keywords
        parameters and at least two parameters that will be used to define the shape of
        the returned weight matrix.
    activation : str or callable, default to :py:func:`~reservoirpy.activationsfunc.tanh`
        Reservoir units activation function.
        - If a str, should be a :py:mod:`~reservoirpy.activationsfunc`
        function name.
        - If a callable, should be an element-wise operator on arrays.
    noise_rc : float, default to 0.0
        Gain of the noise added to the state of the reservoir at each timestep.
    noise_in : float, default to 0.0
        Gain of the noise added to the input of the reservoir at each timestep,
        before the input weights are applied.
    noise_type : {"normal", "uniform"}, default to "normal"
        Distribution of the noise: standard normal, or uniform on
        :math:`[-1, 1]`.
    input_dim : int, optional
        Input dimension. Can be inferred at first call.
    dtype : Numpy dtype, default to jnp.float64
        Numerical type for node parameters.
    seed : int or :py:class:`numpy.random.Generator`, optional
        A random state seed, for weights initialization and noise generation.
    name : str, optional
        Node name.

    Note
    ----

    If W, Win or bias are initialized with an array-like matrix, then all
    initializers parameters such as spectral radius (``sr``) or input scaling
    (``input_scaling``) are ignored.
    See :py:mod:`~reservoirpy.mat_gen` for more information.

    Note
    ----

    When a noise gain is not 0, the state of the node holds a Jax random key
    under ``"noise_key"``, next to ``"out"``. The key advances at each
    timestep and is kept by :py:meth:`reset`, so two runs on the same data
    give different results, as with the NumPy backend. Build two reservoirs
    with the same ``seed`` to get the same noise twice. The noise values
    differ from those of the NumPy backend for the same seed; their
    distribution is the same.

    When a noisy reservoir is run on several timeseries inside a
    :py:class:`~reservoirpy.jax.Model`, all timeseries start from the same key
    and receive the same noise. Run the timeseries one at a time, or run the
    reservoir on its own, to get independent noise.

    Example
    -------

    >>> from reservoirpy.nodes import Reservoir
    >>> reservoir = Reservoir(100, lr=0.2, sr=0.8) # a 100 neurons reservoir

    Using the :py:func:`~reservoirpy.datasets.mackey_glass` timeseries:

    >>> from reservoirpy.datasets import mackey_glass
    >>> x = mackey_glass(200)
    >>> states = reservoir.run(x)

    .. plot::

        from reservoirpy.nodes import Reservoir
        reservoir = Reservoir(100, lr=0.2, sr=0.8)
        from reservoirpy.datasets import mackey_glass
        x = mackey_glass(200)
        states = reservoir.run(x)
        fig, ax = plt.subplots(6, 1, figsize=(7, 10), sharex=True)
        ax[0].plot(x)
        ax[0].grid()
        ax[0].set_title("Neuron states (on Mackey-Glass)")
        for i in range(1, 6):
            ax[i].plot(states[:, i], label=f"Neuron {i}")
            ax[i].legend()
            ax[i].grid()
        ax[-1].set_xlabel("Timesteps")
    """

    initialized: bool
    input_dim: Optional[int]
    output_dim: int
    name: Optional[str]
    state: State

    # params
    #: Number of neuronal units in the reservoir.
    units: int
    #: Leaking rate (1.0 by default) (:math:`\mathrm{lr}`).
    lr: Union[float, jax.Array]
    #: Spectral radius of ``W`` (optional).
    sr: float
    #: Input scaling (float or array) (1.0 by default).
    input_scaling: Union[float, Sequence]
    #: Connectivity (or density) of ``Win`` (0.1 by default).
    input_connectivity: float
    #: Connectivity (or density) of ``W`` (0.1 by default).
    rc_connectivity: float
    #: Input weights matrix (:math:`\mathbf{W}_{in}`).
    Win: Union["Weights", Callable]
    #: Recurrent weights matrix (:math:`\mathbf{W}`).
    W: Union["Weights", Callable]
    #: Bias vector (:math:`\mathbf{b}`).
    bias: Union["Weights", Callable, float]
    #: Type of matrices elements. By default, ``jnp.float64``.
    dtype: type
    #: Activation of the reservoir units (tanh by default) (:math:`f`).
    activation: Callable
    #: Gain of the noise added to the reservoir state (0.0 by default).
    noise_rc: float
    #: Gain of the noise added to the reservoir input (0.0 by default).
    noise_in: float
    #: Distribution of the noise ("normal" by default).
    noise_type: str
    #: A random state generator. Used for generating Win, W and the noise key.
    rng: Generator

    def __init__(
        self,
        units: Optional[int] = None,
        lr: Union[float, jax.Array] = 1.0,
        sr: float = 1.0,
        input_scaling: Union[float, Sequence] = 1.0,
        input_connectivity: float = 0.1,
        rc_connectivity: float = 0.1,
        Win: Union["Weights", Callable] = bernoulli,
        W: Union["Weights", Callable] = normal,
        bias: Union["Weights", Callable, float] = 0.0,
        activation: Union[str, Callable] = tanh,
        noise_rc: float = 0.0,
        noise_in: float = 0.0,
        noise_type: str = "normal",
        input_dim: Optional[int] = None,
        dtype: type = jnp.float64,
        seed: Optional[Union[int, Generator]] = None,
        name: Optional[str] = None,
    ):

        self.lr = lr
        self.sr = sr
        self.input_scaling = input_scaling
        self.input_connectivity = input_connectivity
        self.rc_connectivity = rc_connectivity
        self.Win = Win
        self.W = W
        self.bias = bias
        self.activation = get_function(activation)
        self.dtype = dtype
        self.rng = rand_generator(seed=seed)
        self.name = name

        self._set_noise(noise_rc, noise_in, noise_type)

        # set units / output_dim
        if units is None and not is_array(W):
            raise ValueError("'units' parameter must not be None if 'W' parameter is not a matrix.")
        if units is not None and is_array(W) and W.shape[-1] != units:
            raise ValueError(
                f"Both 'units' and 'W' are set but their dimensions doesn't match: " f"{units} != {W.shape[-1]}."
            )
        self.units = units if units is not None else W.shape[-1]
        self.output_dim = self.units

        # set input_dim (if possible)
        if input_dim is not None and is_array(Win) and Win.shape[-1] != input_dim:
            raise ValueError(
                f"Both 'input_dim' and 'Win' are set but their dimensions doesn't "
                f"match: {input_dim} != {Win.shape[-1]}."
            )
        self.input_dim = Win.shape[-1] if is_array(Win) else input_dim

    def initialize(self, x: Optional[Union[NodeInput, Timestep]]):

        # set input_dim
        self._set_input_dim(x)

        [Win_rng, W_rng, bias_rng] = self.rng.spawn(3)

        if callable(self.Win):
            self.Win = self.Win(
                self.units,
                self.input_dim,
                input_scaling=self.input_scaling,
                connectivity=self.input_connectivity,
                dtype=self.dtype,
                seed=Win_rng,
            )

        if callable(self.W):
            self.W = self.W(
                self.units,
                self.units,
                sr=self.sr,
                connectivity=self.rc_connectivity,
                dtype=self.dtype,
                seed=W_rng,
            )

        if callable(self.bias):
            self.bias = self.bias(
                self.units,
                connectivity=1.0,
                dtype=self.dtype,
                seed=bias_rng,
            )

        self.state = {"out": jnp.zeros((self.units,))}

        # spawned after the weights generators, so that enabling noise does
        # not change the weights obtained from a given seed
        self._init_noise_key()

        self.initialized = True

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, state: State, x: Timestep) -> State:
        W = self.W  # NxN
        Win = self.Win  # NxI
        bias = self.bias  # N or float
        f = self.activation
        lr = self.lr
        s = state["out"]

        # The noise gains are Python floats: a noiseless reservoir is traced
        # without any random number generation.
        if not self._is_noisy:
            next_state = f(W @ s + Win @ x + bias)
            next_state = (1 - lr) * s + lr * next_state
            return {"out": next_state}

        key, key_in, key_rc = jax.random.split(state["noise_key"], 3)

        if self.noise_in > 0.0:
            x = x + self._noise(key_in, self.noise_in, x.shape)

        next_state = f(W @ s + Win @ x + bias)
        next_state = (1 - lr) * s + lr * next_state

        if self.noise_rc > 0.0:
            next_state = next_state + self._noise(key_rc, self.noise_rc, next_state.shape)

        return {"out": next_state, "noise_key": key}
