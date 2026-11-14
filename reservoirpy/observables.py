"""
===========================================================
Metrics and observables (:py:mod:`reservoirpy.observables`)
===========================================================

Metrics and observables for Reservoir Computing:

.. autosummary::
   :toctree: generated/

    spectral_radius
    mse
    rmse
    nrmse
    rsquare
    mae
    memory_capacity
    effective_spectral_radius

Measures of a reservoir that do not depend on a task:

.. autosummary::
   :toctree: generated/

    kernel_rank
    generalization_rank
    effective_rank
    echo_state_distance
"""

# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

from copy import deepcopy
from typing import Literal, Optional, Union

import numpy as np
from scipy import linalg
from scipy.sparse import coo_array, issparse
from scipy.sparse.linalg import eigs

from .type import Weights
from .utils.random import rand_generator


def _as_scipy_or_numpy(W):
    """Convert the matrices of the Jax backend (sparse ``BCOO`` or dense) to
    their SciPy or NumPy equivalent. Other matrices are returned as they are.
    Jax is not imported: a Jax sparse matrix is recognized by its attributes."""
    if issparse(W) or isinstance(W, np.ndarray):
        return W
    if type(W).__name__ == "BCOO" and hasattr(W, "indices") and hasattr(W, "data"):
        indices = np.asarray(W.indices)
        return coo_array((np.asarray(W.data), (indices[:, 0], indices[:, 1])), shape=W.shape).tocsr()
    return np.asarray(W)


def _check_arrays(y_true, y_pred):
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)

    if not y_true_array.shape == y_pred_array.shape:
        raise ValueError(f"Shape mismatch between y_true and y_pred: " f"{y_true_array.shape} != {y_pred_array.shape}")

    return y_true_array, y_pred_array


def spectral_radius(W: Weights, maxiter: Optional[int] = None) -> float:
    """Compute the spectral radius of a matrix `W`.

    Spectral radius is defined as the maximum absolute
    eigenvalue of `W`.

    Parameters
    ----------
    W : array-like (sparse or dense) of shape (N, N)
        Matrix from which the spectral radius will
        be computed.

    maxiter : int, optional
        Maximum number of Arnoldi update iterations allowed.
        By default, is equal to `W.shape[0] * 20`.
        See `Scipy documentation <https://docs.scipy.org/
        doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html>`_
        for more informations.

    Returns
    -------
    float
        Spectral radius of `W`.

    Raises
    ------
    ArpackNoConvergence
        When computing spectral radius on large
        sparse matrices, it is possible that the
        Fortran ARPACK algorithm used to compute
        eigenvalues don't converge towards precise
        values. To avoid this problem, set the `maxiter`
        parameter to an higher value. Be warned that
        this may drastically increase the computation
        time.

    Examples
    --------
    >>> from reservoirpy.observables import spectral_radius
    >>> from reservoirpy.mat_gen import normal
    >>> W = normal(1000, 1000, degree=8)
    >>> print(spectral_radius(W))
    2.886926160973429
    """
    W = _as_scipy_or_numpy(W)

    if issparse(W):
        if maxiter is None:
            maxiter = W.shape[0] * 20

        return max(
            abs(
                eigs(
                    W,
                    k=1,
                    which="LM",
                    maxiter=maxiter,
                    return_eigenvectors=False,
                    v0=np.ones(W.shape[0], W.dtype),
                )
            )
        )

    return max(abs(linalg.eigvals(W)))


def mse(y_true: np.ndarray, y_pred: np.ndarray, dimensionwise: bool = False) -> float:
    """Mean squared error metric:

    .. math::

        \\frac{\\sum_{i=0}^{N-1} (y_i - \\hat{y}_i)^2}{N}

    Parameters
    ----------
    y_true : array-like of shape (N, features)
        Ground truth values.
    y_pred : array-like of shape (N, features)
        Predicted values.
    dimensionwise: boolean, optional
        If True, return a mean squared error for each dimension of the timeseries

    Returns
    -------
    float
        Mean squared error.
    If `dimensionwise` is True, returns a Numpy array of shape $(features, )$.

    Examples
    --------
    >>> from reservoirpy import ESN
    >>> from reservoirpy.datasets import mackey_glass, to_forecasting
    >>> x_train, x_test, y_train, y_test = to_forecasting(mackey_glass(1000), test_size=0.2)
    >>> y_pred = ESN(units=100, sr=1).fit(x_train, y_train).run(x_test)

    >>> from reservoirpy.observables import mse
    >>> print(mse(y_true=y_test, y_pred=y_pred))
    1.0103865448822693e-07
    """
    y_true_array, y_pred_array = _check_arrays(y_true, y_pred)

    if dimensionwise:
        if len(y_true_array.shape) == 3:
            axis = (0, 1)
        else:
            axis = 0
    else:
        axis = None
    return np.mean((y_true_array - y_pred_array) ** 2, axis=axis)


def rmse(y_true: np.ndarray, y_pred: np.ndarray, dimensionwise: bool = False) -> float:
    """Root mean squared error metric:

    .. math::

        \\sqrt{\\frac{\\sum_{i=0}^{N-1} (y_i - \\hat{y}_i)^2}{N}}

    Parameters
    ----------
    y_true : array-like of shape (N, features)
        Ground truth values.
    y_pred : array-like of shape (N, features)
        Predicted values.
    dimensionwise: boolean, optional
        If True, return a mean squared error for each dimension of the timeseries

    Returns
    -------
    float
        Root mean squared error.
    If `dimensionwise` is True, returns a Numpy array of shape $(features, )$.

    Examples
    --------
    >>> from reservoirpy.nodes import Reservoir, Ridge
    >>> model = Reservoir(units=100, sr=1) >> Ridge(ridge=1e-8)

    >>> from reservoirpy.datasets import mackey_glass, to_forecasting
    >>> x_train, x_test, y_train, y_test = to_forecasting(mackey_glass(1000), test_size=0.2)
    >>> y_pred = model.fit(x_train, y_train).run(x_test)

    >>> from reservoirpy.observables import rmse
    >>> print(rmse(y_true=y_test, y_pred=y_pred))
    0.0004845326475756944
    """
    return np.sqrt(mse(y_true, y_pred, dimensionwise=dimensionwise))


def nrmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    norm: Literal["minmax", "var", "mean", "q1q3"] = "minmax",
    norm_value: Optional[float] = None,
    dimensionwise: bool = False,
) -> float:
    """Normalized root mean squared error metric:

    .. math::

       \\frac{1}{\\lambda} * \\sqrt{\\frac{\\sum_{i=0}^{N-1} (y_i - \\hat{y}_i)^2}{N}}

    where :math:`\\lambda` may be:
        - :math:`\\max y - \\min y` (Peak-to-peak amplitude) if ``norm="minmax"``;
        - :math:`\\mathrm{Var}(y)` (variance over time) if ``norm="var"``;
        - :math:`\\mathbb{E}[y]` (mean over time) if ``norm="mean"``;
        - :math:`Q_{3}(y) - Q_{1}(y)` (quartiles) if ``norm="q1q3"``;
        - or any value passed to ``norm_value``.

    Parameters
    ----------
    y_true : array-like of shape (N, features)
        Ground truth values.
    y_pred : array-like of shape (N, features)
        Predicted values.
    norm : {"minmax", "var", "mean", "q1q3"}, default to "minmax"
        Normalization method.
    norm_value : float, optional
        A normalization factor. If set, will override the ``norm`` parameter.
    dimensionwise: boolean, optional
        If True, return a mean squared error for each dimension of the timeseries

    Returns
    -------
    float
        Normalized root mean squared error.
    If `dimensionwise` is True, returns a Numpy array of shape $(features, )$.

    Examples
    --------
    >>> from reservoirpy.nodes import Reservoir, Ridge
    >>> model = Reservoir(units=100, sr=1) >> Ridge(ridge=1e-8)

    >>> from reservoirpy.datasets import mackey_glass, to_forecasting
    >>> x_train, x_test, y_train, y_test = to_forecasting(mackey_glass(1000), test_size=0.2)
    >>> y_pred = model.fit(x_train, y_train).run(x_test)

    >>> from reservoirpy.observables import nrmse
    >>> print(nrmse(y_true=y_test, y_pred=y_pred, norm="var"))
    0.007249877565336089
    """
    error = rmse(y_true, y_pred, dimensionwise=dimensionwise)
    if norm_value is not None:
        return error / norm_value

    else:
        y_true_array, y_pred_array = _check_arrays(y_true, y_pred)

        if dimensionwise:
            if len(y_true_array.shape) == 3:
                axis = (0, 1)
            else:
                axis = 0
        else:
            axis = None

        norms = {
            "minmax": lambda y: np.ptp(y, axis=axis),
            "var": lambda y: y.var(axis=axis),
            "mean": lambda y: y.mean(axis=axis),
            "q1q3": lambda y: np.quantile(y, 0.75, axis=axis) - np.quantile(y, 0.25, axis=axis),
        }

        if norms.get(norm) is None:
            raise ValueError(f"Unknown normalization method. " f"Available methods are {list(norms.keys())}.")
        else:
            return error / norms[norm](y_true_array)


def rsquare(y_true: np.ndarray, y_pred: np.ndarray, dimensionwise: bool = False) -> float:
    """Coefficient of determination :math:`R^2`:

    .. math::

        1 - \\frac{\\sum^{N-1}_{i=0} (y - \\hat{y})^2}
        {\\sum^{N-1}_{i=0} (y - \\bar{y})^2}

    where :math:`\\bar{y}` is the mean value of ground truth.

    Parameters
    ----------
    y_true : array-like of shape (N, features)
        Ground truth values.
    y_pred : array-like of shape (N, features)
        Predicted values.
    dimensionwise: boolean, optional
        If True, return a mean squared error for each dimension of the timeseries

    Returns
    -------
    float
        Coefficient of determination.
    If `dimensionwise` is True, returns a Numpy array of shape $(features, )$.

    Examples
    --------
    >>> from reservoirpy.nodes import Reservoir, Ridge
    >>> model = Reservoir(units=100, sr=1) >> Ridge(ridge=1e-8)

    >>> from reservoirpy.datasets import mackey_glass, to_forecasting
    >>> x_train, x_test, y_train, y_test = to_forecasting(mackey_glass(1000), test_size=0.2)
    >>> y_pred = model.fit(x_train, y_train).run(x_test)

    >>> from reservoirpy.observables import rsquare
    >>> print(rsquare(y_true=y_test, y_pred=y_pred))
    0.9999970566302174
    """
    y_true_array, y_pred_array = _check_arrays(y_true, y_pred)

    if dimensionwise:
        if len(y_true_array.shape) == 3:
            axis = (0, 1)
        else:
            axis = 0
    else:
        axis = None

    d = (y_true_array - y_pred_array) ** 2
    D = (y_true_array - y_true_array.mean(axis=axis)) ** 2
    return 1 - np.sum(d, axis=axis) / np.sum(D, axis=axis)


def mae(y_true: np.ndarray, y_pred: np.ndarray, dimensionwise: bool = False) -> float:
    """Mean absolute error metric:

    .. math::

        \\frac{1}{N} \\sum_{i=0}^{N-1} |y_i - \\hat{y}_i|


    Parameters
    ----------
    y_true : array-like of shape (N, features)
        Ground truth values.
    y_pred : array-like of shape (N, features)
        Predicted values.
    dimensionwise: boolean, optional
        If True, return a mean absolute error for each dimension of the timeseries.

    Returns
    -------
    float
        Mean absolute error. If `dimensionwise` is True, returns a Numpy array of shape $(features, )$.

    Examples
    --------
    >>> from reservoirpy.nodes import Reservoir, Ridge
    >>> model = Reservoir(units=100, sr=1) >> Ridge(ridge=1e-8)

    >>> from reservoirpy.datasets import mackey_glass, to_forecasting
    >>> x_train, x_test, y_train, y_test = to_forecasting(mackey_glass(1000), test_size=0.2)
    >>> y_pred = model.fit(x_train, y_train).run(x_test)

    >>> from reservoirpy.observables import mae
    >>> print(mae(y_true=y_test, y_pred=y_pred))
    0.00025325136638810585
    """
    y_true_array, y_pred_array = _check_arrays(y_true, y_pred)

    if dimensionwise:
        if len(y_true_array.shape) == 3:
            axis = (0, 1)
        else:
            axis = 0
    else:
        axis = None

    error = np.abs(y_true_array - y_pred_array)
    return np.mean(error, axis=axis)


def memory_capacity(
    model: "Model",
    k_max: int,
    as_list: bool = False,
    series: Optional[np.ndarray] = None,
    test_size: Union[int, float] = 0.2,
    seed: Optional[Union[int, np.random.RandomState, np.random.Generator]] = None,
):
    """Memory Capacity of a model

    The Memory Capacity [1]_ (MC) measure is defined as:

    .. math::
        MC = \\sum_{k=1}^{k_{max}} MC_k

    where:

    .. math::
        MC_k = \\rho^2(u(t-k), y_k(t)) = {cov^2[u(t-k), y_k(t)] \\over var(u(t-k)) \\cdot var(y_k(t))}

    This measure is commonly used in reservoir computing to evaluate the ability of a network to
    recall the previous timesteps. By default, the timeseries :math:`u` is an i.i.d. uniform signal in [-0.8, 0.8].

    Parameters
    ----------
    model : :class:`reservoirpy.Model`
        A ReservoirPy model on which the memory capacity is tested.
        Must have only one input and output node.
    k_max : int
        Maximum time lag between input and output.
        A common rule of thumb is to choose `k_max = 2*reservoir.units`.
    as_list: bool, optional, defaults to False
        If True, returns an array in which `out[k]` :math:`= MC_{k+1}`
    series: array of shape (timesteps, 1), optional
        If specified, is used as the timeseries :math:`u`.
    test_size : int or float
        Number of timesteps for the training phase. Can also be specified
        as a float ratio.
    seed : int or :py:class:`numpy.random.Generator`, optional
        Random state seed for reproducibility.

    Returns
    -------
    float, between 0 and `k_max`.
    If `as_list` is set to True, returns an array of shape `(k_max, )`.

    Examples
    --------
    >>> from reservoirpy.nodes import Reservoir, Ridge
    >>> from reservoirpy.observables import memory_capacity
    >>> model = Reservoir(100, sr=1, seed=1) >> Ridge(ridge=1e-4)
    >>> mcs = memory_capacity(model, k_max=50, as_list=True, seed=1)
    >>> print(f"Memory capacity of the model: {np.sum(mcs):.4}")
    Memory capacity of the model: 18.78

    .. plot::

        from reservoirpy.nodes import Reservoir, Ridge
        from reservoirpy.observables import memory_capacity
        model = Reservoir(100, sr=1, seed=1) >> Ridge(ridge=1e-4)
        mcs = memory_capacity(model, k_max=50, as_list=True, seed=1)
        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(1, 51), mcs)
        plt.grid(); plt.xlabel("Time lag (timestep)"); plt.ylabel("$MC_k$")
        plt.show()

    References
    ----------
    .. [1] Jaeger, H. (2001). Short term memory in echo state networks.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    # Task definition
    if series is None:
        rng = rand_generator(seed)
        series = rng.uniform(low=-0.8, high=0.8, size=(10 * k_max, 1))

    if isinstance(test_size, float) and test_size < 1 and test_size >= 0:
        test_len = round(series.shape[0] * test_size)
    elif isinstance(test_size, int):
        test_len = test_size
    else:
        raise ValueError(
            "invalid test_size argument: " "test_size can be an integer or a float " f"in [0, 1[, but is {test_size}."
        )

    # sliding_window_view creates a matrix of the same
    # timeseries with an incremental shift on each column
    dataset = sliding_window_view(series[:, 0], k_max + 1)[:, ::-1]
    X_train = dataset[:-test_len, :1]
    X_test = dataset[-test_len:, :1]
    Y_train = dataset[:-test_len, 1:]
    Y_test = dataset[-test_len:, 1:]
    # Model
    model_clone = deepcopy(model)
    model_clone.fit(X_train, Y_train, warmup=k_max)
    Y_pred = model_clone.run(X_test)

    # u[t-k] - z_k[t] square correlation
    capacities = np.square(
        [np.corrcoef(y_pred, y_test, rowvar=False)[1, 0] for y_pred, y_test in zip(Y_pred.T, Y_test.T)]
    )

    if as_list:
        return capacities
    else:
        return np.sum(capacities)


def effective_spectral_radius(W: np.ndarray, lr: float = 1.0, maxiter: Optional[int] = None):
    """Effective spectral radius

    The effective spectral radius is defined as the maximal singular value of the matrix
    :math:`lr \\cdot W + (1-lr) \\cdot I_{n}`.

    This concept was first introduced by Jaeger & al. [1]_, with an important result on leaky echo
    state networks:

    Supposing:

    #. The activation function is `tanh`
    #. There is no added noise inside the reservoir (`noise_rc = noise_in = 0.0`)
    #. There is no feedback
    #. There is no bias inside the reservoir (`bias = 0`)

    Then, if the effective spectral radius exceeds 1, the ESN does not have the echo state property.

    Parameters
    ----------
    W : array of shape `(units, units)`
        Adjacency matrix of a reservoir
    lr : float
        Leak rate
    maxiter : int, optional
        Maximum number of Arnoldi update iterations allowed.
        By default, is equal to `W.shape[0] * 20`.
        See `Scipy documentation <https://docs.scipy.org/
        doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html>`_
        for more informations.

    Returns
    -------
    float
        Spectral radius of :math:`lr \\cdot W + (1-lr) \\cdot I_{n}`.

    Raises
    ------
    ArpackNoConvergence
        When computing spectral radius on large
        sparse matrices, it is possible that the
        Fortran ARPACK algorithm used to compute
        eigenvalues don't converge towards precise
        values. To avoid this problem, set the `maxiter`
        parameter to an higher value. Be warned that
        this may drastically increase the computation
        time.

    Examples
    --------

    >>> from reservoirpy.observables import spectral_radius, effective_spectral_radius
    >>> from reservoirpy.mat_gen import uniform
    >>> W = uniform(100, 100, sr=0.5, seed=0)
    >>> lr = 0.5
    >>> print(f"{spectral_radius(W)=:.3}")
    spectral_radius(W)=0.5
    >>> print(f"{effective_spectral_radius(W, lr=lr)=:.3}")
    effective_spectral_radius(W, lr=lr)=0.731

    References
    ----------
    .. [1] Jaeger, H., Lukoševičius, M., Popovici, D., & Siewert, U. (2007).
        Optimization and applications of echo state networks with leaky-integrator
        neurons. Neural networks, 20(3), 335-352.
    """
    units = W.shape[0]
    return spectral_radius(W=lr * W + (1 - lr) * np.eye(units), maxiter=maxiter)


def _numerical_rank(matrix: np.ndarray, rtol: Optional[float]) -> int:
    singular_values = np.linalg.svd(np.nan_to_num(np.asarray(matrix, dtype=np.float64)), compute_uv=False)
    if len(singular_values) == 0 or singular_values[0] == 0.0:
        return 0
    if rtol is None:
        # default of numpy.linalg.matrix_rank
        rtol = max(matrix.shape) * np.finfo(np.float64).eps
    return int(np.sum(singular_values > rtol * singular_values[0]))


def _input_dim(node, input_dim: Optional[int]) -> int:
    if getattr(node, "input_dim", None) is not None:
        if input_dim is not None and input_dim != node.input_dim:
            raise ValueError(f"input_dim={input_dim} but the node expects inputs of dimension {node.input_dim}.")
        return node.input_dim
    return 1 if input_dim is None else input_dim


def _final_states(node, streams: np.ndarray) -> np.ndarray:
    """Final state of a copy of `node` for each stream of `streams`, of shape
    (n_streams, timesteps, input_dim). Every stream starts from a null state."""
    clone = deepcopy(node)
    if clone.initialized:
        clone.reset()
    return np.asarray(clone.run(streams))[:, -1, :]


def kernel_rank(
    node: "Node",
    n_streams: Optional[int] = None,
    timesteps: int = 100,
    input_dim: Optional[int] = None,
    series: Optional[np.ndarray] = None,
    rtol: Optional[float] = None,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> int:
    """Kernel rank of a reservoir: how many different input streams it maps
    to linearly independent states.

    The reservoir is driven by ``n_streams`` different input streams, each
    from a null state. Its state at the last timestep of each stream is one
    row of a matrix of shape ``(n_streams, units)``. The kernel rank is the
    rank of this matrix [1]_.

    If the rank is :math:`r`, a linear readout can assign any target values
    to :math:`r` of these streams. A high rank means the reservoir separates
    its inputs well. It does not mean the separation is useful: a chaotic
    reservoir also separates streams that differ only by noise. Compare
    with :py:func:`generalization_rank`; in [1]_, the difference of the two
    is used to predict computational performance.

    By default, input streams are i.i.d. uniform in [-0.8, 0.8], as in
    :py:func:`memory_capacity`.

    Parameters
    ----------
    node : :class:`reservoirpy.Node`
        A reservoir. It is copied: its state and parameters are not modified.
    n_streams : int, optional
        Number of input streams. By default, the number of units of the node.
        The rank cannot exceed it.
    timesteps : int, default to 100
        Length of each stream.
    input_dim : int, optional
        Dimension of the input streams, for a node that was never run and
        does not know it yet. Defaults to 1.
    series : array of shape (n_streams, timesteps, input_dim), optional
        If specified, is used as the input streams.
    rtol : float, optional
        Singular values below ``rtol`` times the largest one are not counted.
        By default, the tolerance of :py:func:`numpy.linalg.matrix_rank`,
        which only removes rounding errors: simulated states are then often
        of full rank. A larger value, such as the relative precision of your
        readout or of your measurements, is more informative.
    seed : int or :py:class:`numpy.random.Generator`, optional
        Random state seed for the input streams.

    Returns
    -------
    int, between 0 and ``min(n_streams, units)``.

    Examples
    --------
    >>> from reservoirpy.nodes import Reservoir
    >>> from reservoirpy.observables import kernel_rank, generalization_rank
    >>> reservoir = Reservoir(100, sr=0.9, seed=0)
    >>> kr = kernel_rank(reservoir, rtol=1e-3, seed=0)
    >>> gr = generalization_rank(reservoir, rtol=1e-3, seed=0)

    References
    ----------
    .. [1] Legenstein, R., & Maass, W. (2007). Edge of chaos and prediction
        of computational performance for neural circuit models. Neural
        Networks, 20(3), 323-334.
    """
    if series is None:
        rng = rand_generator(seed)
        dim = _input_dim(node, input_dim)
        if n_streams is None:
            n_streams = node.output_dim
        series = rng.uniform(low=-0.8, high=0.8, size=(n_streams, timesteps, dim))
    series = np.asarray(series)
    if series.ndim != 3 or series.shape[1] == 0:
        raise ValueError(f"series must be of shape (n_streams, timesteps, input_dim), got {series.shape}.")

    return _numerical_rank(_final_states(node, series), rtol)


def generalization_rank(
    node: "Node",
    n_streams: Optional[int] = None,
    timesteps: int = 100,
    noise: float = 0.05,
    input_dim: Optional[int] = None,
    series: Optional[np.ndarray] = None,
    rtol: Optional[float] = None,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> int:
    """Generalization rank of a reservoir: how many linearly independent
    states it produces from noisy versions of one input stream.

    Computed like :py:func:`kernel_rank`, except that the ``n_streams`` input
    streams are the same stream with independent noise added. In [1]_ this
    rank estimates the VC-dimension of the linear readouts of the reservoir
    on such inputs. **Lower is better**: a reservoir that turns small input
    differences into independent states gives a readout many ways to fit
    noise.

    Parameters
    ----------
    node : :class:`reservoirpy.Node`
        A reservoir. It is copied: its state and parameters are not modified.
    n_streams : int, optional
        Number of noisy versions. By default, the number of units of the node.
    timesteps : int, default to 100
        Length of the stream.
    noise : float, default to 0.05
        Standard deviation of the gaussian noise added to the stream. What
        counts as noise depends on the task: choose the size of the input
        differences that a readout should ignore.
    input_dim : int, optional
        Dimension of the input stream, for a node that was never run and does
        not know it yet. Defaults to 1.
    series : array of shape (timesteps, input_dim), optional
        If specified, is used as the stream to which noise is added. By
        default, i.i.d. uniform in [-0.8, 0.8].
    rtol : float, optional
        Singular values below ``rtol`` times the largest one are not counted.
        See :py:func:`kernel_rank`. Use the same value for both measures.
    seed : int or :py:class:`numpy.random.Generator`, optional
        Random state seed for the stream and the noise.

    Returns
    -------
    int, between 0 and ``min(n_streams, units)``.

    References
    ----------
    .. [1] Legenstein, R., & Maass, W. (2007). Edge of chaos and prediction
        of computational performance for neural circuit models. Neural
        Networks, 20(3), 323-334.
    """
    if noise < 0.0:
        raise ValueError(f"noise must be positive, got {noise}.")
    rng = rand_generator(seed)
    if series is None:
        dim = _input_dim(node, input_dim)
        series = rng.uniform(low=-0.8, high=0.8, size=(timesteps, dim))
    series = np.asarray(series)
    if series.ndim != 2 or series.shape[0] == 0:
        raise ValueError(f"series must be of shape (timesteps, input_dim), got {series.shape}.")
    if n_streams is None:
        n_streams = node.output_dim

    streams = series[None, :, :] + noise * rng.normal(size=(n_streams,) + series.shape)

    return _numerical_rank(_final_states(node, streams), rtol)


def effective_rank(states: np.ndarray, threshold: float = 0.99) -> int:
    """Number of directions that hold most of the variation of a matrix of
    states: the smallest number of singular values whose sum reaches
    ``threshold`` times the sum of all singular values.

    Unlike :py:func:`kernel_rank`, this is computed from states you already
    have, for instance the activity recorded during a task or from a physical
    reservoir, and needs no access to the reservoir itself.

    Parameters
    ----------
    states : array of shape (timesteps, units)
        States of a reservoir. They are not centered.
    threshold : float, default to 0.99
        Fraction of the sum of singular values to reach, in ]0, 1].

    Returns
    -------
    int, between 0 and ``min(timesteps, units)``. 0 if all states are null.

    Examples
    --------
    >>> from reservoirpy.nodes import Reservoir
    >>> from reservoirpy.datasets import mackey_glass
    >>> from reservoirpy.observables import effective_rank
    >>> states = Reservoir(100, sr=0.9, seed=0).run(mackey_glass(500))
    >>> rank = effective_rank(states)
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in ]0, 1], got {threshold}.")
    singular_values = np.linalg.svd(np.nan_to_num(np.asarray(states, dtype=np.float64)), compute_uv=False)
    total = singular_values.sum()
    if total == 0.0:
        return 0
    cumulated = np.cumsum(singular_values) / total
    # (rounding: the last cumulated value may be slightly below 1)
    return int(min(np.searchsorted(cumulated, threshold - 1e-12) + 1, len(singular_values)))


def echo_state_distance(
    node: "Node",
    series: Optional[np.ndarray] = None,
    timesteps: int = 1000,
    n_starts: int = 5,
    scale: float = 1.0,
    input_dim: Optional[int] = None,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> float:
    """Distance left between runs of a reservoir on the same input from
    different initial states.

    A reservoir with the echo state property [1]_ forgets its initial state:
    driven long enough by the same input, it reaches the same state wherever
    it started, and this distance is close to 0. A distance that stays large
    means that the state depends on something other than the input, and that
    a readout trained on one run cannot be trusted on the next. For a
    physical reservoir, the equivalent is the variability between repeated
    trials of the same stimulus.

    The reservoir is run on ``series`` from a null state, then from
    ``n_starts`` random states. The result is the euclidean distance between
    the final state of the first run and that of each other run, averaged.

    Parameters
    ----------
    node : :class:`reservoirpy.Node`
        A reservoir. It is copied: its state and parameters are not modified.
    series : array of shape (timesteps, input_dim), optional
        Input timeseries. By default, i.i.d. uniform in [-0.8, 0.8].
    timesteps : int, default to 1000
        Length of the default timeseries.
    n_starts : int, default to 5
        Number of random initial states.
    scale : float, default to 1.0
        Initial states are uniform in [-scale, scale].
    input_dim : int, optional
        Dimension of the input, for a node that was never run and does not
        know it yet. Defaults to 1.
    seed : int or :py:class:`numpy.random.Generator`, optional
        Random state seed for the timeseries and the initial states.

    Returns
    -------
    float, positive.

    Note
    ----
    Only the ``"out"`` entry of the state is randomized. This is a numerical
    check on one input, not a proof of the echo state property.

    Examples
    --------
    >>> from reservoirpy.nodes import Reservoir
    >>> from reservoirpy.observables import echo_state_distance
    >>> d = echo_state_distance(Reservoir(100, sr=0.5, seed=0), seed=0)  # close to 0
    >>> d = echo_state_distance(Reservoir(100, sr=4.0, input_scaling=0.01, seed=0), seed=0)  # not

    References
    ----------
    .. [1] Jaeger, H. (2001). The "echo state" approach to analysing and
        training recurrent neural networks. GMD Report 148.
    """
    rng = rand_generator(seed)
    if series is None:
        series = rng.uniform(low=-0.8, high=0.8, size=(timesteps, _input_dim(node, input_dim)))
    series = np.asarray(series)
    if series.ndim != 2 or series.shape[0] == 0:
        raise ValueError(
            f"series must be of shape (timesteps, input_dim) with at least one timestep, got {series.shape}."
        )

    clone = deepcopy(node)
    if not clone.initialized:
        clone.initialize(series)
    clone.reset()
    null_state = clone.state
    reference = np.asarray(clone.run(series))[-1]

    distances = []
    for _ in range(n_starts):
        start = rng.uniform(low=-scale, high=scale, size=np.shape(null_state["out"]))
        # (the sum keeps the array type of the backend)
        clone.state = {**null_state, "out": null_state["out"] + start}
        final = np.asarray(clone.run(series))[-1]
        distances.append(np.linalg.norm(final - reference))

    return float(np.mean(distances))
