# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
"""Noise shared by the Jax nodes that have ``noise_rc`` and ``noise_in``.

Jax has no global random state. A noisy node keeps a random key in its state,
under ``"noise_key"``, and advances it at each timestep. A node without noise
has no key and is traced without any random number generation.
"""

from typing import Optional, Sequence

import jax
import jax.numpy as jnp

from ...utils.data_validation import check_node_input
from ..type import NodeInput, State, is_multiseries


class NoisyNodeMixin:
    """For nodes with the attributes ``noise_rc``, ``noise_in`` (Python floats) and ``noise_type``."""

    def _set_noise(self, noise_rc: float, noise_in: float, noise_type: str):
        if noise_rc < 0.0 or noise_in < 0.0:
            raise ValueError(f"Noise gains must be positive, got noise_rc={noise_rc} and noise_in={noise_in}.")
        if noise_type not in ("normal", "uniform"):
            raise ValueError(f"'noise_type' must be 'normal' or 'uniform', got '{noise_type}'.")
        self.noise_rc = float(noise_rc)
        self.noise_in = float(noise_in)
        self.noise_type = noise_type

    @property
    def _is_noisy(self) -> bool:
        return self.noise_rc > 0.0 or self.noise_in > 0.0

    def _noise(self, key, gain: float, shape: tuple):
        if self.noise_type == "uniform":
            return gain * jax.random.uniform(key, shape, minval=-1.0, maxval=1.0)
        return gain * jax.random.normal(key, shape)

    def _init_noise_key(self):
        """To call at the end of ``initialize``, once the weights generators are spawned and the state is set."""
        if self._is_noisy:
            [noise_rng] = self.rng.spawn(1)
            self.state["noise_key"] = jax.random.PRNGKey(int(noise_rng.integers(2**31 - 1)))

    def run(self, x: Optional[NodeInput] = None, iters: Optional[int] = None, workers=1) -> NodeInput:
        if not self._is_noisy or x is None or not is_multiseries(x):
            return super().run(x, iters=iters, workers=workers)

        # Timeseries of a multi-series input all start from the current state.
        # Each one is given its own noise key; without this, they would all
        # receive the same noise.
        check_node_input(x, expected_dim=self.input_dim)
        if not self.initialized:
            self.initialize(x)

        keys = jax.random.split(self.state["noise_key"], len(x) + 1)
        start = {name: value for name, value in self.state.items() if name != "noise_key"}
        if isinstance(x, Sequence):
            outputs = [self._run({**start, "noise_key": key}, series) for key, series in zip(keys, x)]
            final = outputs[-1][0]
            result = [series for _, series in outputs]
        else:
            states, result = jax.lax.map(
                lambda args: self._run({**start, "noise_key": args[0]}, args[1]), (keys[:-1], x)
            )
            final = {name: value[-1] for name, value in states.items()}

        self.state = {**{name: value for name, value in final.items() if name != "noise_key"}, "noise_key": keys[-1]}
        return result

    def reset(self) -> State:
        previous_state = self.state
        self.state = {name: jnp.zeros(value.shape) for name, value in previous_state.items() if name != "noise_key"}
        if "noise_key" in previous_state:
            self.state["noise_key"] = previous_state["noise_key"]
        return previous_state
