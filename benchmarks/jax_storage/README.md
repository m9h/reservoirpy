# Sparse or dense recurrent weights in the Jax backend

Timings behind `DENSE_ABOVE_DENSITY` and `DENSE_BELOW_SIZE` in
`reservoirpy/jax/mat_gen.py`, and behind the tables of
`docs/source/user_guide/jax_backend_differences.rst`.

`crossover.py` times `Reservoir.run` for four variants (NumPy sparse, NumPy
dense, Jax sparse `BCOO`, Jax dense) over reservoir sizes and densities. Each
time is the best of three calls after one warm-up call, so Jax compilation is
excluded. It was run with ReservoirPy 0.4.3, before the Jax initializers chose
the storage themselves: "jax-sparse" is what the default used to be.

```sh
python crossover.py out.jsonl 2000
SWEEP_UNITS=10000 SWEEP_VARIANTS=jax-sparse,jax-dense python crossover.py out.jsonl 500
```

| File | Machine | Notes |
|---|---|---|
| `cpu_apple_m5_max.jsonl` | Apple M5 Max, CPU, Jax 0.12.0.1 | 2000 timesteps, 100 to 3000 units. The test suite ran on the same machine during part of the 3000-unit rows. |
| `gpu_nvidia_l4.jsonl` | NVIDIA L4 on a cloud instance, Jax with CUDA 12 | Rows recovered from the run log. NumPy rows are the instance's CPU. The 10 000-unit run (500 timesteps, Jax only) stopped with a GPU out-of-memory error after three densities. |

Not measured: `fit`, whole models, other GPUs, reservoirs between 3000 and
10 000 units.
