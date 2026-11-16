# Sparse or dense recurrent weights in the Jax backend

Timings behind `DENSE_ABOVE_DENSITY` (per device) and `DENSE_BELOW_SIZE` in
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
| `cpu_apple_m5_max.jsonl` | Apple M5 Max, CPU, Jax 0.12.0.1 | `crossover.py`, 2000 timesteps, 100 to 3000 units. The test suite ran on the same machine during part of the 3000-unit rows. |
| `gpu_nvidia_l4.jsonl` | NVIDIA L4 on a cloud instance, Jax with CUDA 12 | `crossover.py`. Rows recovered from the run log. NumPy rows are the instance's CPU. **Do not use its 10 000-unit rows**: all sizes ran in one process, the GPU was nearly full by then (the run ended with an out-of-memory error), and the dense timings there are 2.5 times those of the clean run below. |
| `gpu_nvidia_l4_large.jsonl` | Same GPU type | `large_point.py`, one process per point, 2000 timesteps, 3000, 6000 and 10 000 units. The matrix is generated sparse, then stored as `BCOO` or dense. |

```sh
python large_point.py 10000 0.05 2000     # units, density, timesteps
```

Thresholds chosen from these files: 2% on a CPU, 5% on a GPU.

Not measured: `fit`, whole models, other GPUs, CPUs other than Apple Silicon,
densities between 5% and 10% on the GPU.
