"""Time Reservoir.run for sparse and dense recurrent matrices, NumPy and Jax.

usage: python crossover.py OUT.jsonl [steps]
One JSON line per (units, connectivity, variant). Times are best of 3 calls
after one warm-up call (so Jax compilation is excluded).
"""
import json, sys, time, platform, warnings
warnings.simplefilter("ignore")
import numpy as np
import jax
import reservoirpy as rpy, reservoirpy.jax as rjx
from reservoirpy import mat_gen
from reservoirpy.nodes import Reservoir
from reservoirpy.jax.nodes import Reservoir as JReservoir

out = sys.argv[1]; steps = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
import os
UNITS = [int(u) for u in os.environ.get("SWEEP_UNITS", "100,300,1000,3000").split(",")]
ONLY = os.environ.get("SWEEP_VARIANTS", "").split(",") if os.environ.get("SWEEP_VARIANTS") else None
DENSITY = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
device = str(jax.devices()[0])
X = np.random.default_rng(0).uniform(size=(steps, 1))

def best(f, n=3):
    f(); ts = []
    for _ in range(n):
        a = time.perf_counter(); f(); ts.append(time.perf_counter() - a)
    return min(ts)

def variants(n, c):
    yield "numpy-sparse", lambda: Reservoir(n, rc_connectivity=c, seed=0), False
    yield "numpy-dense", lambda: Reservoir(n, rc_connectivity=c, W=mat_gen.normal(sparsity_type="dense"), seed=0), False
    yield "jax-sparse", lambda: JReservoir(n, rc_connectivity=c, seed=0), True
    yield "jax-dense", lambda: JReservoir(n, rc_connectivity=c, W=rjx.mat_gen.normal(sparsity_type="dense"), seed=0), True

with open(out, "a") as f:
    for n in UNITS:
        for c in sorted(set(DENSITY + [round(5 / n, 5)])):
            if c * n < 1:
                continue
            for name, make, is_jax in variants(n, c):
                if ONLY and name not in ONLY:
                    continue
                node = make(); node.run(X[:5])
                run = (lambda: jax.block_until_ready(node.run(X))) if is_jax else (lambda: node.run(X))
                t = best(run)
                row = dict(units=n, connectivity=c, variant=name, seconds=t, us_per_step=1e6 * t / steps, steps=steps,
                           device=device if is_jax else "cpu", machine=platform.machine(), jax=jax.__version__, rpy=rpy.__version__)
                f.write(json.dumps(row) + "\n"); f.flush()
                print(n, c, name, round(1e6 * t / steps, 1), "us/step", flush=True)
