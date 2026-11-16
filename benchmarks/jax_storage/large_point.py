"""One (units, density) point for large reservoirs: Jax sparse and Jax dense
from the same matrix. The matrix is generated sparse (fast spectral radius)
and converted. Run one point per process so GPU memory is released."""
import json, sys, time, warnings
warnings.simplefilter("ignore")
import numpy as np, jax, jax.numpy as jnp
from jax.experimental import sparse as jsparse
from reservoirpy import mat_gen
from reservoirpy.jax.nodes import Reservoir

n, c, steps = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
X = np.random.default_rng(0).uniform(size=(steps, 1))
W = mat_gen.normal(n, n, connectivity=c, sr=0.9, seed=0)           # scipy csr
Win = jnp.asarray(mat_gen.bernoulli(n, 1, connectivity=0.1, seed=1).toarray())

def best(f, k=3):
    f(); ts = []
    for _ in range(k):
        a = time.perf_counter(); f(); ts.append(time.perf_counter() - a)
    return min(ts)

for name, Wj in (("jax-sparse", lambda: jsparse.BCOO.from_scipy_sparse(W)), ("jax-dense", lambda: jnp.asarray(W.toarray()))):
    try:
        node = Reservoir(units=n, W=Wj(), Win=Win); node.run(X[:5])
        t = best(lambda: jax.block_until_ready(node.run(X)))
        row = dict(units=n, connectivity=c, variant=name, us_per_step=1e6 * t / steps, steps=steps, device=str(jax.devices()[0]))
    except Exception as e:
        row = dict(units=n, connectivity=c, variant=name, error=type(e).__name__ + ": " + str(e)[:120])
    print("ROW " + json.dumps(row), flush=True)
    node = None
