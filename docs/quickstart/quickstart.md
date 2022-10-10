# Quickstart

```python
from jax.config import config
import jax.numpy as jnp
import matplotlib.pyplot as plt

from diffeqzoo import ivps, backend

from odefilter import ivpsolve, recipes

config.update("jax_enable_x64", True)
backend.select("jax")

f, u0, (t0, t1), f_args = ivps.lotka_volterra()

ekf0 = recipes.dynamic_isotropic_ekf0(
    num_derivatives=1, atol=1e-1, rtol=1e-1
)

ts = jnp.linspace(t0, t1, num=1000, endpoint=True)
solution = ivpsolve.simulate_checkpoints(
    vector_field=lambda y, t, *p: f(y, *p),
    initial_values=(u0,),
    ts = ts,
    solver=ekf0,
    parameters=f_args
)

i = 1
ms = solution.posterior.filtered.mean[:, i, :]
ls = solution.posterior.filtered.cov_sqrtm_lower
stds = jnp.sqrt(jnp.einsum("nkj,nkj->nk", ls, ls))
print(ms, stds)

plt.plot(solution.t, ms[:, 0])
plt.fill_between(solution.t, ms[:, 0] - stds[:, i], ms[:, 0] + stds[:, i], alpha=0.5)
plt.show()

```
