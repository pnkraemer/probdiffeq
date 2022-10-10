# Quickstart

```python
from jax.config import config
import jax.numpy as jnp
import matplotlib.pyplot as plt

from diffeqzoo import ivps, backend

from odefilter import ivpsolve, recipes

backend.select("jax")  # ivp examples in jax
config.update("jax_enable_x64", True)  # x64 precision

# Make a problem
f, u0, (t0, t1), f_args = ivps.lotka_volterra()

# High-res plot
ts = jnp.linspace(t0, t1, num=150, endpoint=True)
ekf0 = recipes.dynamic_isotropic_eks0(num_derivatives=1, atol=1e-1, rtol=1e-1)
solution = ivpsolve.simulate_checkpoints(
    vector_field=lambda y, t, *p: f(y, *p),
    initial_values=(u0,),
    ts=ts,
    parameters=f_args,
    solver=ekf0,
)

# Extract the first derivative and the marginal standard deviations
ms = solution.posterior.mean[:, 1, :]
ls = solution.posterior.cov_sqrtm_lower
stds = jnp.sqrt(jnp.einsum("nkj,nkj->nk", ls, ls))[:, 1]

plt.title("$\dot x(t)$ of LV via EKF0")
plt.plot(solution.t, ms[:, 0])
plt.plot(solution.t, ms[:, 1])
plt.fill_between(solution.t, ms[:, 0] - 3* stds, ms[:, 0] + 3*stds, alpha=0.5)
plt.fill_between(solution.t, ms[:, 1] - 3* stds, ms[:, 1] + 3*stds, alpha=0.5)
plt.ylim((-25, 25))
plt.show()
```
