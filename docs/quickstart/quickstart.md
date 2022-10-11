# Quickstart

```python
from jax.config import config
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
from diffeqzoo import ivps, backend

from odefilter import ivpsolve, recipes

backend.select("jax")  # ivp examples in jax
config.update("jax_enable_x64", False)  # x64 precision

# Make a problem
f, u0, (t0, t1), f_args = ivps.lotka_volterra()

# High-res plot
ts = jnp.linspace(t0, .3, num=1_000, endpoint=True)
ekf0 = recipes.dynamic_isotropic_ekf0(num_derivatives=1, atol=1e-1, rtol=1e-1)
eks0 = recipes.dynamic_isotropic_eks0(num_derivatives=1, atol=1e-1, rtol=1e-1)

fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey="row")
for a, ek0, label in zip(ax.T, [ekf0, eks0], ["EKF", "EKS"]):
    solution = ivpsolve.simulate_checkpoints(
        vector_field=lambda y, t, *p: f(y, *p),
        initial_values=(u0,),
        ts=ts,
        parameters=f_args,
        solver=ek0,
    )
    # Extract the first derivative and the marginal standard deviations
    ms = solution.posterior.mean[:, -1, :]
    ls = solution.posterior.cov_sqrtm_lower
    stds = jnp.sqrt(jnp.einsum("nkj,nkj->nk", ls, ls))[:, -1]
    # Plot the usual sausages
    a[0].set_title(fr"$D^\nu x(t)$ of LV via {label}, N={len(solution.t)}")
    a[0].plot(solution.t, ms[:, 0])
    a[0].plot(solution.t, ms[:, 1])
    a[0].fill_between(solution.t, ms[:, 0] - 3* stds, ms[:, 0] + 3*stds, alpha=0.5)
    a[0].fill_between(solution.t, ms[:, 1] - 3* stds, ms[:, 1] + 3*stds, alpha=0.5)
    a[1].semilogy(solution.t, stds)

plt.show()

```
