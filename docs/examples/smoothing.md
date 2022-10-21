---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Smooooooothing

There are many ways to skin the cat, and there are even more options to use smoothing for ODE filtering.
Here is how.


```python
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffeqzoo import backend, ivps
from jax.config import config

from odefilter import ivpsolve, recipes

config.update("jax_enable_x64", True)
backend.select("jax")
```

```python
f, u0, (t0, t1), f_args = ivps.lotka_volterra(time_span=(0.0, 10.0))


@jax.jit
def vf(*ys, t, p):
    return f(*ys, *p)


num_derivatives = 2
```

## Terminal-value simulation

If you are interested in the terminal value of the ODE solution, you can use filters and smoothers interchangeably.
But be aware that a smoother computes more intermediate values than a filter, so filters are more efficient.

```python
ek0, info_op = recipes.ekf0_isotropic_dynamic(num_derivatives=num_derivatives)
ek0sol = ivpsolve.simulate_terminal_values(
    vf,
    initial_values=(u0,),
    t0=t0,
    t1=t1,
    solver=ek0,
    info_op=info_op,
    parameters=f_args,
)
print(ek0sol.t, ek0sol.u)
```

## Traditional simulation

If you are used to calling traditional solve() methods, use one of the conventional smoothers (i.e. not the fixed-point smoothers).

```python
ek0, info_op = recipes.eks0_isotropic_dynamic(num_derivatives=num_derivatives)
ek0sol = ivpsolve.solve(
    vf,
    initial_values=(u0,),
    t0=t0,
    t1=t1,
    solver=ek0,
    info_op=info_op,
    parameters=f_args,
)

plt.subplots(figsize=(5, 3))
plt.title("EKS0 solution")
plt.plot(ek0sol.t, ek0sol.u, "o-")
plt.show()
```

If you like, compute the solution on a dense grid after solving.

```python
ts_dense = jnp.linspace(
    t0 + 1e-4, t1 - 1e-4, num=500, endpoint=True
)  # must be off-grid
dense, _ = ek0.offgrid_marginals_searchsorted(ts=ts_dense, solution=ek0sol)

ts_coarse = jnp.linspace(
    t0 + 1e-4, t1 - 1e-4, num=25, endpoint=True
)  # must be off-grid
coarse, _ = ek0.offgrid_marginals_searchsorted(ts=ts_coarse, solution=ek0sol)

fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(8, 3))


ax1.set_title("EKS0 solution (dense)")
ax1.plot(ts_dense, dense, ".")

ax2.set_title("EKS0 solution (coarse)")
ax2.plot(ts_coarse, coarse, ".")
plt.show()
```

## Checkpoint simulation

If you know in advance that you like to have the solution at a pre-specified set of points only,
use the simulate_checkpoints function together with a fixed-point smoother.

```python
fixedpoint_ek0, info_op = recipes.eks0_isotropic_dynamic_fixedpoint(
    num_derivatives=num_derivatives
)
fixedpointsol = ivpsolve.simulate_checkpoints(
    vf,
    initial_values=(u0,),
    ts=ts_dense,  # reuse from above
    solver=fixedpoint_ek0,
    info_op=info_op,
    parameters=f_args,
)

plt.subplots(figsize=(5, 3))
plt.title("FixedPt-EKS0 solution")
plt.plot(fixedpointsol.t, fixedpointsol.u, ".-")
plt.show()
```
