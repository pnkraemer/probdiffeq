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
f, u0, (t0, t1), f_args = ivps.lotka_volterra(time_span=(0.0, 1e-1))


@jax.jit
def vf(*ys, t, p):
    return f(*ys, *p)


num_derivatives = 3
```

```python
ek0, info_op = recipes.dynamic_isotropic_eks0(num_derivatives=num_derivatives)
```

```python
ek0sol = ivpsolve.solve(
    vf,
    initial_values=(u0,),
    t0=t0,
    t1=t1,
    solver=ek0,
    info_op=info_op,
    parameters=f_args,
)
```

```python
plt.title(len(ek0sol.t))
plt.plot(ek0sol.t, ek0sol.u, "o-")
plt.show()
```

```python
fixedpt_ek0, info_op = recipes.dynamic_isotropic_fixedpt_eks0(
    num_derivatives=num_derivatives
)
```

```python
print()
fixedptsol = ivpsolve.simulate_checkpoints(
    vf,
    initial_values=(u0,),
    ts=ek0sol.t,
    solver=fixedpt_ek0,
    info_op=info_op,
    parameters=f_args,
)
print()

fixedptsol2 = ivpsolve.simulate_checkpoints(
    vf,
    initial_values=(u0,),
    ts=jnp.linspace(t0, t1, num=200, endpoint=True),
    solver=fixedpt_ek0,
    info_op=info_op,
    parameters=f_args,
)
print()
```

```python
plt.title(len(fixedptsol.t))

style = {"linestyle": "None", "marker": "x"}
plt.plot(fixedptsol.t, fixedptsol.u, **style, label="FixPtEKS0(t=EKS0.t)")
plt.plot(ek0sol.t, ek0sol.u, **style, color="red", linewidth=3, label="EKS0")
plt.plot(
    fixedptsol2.t, fixedptsol2.u, **style, color="gray", label="FixPtEKS0(t=dense)"
)
plt.legend()
# plt.ylim((-20, 30))
plt.show()
```

```python
plt.plot(
    fixedptsol.t,
    fixedptsol.marginals.mean[:, -1, :],
    linestyle="None",
    marker="P",
    label="FixPtEKS0(t=EKS0.t)",
)
plt.plot(
    ek0sol.t,
    ek0sol.marginals.mean[:, -1, :],
    linestyle="None",
    marker="o",
    color="red",
    label="EKS0",
)
plt.plot(
    fixedptsol2.t,
    fixedptsol2.marginals.mean[:, -1, :],
    linestyle="None",
    marker="^",
    color="gray",
    label="FixPtEKS0(t=dense)",
)
# plt.ylim((-30, 30))
# plt.xlim((8, 9))
plt.legend()
plt.show()
```

```python

```
