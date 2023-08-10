# An easy example

```python
import jax
import jax.numpy as jnp

from probdiffeq import ivpsolve
from probdiffeq.impl import impl
from probdiffeq.ivpsolvers import uncalibrated
from probdiffeq.statespace import correction, extrapolation
from probdiffeq.strategies import smoothers

# Make a problem


@jax.jit
def vf(y, *, t, p):
    return p * y * (1 - y)


u0 = jnp.asarray([0.1])
t0, t1 = 0.0, 1.0

# Make a solver:
impl.select("isotropic", ode_shape=(1,))
ibm = extrapolation.ibm_adaptive(num_derivatives=4)
ts0 = correction.taylor_order_zero(ode_order=1)
strategy = smoothers.smoother_adaptive(ibm, ts0)
solver = uncalibrated.solver(strategy)


# Solve
solution = ivpsolve.solve_with_python_while_loop(
    vf,
    initial_values=(u0,),
    t0=t0,
    t1=t1,
    solver=solver,
    output_scale=1.0,
    parameters=0.5,
)


# Look at the solution
print("u =", solution.u)
```
