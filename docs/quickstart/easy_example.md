# An easy example

```python
import jax
import jax.numpy as jnp
from jax.config import config

from probdiffeq import ivpsolve
from probdiffeq.impl import impl
from probdiffeq.solvers import uncalibrated
from probdiffeq.solvers.statespace import correction, extrapolation
from probdiffeq.solvers.strategies import smoothers

config.update("jax_platform_name", "cpu")
```

Create a problem:

```python
@jax.jit
def vf(y, *, t, p):
    return p * y * (1 - y)


u0 = jnp.asarray([0.1])
t0, t1 = 0.0, 1.0
```

<!-- #region -->
Create a solver.


ProbDiffEq consists of three levels:

**Low:** Implementations of random-variable-arithmetic (marginalisation, conditioning, etc.)

**Medium:** Probabilistic IVP solver components (this is what you're here for.)

**High:** ODE-solving routines.


There are several random-variable implementations which model different correlations between variables.
Since the implementations power almost everything, we choose one (and only one) of them and call it the "impl(ementation)".

<!-- #endregion -->

```python
impl.select("isotropic", ode_shape=(1,))
```

Configuring a probabilistic IVP solver amounts to choosing an extrapolation model and a correction scheme, putting it together as a filter or smoother, and wrapping everything into a solver. 

```python
ibm = extrapolation.ibm_adaptive(num_derivatives=4)
ts0 = correction.taylor_order_zero(ode_order=1)

strategy = smoothers.smoother_adaptive(ibm, ts0)
solver = uncalibrated.solver(strategy)
```

The rest is standard ODE-solver machinery.

```python
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

```python

```
