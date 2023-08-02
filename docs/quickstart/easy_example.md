# An easy example

```python
import jax
import jax.numpy as jnp

from probdiffeq import ivpsolve
from probdiffeq.ivpsolvers import calibrated
from probdiffeq.statespace import recipes
from probdiffeq.strategies import smoothers

# Make a problem


@jax.jit
def vector_field(y, *, t, p):
    return p * y * (1 - y)


u0 = jnp.asarray([0.1])
t0, t1 = 0.0, 1.0

# Make a solver:
#     DenseTS1: dense covariance structure with first-order Taylor linearisation
#     Smoother: Compute a global estimate of the solution
#     solver_mle:
#       Calibrate unknown parameters with (quasi-)maximum-likelihood estimation
implementation = recipes.ts1_dense(ode_shape=(1,))
strategy = smoothers.smoother(*implementation)
solver = calibrated.mle(*strategy)


# Solve
solution = ivpsolve.solve_with_python_while_loop(
    vector_field, initial_values=(u0,), t0=t0, t1=t1, solver=solver, parameters=0.5
)


# Look at the solution
print("u =", solution.u)
```
