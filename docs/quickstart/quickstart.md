# Quickstart

```python
import jax
import jax.numpy as jnp

from diffeqzoo import ivps, backend
from jax.config import config

from probdiffeq import ivpsolve, recipes, controls

backend.select("jax")  # ivp examples in jax


config.update("jax_enable_x64", True)

# Make a problem
f, u0, (t0, t1), f_args = ivps.van_der_pol_first_order(stiffness_constant=1.)

@jax.jit
def vector_field(t, y):
    return f(y, *f_args)

# Make a solver
ekf0, info_op = recipes.dynamic_isotropic_eks0(num_derivatives=5)

# Solve
with jax.disable_jit():
    solution = ivpsolve.solve_with_python_while_loop(
        vector_field, initial_values=(u0,), t0=t0, t1=t1,
        solver=ekf0, info_op=info_op, control=controls.Integral(safety=.95)
    )
# Look at the solution
print(len(solution))

print(solution.t, solution.u)
```
