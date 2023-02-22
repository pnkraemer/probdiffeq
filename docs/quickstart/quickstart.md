# Quickstart

```python
import jax
from diffeqzoo import backend, ivps

from probdiffeq import solution_routines, solvers
from probdiffeq.implementations import recipes
from probdiffeq.strategies import smoothers

if not backend.has_been_selected:
    backend.select("jax")  # ivp examples in jax

# Make a problem
f, u0, (t0, t1), f_args = ivps.van_der_pol_first_order(stiffness_constant=1.0)


@jax.jit
def vector_field(y, *, t, p):
    return f(y, *p)


# Make a solver:
#     DenseTS1: dense covariance structure with first-order Taylor linearisation
#     Smoother: Compute a global estimate of the solution
#     MLESolver: Calibrate unknown parameters with (quasi-)maximum-likelihood estimation
implementation = recipes.DenseTS1.from_params(ode_shape=(2,))
strategy = smoothers.Smoother(implementation)
solver = solvers.MLESolver(strategy)


# Solve
solution = solution_routines.solve_with_python_while_loop(
    vector_field, initial_values=(u0,), t0=t0, t1=t1, solver=solver, parameters=f_args
)


# Look at the solution
print("u =", solution.u)
```
