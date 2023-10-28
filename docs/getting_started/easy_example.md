# An easy example

Let's have a look at an easy example.

```python
"""Solve the logistic equation."""

import jax
import jax.numpy as jnp
from jax.config import config

from probdiffeq import adaptive, ivpsolve, timestep
from probdiffeq.impl import impl
from probdiffeq.solvers import uncalibrated
from probdiffeq.solvers.strategies import smoothers
from probdiffeq.solvers.strategies.components import corrections, priors
from probdiffeq.taylor import autodiff

config.update("jax_platform_name", "cpu")
```

Create a problem:

```python
@jax.jit
def vf(y, *, t):  # noqa: ARG001
    """Evaluate the vector field."""
    return 0.5 * y * (1 - y)


u0 = jnp.asarray([0.1])
t0, t1 = 0.0, 1.0
```

<!-- #region -->

ProbDiffEq contains three levels of implementations:

**Low:** Implementations of random-variable-arithmetic (marginalisation, conditioning, etc.)

**Medium:** Probabilistic IVP solver components (this is what you're here for.)

**High:** ODE-solving routines.


There are several random-variable implementations (read: state-space model factorisations) which model different correlations between variables.
All factorisations can be used interchangeably, but they have different speed, stability, and uncertainty-quantification properties.
Since the chosen implementation powers almost everything, we choose one (and only one) of them, assign it to a global variable, and call it the "impl(ementation)".

<!-- #endregion -->

```python
impl.select("dense", ode_shape=(1,))
```
But don't worry, this configuration does not make the library any less light-weight.
It merely affects the shapes of the arrays describing means and covariances of Gaussian 
random variables, and assigns functions that know how to manipulate those parameters.



Configuring a probabilistic IVP solver is a little more involved than configuring your favourite Runge-Kutta method:
we must choose a prior distribution and a correction scheme, then we put them together as a filter or smoother, wrap everything into a solver, and (finally) make the solver adaptive.


```python
ibm = priors.ibm_adaptive(num_derivatives=4)
ts0 = corrections.ts1(ode_order=1)

strategy = smoothers.smoother_adaptive(ibm, ts0)
solver = uncalibrated.solver(strategy)
adaptive_solver = adaptive.adaptive(solver)
```

Why so many layers?

* Prior distributions incorporate prior knowledge; better prior knowledge should `improve the simulation'' (which might mean something different for different applications, hence the quotation marks)
* Different correction schemes imply different stability concerns
* Filters and smoothers are optimised estimators for either forward-only or time-series estimation
* Calibration schemes affect the behaviour of the solver
* Not all solution routines expect adaptive solvers.

The granularity of construction a solver is an asset, not a drawback.

Finally, we must prepare one last component before we can solve the differential equation:

The probabilistic IVP solvers in ProbDiffEq implement state-space-model-based IVP solvers; this means that as an initial condition, we must provide a data structure that represents the initial state in this model.
For all current solvers, this amounts to computing a $\nu$-th order Taylor approximation of the IVP solution
and to wrapping this approximation into a state-space-model variable.

Use the following functions:

```python
tcoeffs = autodiff.taylor_mode_scan(lambda y: vf(y, t=t0), (u0,), num=4)
output_scale = 1.0  # or any other value with the same shape
init = solver.initial_condition(tcoeffs, output_scale)
```

Other software packages that implement probabilistic IVP solvers do a lot of this work 
implicitly; probdiffeq enforces that the user makes these decisions, not only because 
it simplifies the solver implementations (quite a lot, actually), 
but it also shows how easily we can build a custom solver for our favourite problem 
(consult the other tutorials for examples).


From here on, the rest is standard ODE-solver machinery:

```python
dt0 = timestep.initial(lambda y: vf(y, t=t0), (u0,))  # or use e.g. dt0=0.1
solution = ivpsolve.solve_and_save_every_step(
    vf, init, t0=t0, t1=t1, dt0=dt0, adaptive_solver=adaptive_solver
)


# Look at the solution
print("u =", solution.u, "\n")
print("solution =", solution)
```
