# An easy example

```python
import jax
import jax.numpy as jnp
from jax.config import config

from probdiffeq import ivpsolve, timestep
from probdiffeq.impl import impl
from probdiffeq.solvers import uncalibrated
from probdiffeq.solvers.strategies import smoothers, correction, priors
from probdiffeq.solvers.taylor import autodiff

config.update("jax_platform_name", "cpu")
```

Create a problem:

```python
@jax.jit
def vf(y, *, t):
    return 0.5 * y * (1 - y)


u0 = jnp.asarray([0.1])
t0, t1 = 0.0, 1.0
```

<!-- #region -->
Create a solver.


ProbDiffEq contains three levels of implementations:

**Low:** Implementations of random-variable-arithmetic (marginalisation, conditioning, etc.)

**Medium:** Probabilistic IVP solver components (this is what you're here for.)

**High:** ODE-solving routines.


There are several random-variable implementations which model different correlations between variables.
Since the implementations power almost everything, we choose one (and only one) of them, assign it to a global variable, and call it the "impl(ementation)".

<!-- #endregion -->

```python
impl.select("dense", ode_shape=(1,))
```

Configuring a probabilistic IVP solver is a little more involved than configuring your favourite Runge-Kutta method:
we must choose a prior distribution and a correction scheme, then we put them together as a filter or smoother and wrapping everything into a solver.

```python
ibm = priors.ibm_adaptive(num_derivatives=4)
ts0 = correction.ts1(ode_order=1)

strategy = smoothers.smoother_adaptive(ibm, ts0)
solver = uncalibrated.solver(strategy)
```

Finally, we must prepare one last component before we can solve the differential equation.

The probabilistic IVP solvers in ProbDiffEq implement state-space-model-based IVP solvers; this means that as an initial condition we must provide a $\nu$-th order Taylor approximation of the IVP solution.

Initial conditions can be turned into Taylor series as follows:

```python
tcoeffs = autodiff.taylor_mode(lambda y: vf(y, t=t0), (u0,), num=4)
```

Other software packages that implement probabilistic IVP solvers do a lot of this work implicitly; probdiffeq enforces that the user makes these decisions, not only because it simplifies the solver implementations quite a lot, but it also shows how easily we can build a custom solver for our favourite problem (consult the other tutorials for examples).


From here on, the rest is standard ODE-solver machinery.

```python
dt0 = timestep.propose(lambda y: vf(y, t=t0), (u0,))  # or use e.g. dt0=0.1
solution = ivpsolve.solve_and_save_every_step(
    vf,
    tcoeffs,
    t0=t0,
    t1=t1,
    solver=solver,
    dt0=dt0,
    output_scale=1.0,
)


# Look at the solution
print("u =", solution.u, "\n")
print("solution =", solution)
```
