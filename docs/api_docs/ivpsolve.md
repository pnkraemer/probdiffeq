# Solution routines

There are many ways to skin the cat, and there are even more ways
of stepping through time to solve an ODE.
Most libraries expose a single ``solve_ode()`` method.
``odefilters`` does not, there are differnet function calls for different
time-stepping modes and/or quantities of interest. We distinguish

### Adaptive time-stepping
* Terminal value simulation (traditional API)
* Checkpoint simulation (traditional API)
* Complete simulation (native python)
* Terminal value simulation (ODE-filter-specific)
* Checkpoint simulation (ODE-filter-specific)
* Complete simulation (ODE-filter-specific)


### Constant time-stepping
* Fixed step-sizes (TBD)
* Fixed evaluation grids (TBD)


## Why this distinction?

### Specialised solvers

If we know that only the terminal value is of interest,
we may build specialised ODE filters that compute this value,
and only this value, very efficiently.
In a more technical speak, we enter the world of Gaussian filters
and don't have to concern ourselves with smoothing mechanisms.
This avoids unnecessary matrix-matrix operations during the forward pass
and is generally a good idea if speed is required (which it always is).



### Specialised control flow

ODE solvers are iterative algorithms and heavily rely on control flow
such as while loops or scans.
But not all control flow is created equal.
Consider the most obvious example: while-loops
are not reverse-mode differentiable, and we need some tricks
to make it work (for the experts: we need to rely on continuous
sensitivity analysis to replace automatic differentiation).
But if we use a pre-determined grid, we can simply scan through time
and get all differentiation for free.
There are more of those subtleties, and it is simply easier to
split the simulations up into different functions.

### Code is easier to read
Most ``solve_ode()`` style codes are very involved and quite complex,
because they need to cover all eventualities of evaluation points (checkpoints),
events, constant or adaptive step-sizes, and so on, often with many
unused arguments depending on the function call.
In contrast, the simulation routines in ``odefilters`` are ~10 LoC and
all arguments are used (with some minor exceptions).


!!! warning "Initial value format"
    Functions in this module expect that the initial values are a tuple of arrays
    such that the vector field evaluates as
    ``vector_field(t, *initial_values, *parameters)``.
    This is different to most other ODE solver libraries, and done
    on purpose because higher-order ODEs are treated very similarly
    to first-order ODEs in this package.

## ODEFilters are not traditional IVP solvers
ODE-filters are not traditional IVP solvers.
What does this mean?
It means that the problem they are solving is not that of inferring the
solution of an initial value problem

$$
\dot u(t) = f(u(t)), \quad u(0) = u_0
$$

but a $\nu$-th order ODE filter solves the same problem subject to
what is often thought of as consistent initialisation,

$$
\dot u(t) = f(u(t)),
\quad u(0) = u_0, ~ \dot u(0) = \dot u_0, ..., u^{(\nu)}(0) = u^{(\nu)}_0.
$$

We can always create the second problem out of the first problem
with efficient automatic differentiation (see ``odefilters.taylor``).
The initial values are the Taylor coefficients of $u$ at time $t=0$.

This difference is evident if you compare the different signatures of the
ODE-filter-specific routines to those of the traditional APIs.

## Which one should I use?

It is perfectly fine to use the traditional API (not the functions marked with ``odefilter_``).
But a true pro calls the ODE filter the way it is supposed to be called.

### Adaptive simulation of specific time-points

::: odefilter.ivpsolve.simulate_terminal_values
::: odefilter.ivpsolve.odefilter_terminal_values
::: odefilter.ivpsolve.simulate_checkpoints
::: odefilter.ivpsolve.odefilter_checkpoints

### Adaptive simulation using native Python control flow
::: odefilter.ivpsolve.solve
::: odefilter.ivpsolve.odefilter
