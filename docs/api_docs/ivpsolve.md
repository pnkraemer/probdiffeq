# Solution routines

There are many ways to skin the cat, and there are even more ways
of stepping through time to solve an ODE.
Most libraries expose a single ``solve_ode()`` method.
``odefilters`` does not, there are differnet function calls for different
time-stepping modes and/or quantities of interest. We distinguish

### Adaptive time-stepping
* Terminal value simulation
* Checkpoint simulation
* Complete simulation (native python)
* Complete simulation (diffrax' bounded_while_loop)

### Constant time-stepping
* Fixed step-sizes
* Fixed evaluation grids



_**Why this distinction?**_

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
    ``vector_field(*initial_values, t, *parameters)``.
    This is different to most other ODE solver libraries, and done
    on purpose because higher-order ODEs are treated very similarly
    to first-order ODEs in this package.


::: odefilter.ivpsolve.simulate_terminal_values

::: odefilter.ivpsolve.simulate_checkpoints
