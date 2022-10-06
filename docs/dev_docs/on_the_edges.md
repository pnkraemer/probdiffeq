# On the edges

ODE solvers are adaptive, iterative algorithms that start at some initial time $t_0$ and iterate until a terminal time $t_1$ is reachced.

What is the stepping logic close to the terminal time $t_1$?
More specifically, what happens if the ODE solver is at time $t = t_1 - \epsilon$, and a time-increment of $\Delta t = 2\epsilon$ is suggested by the control algorithm?
In this case, we have the choices between

1. Stepping over the terminal time, in which case the solver finishes at time-point $t_1 + \epsilon$.
2. Stepping to exactly the terminal time, i.e., $\Delta t$ is capped to $\min(t_1 - t, \Delta t)$. A normal step is taken.
3. Extrapolating to the terminal time-point. The ODE vector field is not evaluated at the terminal value.
4. Stepping over the terminal time and using dense output to evaluate the solution at the terminal time.

Option 1. is not good, because it goes against what a user would expect.
Optional 2. is easiest to implement, but leads to issues with checkpointing:
when the number of checkpoints is larger than the number of solver-grid points,
which happens if you want to plot a high-resolution version of an inaccurate approximation,
then this strategy does not work.
Option 3. requires something like dense output.
Option 4. might be the "best" solution in terms of intuitive behaviour, but it requires that the ODE solution is defined beyond the terminal time point.
Most are, but some are not (todo: give example).


**What do SciPy/Diffrax/JAX-ODE/Matlab/OrdinaryDiffeq.jl do?**


ODE filters are extrapolate-correct algorithms.
