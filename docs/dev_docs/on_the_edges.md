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
Option 2. is easiest to implement, but leads to issues with checkpointing:
when the number of checkpoints is larger than the number of solver-grid points,
which happens if you want to plot a high-resolution version of an inaccurate approximation,
then this strategy does not work.
Option 3. requires something like dense output.
Option 4. might be the "best" solution in terms of intuitive behaviour, but it requires that the ODE solution is defined beyond the terminal time point.

We do 4. Here is how.

## Implementation

Adaptive ODE solvers track a ``proposed`` state and an ``accepted`` state.
The ``proposed`` state is refined until some error criterion is satisfied,
and then the ``proposed`` state becomes the ``accepted`` state.

To implement this overstepping behaviour, we additionally track a ``solution`` state.
The ``solution`` state usually coincides with the ``accepted`` state.
It gets interesting when they disagree, which is usually when the time-location
of the accepted state disagrees with a user-specified checkpoint.
The following cases are distinguished.

### Case 1: $t < t + \Delta t < t_1$
Nothing special happens, we step as usual. The ``accepted`` and the ``solution``
state coincide at all locations.
The next step starts at the ``solution`` state
(which coincides with the ``accepted`` state anyway).

### Case 2: $t < t_1 < t + \Delta t$
We step from $t$ to $\Delta t$ and interpolate the solution to obtain
an estimate at $t_1$.
Here, the ``accepted`` state is at time-point $t + \Delta t$,
but the ``solution`` state is at time-point $t_1 < t + \Delta t$.
How does this work?
Consider the following example implementation

```python
def iterate(state, t1, step_fn, *problem):
    while state.accepted.t < t1:
        # state.solution = state.accepted
        state = step_fn(*problem, state)

    if state.accepted.t > t1:
        # state.solution != state.accepted
        state.solution = interpolate(*problem, s0=state.previous, s1=state.accepted, t=t1)
    return state

```
Interpolation itself is solver-specific.
