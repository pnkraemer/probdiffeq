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

To implement this overstepping behaviour, we need to be super careful:
Smoothing-based ODE filters finalise the function call with a backward pass
after the forward-solve, and the state-space model for this backward pass
is implemented during the forward pass.
But overstepping-interpolating changes the backward model quite drastically:
the interpolating depends on the previous _and the future state_, and
the backward state-space model needs to be reset appropriately (trust us for now).


One extra factor is to additionally track a ``solution`` state.
The ``solution`` state usually coincides with the ``accepted`` state.
It gets interesting when they disagree, which is usually when the time-location
of the accepted state is beyond the next user-specified checkpoint.


Consider the following example implementation

```python
def step_fn(*problem, state):

    if state.accepted.t < t1:  # A/A'
        state = reject_accept_loop(*problem, state)

        if state.accepted.t == t1:  # B/B'
            state.accepted = reset_at_checkpoint(state.accepted)

        elif state.accepted.t > t1:  # C/C'
            state = interpolate(state, t0=t1)

    else:  # C/C'
        assert state.accepted.t > t1
        state = interpolate(state, to=t1)

    return state

```

A, A') Attempt-step-loop / no loop

B, B') Stepped exactly to final time / did not

C, C') Overstepped - interpolate / did not



* if A happens, only _either_ B or C or B'&C' can happen
* if A does not happen, B cannot happen; but C must happen because otherwise, nothing happens
* if in the previous step, B happened, A must happen in the next step
* if in the previous step, A-B'-C' happens, A must happen in the next step

Cases for a single step:

1) A-B'-C'
2) A-B-C'
3) A-B'-C
4) A'-B'-C

Cases for a step-step combination:
1-1,
1-2,
1-3,
2-1,
2-2,
2-3,
3-1,
3-2,
3-3,
3-4,
4-1,
4-2,
4-3,
4-4,

Impossible combinations:
2-4,
1-4,
