"""Initial value problem solver loops.

This module does not contain any probabilistic numerics logic.
Instead, its sole purpose is to make the probabilistic solvers
in probdiffeq.probdiffeq easier to access.

See the tutorials for example use cases.
"""

from probdiffeq.backend import (
    containers,
    control_flow,
    func,
    linalg,
    np,
    tree,
    warnings,
)
from probdiffeq.backend.typing import (
    Any,
    Array,
    Callable,
    Generic,
    NamedArg,
    Protocol,
    TypeVar,
)

T = TypeVar("T", contravariant=True)
S = TypeVar("S")


class Solution(Protocol, Generic[S]):
    """An IVP solution protocol."""

    t: Array
    """The timepoints that the solution is expressed on."""

    u: S
    """The IVP solution state.

    This is the usually the same type as the initial condition.
    """


# Revisit this dependent typing one Python >=3.12 is enforced
# Concretely, Something like Solver[T, S: Solution[T]](Protocol):...
# can now be written.


class Solver(Protocol, Generic[T, S]):
    """An IVP solver protocol."""

    def init(self, *, t, u: T) -> S:
        """Initialise the solver's state."""

    def step(self, state: S, *, dt: float, damp: float) -> S:
        """Perform a step."""

    @property
    def is_suitable_for_save_at(self) -> bool:
        """Whether or not the solver can be used with adaptive time-stepping."""

    def userfriendly_output(self, *, solution: S, solution0: S) -> S:
        """Postprocess the solution before returning."""


def solve_adaptive_terminal_values(
    solver: Solver,
    errorest,
    control=None,
    clip_dt=False,
    while_loop=control_flow.while_loop,
) -> Callable:
    """Simulate the terminal values of an initial value problem."""
    # Turn off warnings because any solver goes for terminal values
    solve_save_at = solve_adaptive_save_at(
        solver=solver,
        errorest=errorest,
        control=control,
        clip_dt=clip_dt,
        warn=False,
        while_loop=while_loop,
    )

    def solve(
        u: T, /, *, t0, t1, atol, rtol, dt0=0.1, eps=1e-8, damp=0.0
    ) -> Solution[T]:
        save_at = np.asarray([t0, t1])
        solution = solve_save_at(
            u, save_at=save_at, atol=atol, rtol=rtol, dt0=dt0, eps=eps, damp=damp
        )
        return tree.tree_map(lambda s: s[-1], solution)

    return solve


def solve_adaptive_save_at(
    *,
    solver: Solver,
    errorest,
    control=None,
    clip_dt=False,
    warn=True,
    while_loop=control_flow.while_loop,
) -> Callable:
    r"""Solve an initial value problem and return the solution at a pre-determined grid.

    This algorithm implements the method by Krämer (2025). Please consider citing it
    if you use it for your research. A PDF is available
    [here](https://arxiv.org/abs/2410.10530) and Krämer's (2025) experiments are
    available [here](https://github.com/pnkraemer/code-adaptive-prob-ode-solvers).

    ??? note "BibTex for Krämer (2025)"
        ```bibtex
        @InProceedings{kramer2024adaptive,
            title     = {Adaptive Probabilistic ODE Solvers Without Adaptive Memory
                        Requirements},
            author    = {Kr{\"a}mer, Nicholas},
            booktitle = {Proceedings of the First International Conference on
                        Probabilistic Numerics},
            pages     = {12--24},
            year      = {2025},
            editor    = {Kanagawa, Motonobu and Cockayne, Jon and Gessner, Alexandra
                        and Hennig, Philipp},
            volume    = {271},
            series    = {Proceedings of Machine Learning Research},
            publisher = {PMLR},
            url       = {https://proceedings.mlr.press/v271/kramer25a.html}
        }
        ```
    """
    if not solver.is_suitable_for_save_at and warn:
        msg = f"Strategy {solver} should not be used in solve_adaptive_save_at. "
        warnings.warn(msg, stacklevel=1)

    if control is None:
        control = control_proportional_integral()

    loop = RejectionLoop(
        solver=solver,
        clip_dt=clip_dt,
        control=control,
        errorest=errorest,
        while_loop=while_loop,
    )

    def solve(
        u: T, save_at: Array, atol: float, rtol: float, dt0=0.1, eps=1e-8, damp=0.0
    ):
        def advance(sol_and_state: tuple, t_next) -> tuple[tuple, Any]:
            """Advance the adaptive solver to the next checkpoint.

            Note: we may already be beyond the checkpoint in which case
            the rejection loop automatically interpolates.
            """

            @tree.register_dataclass
            @containers.dataclass
            class AdvanceState:
                do_continue: bool
                solution: Any
                loopstate: Any

            def cond_fun(c: AdvanceState) -> bool:
                return c.do_continue

            def body_fun(state: AdvanceState) -> AdvanceState:
                solution, state_new = loop.loop(
                    state.loopstate, t1=t_next, atol=atol, rtol=rtol, eps=eps, damp=damp
                )
                do_continue = state_new.step_from.t + eps < t_next
                return AdvanceState(do_continue, solution, state_new)

            # Always step >=1x into the rejection loop
            init = AdvanceState(True, *sol_and_state)
            advanced = while_loop(cond_fun, body_fun, init)
            return (advanced.solution, advanced.loopstate), advanced.solution

        # Initialise the adaptive solver
        solution0 = solver.init(t=save_at[0], u=u)
        state = loop.init(solution0, dt=dt0)

        # Advance to one checkpoint after the other
        init = (solution0, state)
        xs = save_at[1:]
        (_solution, _state), solution = control_flow.scan(
            advance, init=init, xs=xs, reverse=False
        )

        # Stack the initial value into the solution and return
        return solver.userfriendly_output(solution0=solution0, solution=solution)

    return solve


def solve_fixed_grid(*, solver: Solver) -> Callable:
    """Solve an initial value problem on a fixed, pre-determined grid."""

    def solve(u: T, /, *, grid, damp: float = 0.0) -> Solution[T]:
        def body_fn(s, dt):
            s_new = solver.step(state=s, dt=dt, damp=damp)
            return s_new, s_new

        t0 = grid[0]
        state0 = solver.init(t=t0, u=u)
        _, result = control_flow.scan(body_fn, init=state0, xs=np.diff(grid))

        return solver.userfriendly_output(solution0=state0, solution=result)

    return solve


def dt0(vf_autonomous, initial_values, /, scale=0.01, nugget=1e-5):
    """Propose an initial time-step."""
    u0, *_ = initial_values
    f0 = vf_autonomous(*initial_values)

    u0, _ = tree.ravel_pytree(u0)
    f0, _ = tree.ravel_pytree(f0)

    norm_y0 = linalg.vector_norm(u0)
    norm_dy0 = linalg.vector_norm(f0) + nugget

    return scale * norm_y0 / norm_dy0


def dt0_adaptive(vf, initial_values, /, t0, *, error_contraction_rate, rtol, atol):
    """Propose an initial time-step as a function of the tolerances."""
    # Algorithm from:
    # E. Hairer, S. P. Norsett G. Wanner,
    # Solving Ordinary Differential Equations I: Nonstiff Problems, Sec. II.4.
    # Implementation mostly copied from
    #
    # https://github.com/google/jax/blob/main/jax/experimental/ode.py
    #

    if len(initial_values) > 1:
        raise ValueError
    y0 = initial_values[0]

    f0 = vf(y0, t=t0)

    y0, unravel = tree.ravel_pytree(y0)
    f0, _ = tree.ravel_pytree(f0)

    scale = atol + np.abs(y0) * rtol
    d0, d1 = linalg.vector_norm(y0), linalg.vector_norm(f0)

    dt0 = np.where((d0 < 1e-5) | (d1 < 1e-5), 1e-6, 0.01 * d0 / d1)

    y1 = y0 + dt0 * f0
    f1 = vf(unravel(y1), t=t0 + dt0)
    f1, _ = tree.ravel_pytree(f1)
    d2 = linalg.vector_norm((f1 - f0) / scale) / dt0

    dt1 = np.where(
        (d1 <= 1e-15) & (d2 <= 1e-15),
        np.maximum(1e-6, dt0 * 1e-3),
        (0.01 / np.maximum(d1, d2)) ** (1.0 / (error_contraction_rate + 1.0)),
    )
    return np.minimum(100.0 * dt0, dt1)


@tree.register_dataclass
@containers.dataclass
class TimeStepState(Generic[T]):
    """State for adaptive time-stepping."""

    dt: float
    """The time-step-size proposal for the next step."""

    step_from: T
    """Where to continue time-stepping from.

    This is the right-hand side boundary of the current subinterval.
    """

    interp_from: T
    """Where to continue interpolation from.

    This is the left-hand side of the current subinterval.
    """

    control: Any
    """The controller state."""

    errorest_step_from: Any
    """The error-estimate corresponding to 'step_from'."""


@tree.register_dataclass
@containers.dataclass
class _RejectionLoopState:
    """State for a single rejection loop.

    Keep decreasing step-size until error norm is small.
    This is a critical part of an IVP solver step.
    """

    dt: float
    error_norm_proposed: float
    control: Any
    proposed: Any
    step_from: Any
    errorest_step_from: Any
    errorest_proposed: Any


@containers.dataclass
class RejectionLoop:
    """Implement a rejection loop."""

    solver: Any
    """The solver."""

    clip_dt: bool
    """Whether or not to clip the timestep before stepping."""

    errorest: Any
    """Error estimator."""

    control: Any
    """The controller."""

    while_loop: Callable
    """An implementation of a while-loop to be used for the rejection loop.

    Change this to a bounded while loop to make the solvers reverse-differentiable.
    """

    def init(self, state_solver, dt) -> TimeStepState:
        """Initialise the adaptive solver state."""
        state_control = self.control.init(dt)
        state_errorest = self.errorest.init_errorest()
        return TimeStepState(
            dt=dt,
            step_from=state_solver,
            interp_from=state_solver,
            control=state_control,
            errorest_step_from=state_errorest,
        )

    def loop(
        self, state0: TimeStepState, *, t1, atol, rtol, eps, damp
    ) -> tuple[Any, TimeStepState]:
        """Repeatedly attempt a step until the controller is happy.

        Notably:
        - This function may never attempt a step if the current timestep
            is beyond t1.
        - If we step beyond t1, this function interpolates to t1.
        """
        # If t1 is in the future, enter the rejection loop (otherwise do nothing)
        is_before_t1 = state0.step_from.t + eps < t1
        args = (state0, t1, atol, rtol, damp)
        state = control_flow.cond(is_before_t1, self.step, lambda s: s[0], args)

        # Interpolate
        is_before_t1 = state.step_from.t + eps < t1
        is_after_t1 = state.step_from.t > t1 + eps
        branch_idx = np.where(is_before_t1, 0, np.where(is_after_t1, 1, 2))
        options = (self.interp_skip, self.interp_beyond_t1, self.interp_at_t1)
        return control_flow.switch(branch_idx, options, (state, t1))

    def step(self, s_and_t1_and_tols_and_damp):
        """Do a rejection-loop step.

        Keep attempting steps until one is accepted.
        """
        s, t1, atol, rtol, damp = s_and_t1_and_tols_and_damp

        def cond(state: _RejectionLoopState) -> bool:
            # error_norm_proposed is EEst ** (-1/rate), thus "<"
            return state.error_norm_proposed < 1.0

        init = self.step_init_loopstate(s)
        step_attempt = func.partial(
            self.step_attempt, t1=t1, atol=atol, rtol=rtol, damp=damp
        )
        state_new = self.while_loop(cond, step_attempt, init)
        return self.step_extract_timestep_state(state_new)

    def step_init_loopstate(self, s0: TimeStepState) -> _RejectionLoopState:
        """Initialise the rejection state."""

        def _ones_like(tree):
            return tree.tree_map(np.ones_like, tree)

        smaller_than_1 = 0.9  # the cond() must return True
        return _RejectionLoopState(
            error_norm_proposed=smaller_than_1,
            dt=s0.dt,
            control=s0.control,
            step_from=s0.step_from,
            errorest_step_from=s0.errorest_step_from,
            proposed=_ones_like(s0.step_from),  # irrelevant
            errorest_proposed=_ones_like(s0.errorest_step_from),  # irrelevant
        )

    def step_attempt(
        self, state: _RejectionLoopState, *, t1, atol, rtol, damp
    ) -> _RejectionLoopState:
        """Attempt a step.

        Perform a step with an IVP solver and
        propose a future time-step based on tolerances and error estimates.
        """
        dt = state.dt

        # Some controllers like to clip the terminal value instead of interpolating.
        # This must happen _before_ the step.
        if self.clip_dt:
            dt = np.minimum(dt, t1 - state.step_from.t)

        # Perform the actual step.
        state_proposed = self.solver.step(state=state.step_from, dt=dt, damp=damp)

        error_power, errorstate = self.errorest.estimate_error_norm(
            state.errorest_step_from,
            previous=state.step_from,
            proposed=state_proposed,
            dt=dt,
            atol=atol,
            rtol=rtol,
            damp=damp,
        )

        # Propose a new step
        dt, state_control = self.control.apply(
            dt, state.control, error_power=error_power
        )
        return _RejectionLoopState(
            dt=dt,  # new
            error_norm_proposed=error_power,  # new
            proposed=state_proposed,  # new
            control=state_control,  # new
            errorest_proposed=errorstate,  # new
            errorest_step_from=state.errorest_step_from,
            step_from=state.step_from,
        )

    def step_extract_timestep_state(self, state: _RejectionLoopState) -> TimeStepState:
        """Extract a time-step-state after a successful rejection loop."""
        return TimeStepState(
            dt=state.dt,
            step_from=state.proposed,
            interp_from=state.step_from,
            control=state.control,
            errorest_step_from=state.errorest_proposed,
        )

    def interp_skip(self, args):
        """If step_from.t < t1, don't interpolate."""
        state, t1 = args
        solution = state.step_from
        return solution, state

    def interp_beyond_t1(self, args):
        """If we stepped cleanly over t1, interpolate."""
        state, t1 = args
        solution, interp_res = self.solver.interpolate(
            t=t1, interp_from=state.interp_from, interp_to=state.step_from
        )

        new_state = TimeStepState(
            dt=state.dt,
            step_from=interp_res.step_from,
            interp_from=interp_res.interp_from,
            control=state.control,
            errorest_step_from=state.errorest_step_from,
        )
        return solution, new_state

    def interp_at_t1(self, args):
        """If we stepped exactly to t1, still interpolate."""
        state, t1 = args
        solution, interp_res = self.solver.interpolate_at_t1(
            t=t1, interp_from=state.interp_from, interp_to=state.step_from
        )
        new_state = TimeStepState(
            dt=state.dt,
            step_from=interp_res.step_from,
            interp_from=interp_res.interp_from,
            control=state.control,
            errorest_step_from=state.errorest_step_from,
        )
        return solution, new_state


@containers.dataclass
class _Controller(Generic[T]):
    """Control algorithm."""

    init: Callable[[float], T]
    """Initialise the controller state."""

    apply: Callable[[float, T, NamedArg(float, "error_power")], tuple[float, T]]
    """Propose a time-step-size."""


def control_proportional_integral(
    *,
    safety=0.95,
    factor_min=0.2,
    factor_max=10.0,
    power_integral_unscaled=0.3,
    power_proportional_unscaled=0.4,
) -> _Controller[float]:
    """Construct a proportional-integral-controller with time-clipping."""

    def init(_dt: float, /) -> float:
        return 1.0

    def apply(dt: float, error_power_prev: float, /, *, error_power):
        # Equivalent: error_power = error_norm ** (-1.0 / error_contraction_rate)
        a1 = error_power**power_integral_unscaled
        a2 = (error_power / error_power_prev) ** power_proportional_unscaled
        scale_factor_unclipped = safety * a1 * a2

        scale_factor_clipped_min = np.minimum(scale_factor_unclipped, factor_max)
        scale_factor = np.maximum(factor_min, scale_factor_clipped_min)

        # >= 1.0 because error_power is 1/scaled_error_norm
        error_power_prev = np.where(error_power >= 1.0, error_power, error_power_prev)

        dt_proposed = scale_factor * dt
        return dt_proposed, error_power_prev

    return _Controller(init=init, apply=apply)


def control_integral(
    *, safety=0.95, factor_min=0.2, factor_max=10.0
) -> _Controller[None]:
    """Construct an integral-controller."""

    def init(_dt, /) -> None:
        return None

    def apply(dt, _state, /, *, error_power):
        scale_factor_unclipped = safety * error_power

        scale_factor_clipped_min = np.minimum(scale_factor_unclipped, factor_max)
        scale_factor = np.maximum(factor_min, scale_factor_clipped_min)
        return scale_factor * dt, None

    return _Controller(init=init, apply=apply)
