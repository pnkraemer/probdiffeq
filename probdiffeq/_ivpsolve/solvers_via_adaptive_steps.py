from probdiffeq._ivpsolve import controllers, solver_protocols
from probdiffeq.backend import flow, func, np, structs, tree, warnings
from probdiffeq.backend.typing import Any, Array, Callable, Generic, TypeVar

T = TypeVar("T")
S = TypeVar("S")

__all__ = [
    "RejectionLoop",
    "TimeStepState",
    "solve_adaptive_save_at",
    "solve_adaptive_terminal_values",
]


def solve_adaptive_terminal_values(
    solver: solver_protocols.Solver,
    error,
    control: controllers.Control | None = None,
    clip_dt: bool = True,
    while_loop: Callable = flow.while_loop,
) -> Callable[..., solver_protocols.Solution]:
    """Simulate the terminal values of an initial value problem."""
    # Turn off warnings because any solver goes for terminal values
    solve_save_at = solve_adaptive_save_at(
        solver=solver,
        error=error,
        control=control,
        clip_dt=clip_dt,
        warn=False,
        while_loop=while_loop,
    )

    def solve(
        u: T, /, *, t0, t1, atol, rtol, dt0=0.1, eps=1e-8, damp=0.0
    ) -> solver_protocols.Solution[T]:
        save_at = np.asarray([t0, t1])
        solution = solve_save_at(
            u, save_at=save_at, atol=atol, rtol=rtol, dt0=dt0, eps=eps, damp=damp
        )
        return tree.tree_map(lambda s: s[-1], solution)

    return solve


def solve_adaptive_save_at(
    *,
    solver: solver_protocols.Solver,
    error,
    control: controllers.Control | None = None,
    clip_dt: bool = False,
    while_loop: Callable = flow.while_loop,
    warn=True,
) -> Callable[..., solver_protocols.Solution]:
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
        msg = f"Solver {solver} should not be used in solve_adaptive_save_at."
        msg += " This is typically caused by the wrong strategy selection."
        msg += " Try using filters or fixed-point smoothers."
        warnings.warn(msg, stacklevel=1)

    if control is None:
        # In probabilistic solvers, integral controllers seem to work better
        # than proportional-integral controllers.
        control = controllers.control_integral()

    loop = RejectionLoop(
        solver=solver,
        clip_dt=clip_dt,
        control=control,
        error=error,
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
            @structs.dataclass
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
        solution0 = solver.init(t=save_at[0], u=u, damp=damp)
        state = loop.init(solution0, dt=dt0)

        # Advance to one checkpoint after the other
        init = (solution0, state)
        xs = save_at[1:]
        (_solution, state), solution = flow.scan(
            advance, init=init, xs=xs, reverse=False
        )

        # Stack the initial value into the solution and return
        return solver.userfriendly_output(
            solution0=solution0, solution=solution, solution1=state.step_from
        )

    return solve


@tree.register_dataclass
@structs.dataclass
class TimeStepState(Generic[T]):
    """A state variable type for adaptive time-stepping."""

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

    error_step_from: Any
    """The error-estimate corresponding to 'step_from'."""


@tree.register_dataclass
@structs.dataclass
class _RejectionLoopState:
    """State for a single rejection loop.

    Keep decreasing step-size until error norm is small.
    This is a critical part of an IVP solver step.
    """

    dt: float
    acceptance_factor_proposed: float
    control: Any
    proposed: Any
    step_from: Any
    error_step_from: Any
    error_proposed: Any


class RejectionLoop:
    """An implementation of a rejection loop."""

    def __init__(
        self,
        solver: solver_protocols.Solver,
        clip_dt: bool,
        error: Any,
        control: controllers.Control,
        while_loop: Callable,
        stop_gradient_through_dt: bool = True,
    ) -> None:
        self.solver = solver
        self.clip_dt = clip_dt
        self.error = error
        self.control = control
        self.while_loop = while_loop
        self.stop_gradient_through_dt = stop_gradient_through_dt

    def init(self, state_solver, dt) -> TimeStepState:
        """Initialise the adaptive solver state."""
        state_control = self.control.init(dt)
        state_error = self.error.init_error()
        return TimeStepState(
            dt=dt,
            step_from=state_solver,
            interp_from=state_solver,
            control=state_control,
            error_step_from=state_error,
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
        state = flow.cond(is_before_t1, self.step, lambda s: s[0], args)

        # Interpolate
        is_before_t1 = state.step_from.t + eps < t1
        is_after_t1 = state.step_from.t > t1 + eps
        branch_idx = np.where(is_before_t1, 0, np.where(is_after_t1, 1, 2))
        options = (self.interp_skip, self.interp_beyond_t1, self.interp_at_t1)
        return flow.switch(branch_idx, options, (state, t1))

    def step(self, s_and_t1_and_tols_and_damp):
        """Do a rejection-loop step.

        Keep attempting steps until one is accepted.
        """
        s, t1, atol, rtol, damp = s_and_t1_and_tols_and_damp

        def cond(state: _RejectionLoopState) -> bool:
            # acceptance_factor_proposed is error_norm ** (-1/rate), thus "<"
            return state.acceptance_factor_proposed < 1.0

        init = self.step_init_loopstate(s)
        step_attempt = func.partial(
            self.step_attempt, t1=t1, atol=atol, rtol=rtol, damp=damp
        )
        state_new = self.while_loop(cond, step_attempt, init)
        return self.step_extract_timestep_state(state_new)

    def step_init_loopstate(self, s0: TimeStepState) -> _RejectionLoopState:
        """Initialise the rejection state."""

        def _ones_like(pytree):
            return tree.tree_map(np.ones_like, pytree)

        acceptance_factor_init = (
            0.9  # must be less than 1.0 to enter the while loop on the first iteration
        )
        return _RejectionLoopState(
            acceptance_factor_proposed=acceptance_factor_init,
            dt=s0.dt,
            control=s0.control,
            step_from=s0.step_from,
            error_step_from=s0.error_step_from,
            proposed=_ones_like(s0.step_from),  # irrelevant
            error_proposed=_ones_like(s0.error_step_from),  # irrelevant
        )

    def step_attempt(
        self, state: _RejectionLoopState, *, t1, atol, rtol, damp
    ) -> _RejectionLoopState:
        """Attempt a step.

        Perform a step with an IVP solver and
        propose a future time-step based on tolerances and error estimates.
        """
        dt = state.dt
        if self.stop_gradient_through_dt:
            dt = func.stop_gradient(dt)

        # Some controllers like to clip the terminal value instead of interpolating.
        # This must happen _before_ the step.
        if self.clip_dt:
            dt = np.minimum(dt, t1 - state.step_from.t)

        # Perform the actual step.
        state_proposed = self.solver.step(state=state.step_from, dt=dt, damp=damp)

        error_power, errorstate = self.error.estimate_error_norm(
            state.error_step_from,
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
            acceptance_factor_proposed=error_power,  # new
            proposed=state_proposed,  # new
            control=state_control,  # new
            error_proposed=errorstate,  # new
            error_step_from=state.error_step_from,
            step_from=state.step_from,
        )

    def step_extract_timestep_state(self, state: _RejectionLoopState) -> TimeStepState:
        """Extract a time-step-state after a successful rejection loop."""
        return TimeStepState(
            dt=state.dt,
            step_from=state.proposed,  # new!
            interp_from=state.step_from,
            control=state.control,
            error_step_from=state.error_proposed,
        )

    def interp_skip(self, args):
        """If step_from.t < t1, don't interpolate."""
        state, _t1 = args
        solution = state.step_from
        return solution, state

    def interp_beyond_t1(self, args):
        """If we stepped cleanly over t1, interpolate."""
        state, t1 = args
        solution, interp_res = self.solver.interpolate_fwd(
            t=t1, interp_from=state.interp_from, interp_to=state.step_from
        )

        new_state = TimeStepState(
            dt=state.dt,
            step_from=interp_res.step_from,
            interp_from=interp_res.interp_from,
            control=state.control,
            error_step_from=state.error_step_from,
        )
        return solution, new_state

    def interp_at_t1(self, args):
        """If we stepped exactly to t1, still interpolate."""
        state, t1 = args
        solution, interp_res = self.solver.interpolate_fwd_at_t1(
            t=t1, interp_from=state.interp_from, interp_to=state.step_from
        )
        new_state = TimeStepState(
            dt=state.dt,
            step_from=interp_res.step_from,
            interp_from=interp_res.interp_from,
            control=state.control,
            error_step_from=state.error_step_from,
        )
        return solution, new_state
