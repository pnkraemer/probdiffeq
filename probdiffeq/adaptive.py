"""Adaptive solvers for initial value problems (IVPs)."""
from probdiffeq import controls
from probdiffeq.backend import containers, control_flow, functools, linalg, tree_util
from probdiffeq.backend import numpy as np
from probdiffeq.backend.typing import Any
from probdiffeq.impl import impl


def adaptive(solver, atol=1e-4, rtol=1e-2, control=None, norm_ord=None):
    """Make an IVP solver adaptive."""
    if control is None:
        control = controls.proportional_integral()

    return _AdaptiveIVPSolver(
        solver, atol=atol, rtol=rtol, control=control, norm_ord=norm_ord
    )


class _RejectionState(containers.NamedTuple):
    """State for rejection loops.

    (Keep decreasing step-size until error norm is small.
    This is one part of an IVP solver step.)
    """

    error_norm_proposed: float
    control: Any
    proposed: Any
    step_from: Any


class _AdaptiveState(containers.NamedTuple):
    step_from: Any
    interp_from: Any
    control: Any
    stats: Any

    @property
    def t(self):
        return self.step_from.t


class _AdaptiveIVPSolver:
    """Adaptive IVP solvers."""

    def __init__(self, solver, atol, rtol, control, norm_ord):
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.control = control
        self.norm_ord = norm_ord

    def __repr__(self):
        return (
            f"\n{self.__class__.__name__}("
            f"\n\tsolver={self.solver},"
            f"\n\tatol={self.atol},"
            f"\n\trtol={self.rtol},"
            f"\n\tcontrol={self.control},"
            f"\n\tnorm_order={self.norm_ord},"
            "\n)"
        )

    @functools.jit
    def init(self, t, initial_condition, dt0, num_steps):
        """Initialise the IVP solver state."""
        state_solver = self.solver.init(t, initial_condition)
        state_control = self.control.init(dt0)
        return _AdaptiveState(state_solver, state_solver, state_control, num_steps)

    @functools.jit
    def rejection_loop(self, state0, *, vector_field, t1):
        def cond_fn(s):
            return s.error_norm_proposed > 1.0

        def body_fn(s):
            return self._attempt_step(state=s, vector_field=vector_field, t1=t1)

        def init(s0):
            larger_than_1 = 1.1
            return _RejectionState(
                error_norm_proposed=larger_than_1,
                control=s0.control,
                proposed=_inf_like(s0.step_from),
                step_from=s0.step_from,
            )

        def extract(s):
            num_steps = state0.stats + 1
            return _AdaptiveState(s.proposed, s.step_from, s.control, num_steps)

        init_val = init(state0)
        state_new = control_flow.while_loop(cond_fn, body_fn, init_val)
        return extract(state_new)

    def _attempt_step(self, *, state: _RejectionState, vector_field, t1):
        """Attempt a step.

        Perform a step with an IVP solver and
        propose a future time-step based on tolerances and error estimates.
        """
        # Some controllers like to clip the terminal value instead of interpolating.
        # This must happen _before_ the step.
        state_control = self.control.clip(state.control, t=state.step_from.t, t1=t1)

        # Perform the actual step.
        # todo: error estimate should be a tuple (abs, rel)
        error_estimate, state_proposed = self.solver.step(
            state=state.step_from,
            vector_field=vector_field,
            dt=self.control.extract(state_control),
        )
        # Normalise the error
        u_proposed = impl.hidden_model.qoi(state_proposed.strategy.hidden)
        u_step_from = impl.hidden_model.qoi(state_proposed.strategy.hidden)
        u = np.maximum(np.abs(u_proposed), np.abs(u_step_from))
        error_normalised = self._normalise_error(error_estimate, u=u)

        # Propose a new step
        error_contraction_rate = self.solver.strategy.extrapolation.num_derivatives + 1
        state_control = self.control.apply(
            state_control,
            error_normalised=error_normalised,
            error_contraction_rate=error_contraction_rate,
        )
        return _RejectionState(
            error_norm_proposed=error_normalised,  # new
            proposed=state_proposed,  # new
            control=state_control,  # new
            step_from=state.step_from,
        )

    def _normalise_error(self, error_estimate, *, u):
        error_relative = error_estimate / (self.atol + self.rtol * np.abs(u))
        dim = np.atleast_1d(u).size
        return linalg.vector_norm(error_relative, order=self.norm_ord) / np.sqrt(dim)

    def extract(self, state):
        solution_solver = self.solver.extract(state.step_from)
        solution_control = self.control.extract(state.control)
        return solution_solver, solution_control, state.stats

    def right_corner_and_extract(self, state):
        interp = self.solver.right_corner(state.interp_from, state.step_from)
        accepted, solution, previous = interp
        state = _AdaptiveState(accepted, previous, state.control, state.stats)

        solution_solver = self.solver.extract(solution)
        solution_control = self.control.extract(state.control)
        return state, (solution_solver, solution_control, state.stats)

    def interpolate_and_extract(self, state, t):
        interp = self.solver.interpolate(s1=state.step_from, s0=state.interp_from, t=t)

        accepted, solution, previous = interp
        state = _AdaptiveState(accepted, previous, state.control, state.stats)

        solution_solver = self.solver.extract(solution)
        solution_control = self.control.extract(state.control)
        return state, (solution_solver, solution_control, state.stats)


# Register outside of class to declutter the AdaptiveIVPSolver source code a bit


def _asolver_flatten(asolver):
    children = (asolver.solver, asolver.atol, asolver.rtol, asolver.control)
    aux = (asolver.norm_ord,)
    return children, aux


def _asolver_unflatten(aux, children):
    solver, atol, rtol, control = children
    (norm_ord,) = aux
    return _AdaptiveIVPSolver(
        solver=solver, atol=atol, rtol=rtol, control=control, norm_ord=norm_ord
    )


tree_util.register_pytree_node(
    _AdaptiveIVPSolver, flatten_func=_asolver_flatten, unflatten_func=_asolver_unflatten
)


def _inf_like(tree):
    return tree_util.tree_map(lambda x: np.inf() * np.ones_like(x), tree)
