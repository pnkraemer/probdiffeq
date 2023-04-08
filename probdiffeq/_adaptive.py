"""Adaptive solvers for initial value problems (IVPs)."""
from typing import Generic, TypeVar

import jax
import jax.numpy as jnp

from probdiffeq import controls, ivpsolvers

S = TypeVar("S")
"""A type-variable for generic IVP solver states."""

C = TypeVar("C", bound=controls.AbstractControl)
"""A type-variable for generic controller states."""

T = TypeVar("T", bound=ivpsolvers.AbstractSolver)
"""A type-variable for (non-adaptive) IVP solvers."""


# basically a namedtuple, but NamedTuples cannot be generic,
#  which is why we implement this functionality manually.
@jax.tree_util.register_pytree_node_class
class _AdaptiveState(Generic[S, C]):
    """Adaptive IVP solver state."""

    def __init__(
        self,
        dt_proposed,
        error_norm_proposed,
        control: C,
        proposed: S,
        accepted: S,
        solution: S,
        previous: S,
    ):
        self.dt_proposed = dt_proposed
        self.error_norm_proposed = error_norm_proposed
        self.control = control
        self.proposed = proposed
        self.accepted = accepted
        self.solution = solution
        self.previous = previous

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"\n\tdt_proposed={self.dt_proposed},"
            f"\n\terror_norm_proposed={self.error_norm_proposed},"
            f"\n\tcontrol={self.control},"
            f"\n\tproposed={self.proposed},"
            f"\n\taccepted={self.accepted},"
            f"\n\tsolution={self.solution},"
            f"\n\tprevious={self.previous},"
            "\n)"
        )

    def tree_flatten(self):
        children = (
            self.dt_proposed,
            self.error_norm_proposed,
            self.control,
            self.proposed,
            self.accepted,
            self.solution,
            self.previous,
        )
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        (
            dt_proposed,
            error_norm_proposed,
            control,
            proposed,
            accepted,
            solution,
            previous,
        ) = children
        return cls(
            dt_proposed=dt_proposed,
            error_norm_proposed=error_norm_proposed,
            control=control,
            proposed=proposed,
            accepted=accepted,
            solution=solution,
            previous=previous,
        )


def _reference_state_fn_max_abs(sol, sol_previous):
    return jnp.maximum(jnp.abs(sol), jnp.abs(sol_previous))


@jax.tree_util.register_pytree_node_class
class AdaptiveIVPSolver(Generic[T]):
    """Adaptive IVP solvers."""

    def __init__(
        self,
        solver: T,
        atol=1e-4,
        rtol=1e-2,
        control=controls.ProportionalIntegral(),
        norm_ord=None,
        numerical_zero=1e-10,
        while_loop_fn=jax.lax.while_loop,
        reference_state_fn=_reference_state_fn_max_abs,
    ):
        self.solver = solver
        self.while_loop_fn = while_loop_fn
        self.atol = atol
        self.rtol = rtol
        self.control = control
        self.norm_ord = norm_ord
        self.numerical_zero = numerical_zero
        self.reference_state_fn = reference_state_fn

    def __repr__(self):
        return (
            f"\n{self.__class__.__name__}("
            f"\n\tsolver={self.solver},"
            f"\n\tatol={self.atol},"
            f"\n\trtol={self.rtol},"
            f"\n\tcontrol={self.control},"
            f"\n\tnorm_order={self.norm_ord},"
            f"\n\tnumerical_zero={self.numerical_zero},"
            f"\n\treference_state_fn={self.reference_state_fn},"
            "\n)"
        )

    def tree_flatten(self):
        children = (
            self.solver,
            self.atol,
            self.rtol,
            self.control,
            self.numerical_zero,
        )
        aux = self.norm_ord, self.reference_state_fn, self.while_loop_fn
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        solver, atol, rtol, control, numerical_zero = children
        norm_ord, reference_state_fn, while_loop_fn = aux
        return cls(
            solver=solver,
            while_loop_fn=while_loop_fn,
            atol=atol,
            rtol=rtol,
            control=control,
            numerical_zero=numerical_zero,
            norm_ord=norm_ord,
            reference_state_fn=reference_state_fn,
        )

    @property
    def error_contraction_rate(self):
        """Error order."""
        return self.solver.strategy.implementation.extrapolation.num_derivatives + 1

    @jax.jit
    def init_fn(self, *, taylor_coefficients, t0, dt0):
        """Initialise the IVP solver state."""
        # todo: make a function of posterior, state_control, and dt_proposed
        # Initialise the components
        state_solver = self.solver.init_fn(
            taylor_coefficients=taylor_coefficients, t0=t0
        )
        state_control = self.control.init_fn()

        # Initialise (prototypes for) proposed values
        error_norm_proposed = self._normalise_error(
            error_estimate=state_solver.error_estimate,
            u=state_solver.u,
            atol=self.atol,
            rtol=self.rtol,
            norm_ord=self.norm_ord,
        )
        return _AdaptiveState(
            dt_proposed=dt0,
            error_norm_proposed=error_norm_proposed,
            solution=state_solver,
            proposed=state_solver,
            accepted=state_solver,
            previous=state_solver,
            control=state_control,
        )

    @jax.jit
    def step_fn(self, state, vector_field, t1, parameters):
        """Perform a full step (including acceptance/rejection)."""
        enter_rejection_loop = state.accepted.t + self.numerical_zero < t1
        state = jax.lax.cond(
            enter_rejection_loop,
            lambda s: self._rejection_loop(
                state0=s, vector_field=vector_field, t1=t1, parameters=parameters
            ),
            lambda s: s,
            state,
        )
        state = jax.lax.cond(
            state.accepted.t + self.numerical_zero >= t1,
            lambda s: self._interpolate(state=s, t=t1),
            lambda s: s,
            state,
        )
        return state

    def _rejection_loop(self, *, vector_field, state0, t1, parameters):
        def cond_fn(x):
            proceed_iteration, _ = x
            return proceed_iteration

        def body_fn(x):
            _, s = x
            s = self._attempt_step_fn(
                state=s, vector_field=vector_field, t1=t1, parameters=parameters
            )
            proceed_iteration = s.error_norm_proposed > 1.0
            return proceed_iteration, s

        def init_fn(s):
            return True, s

        _, state_new = self.while_loop_fn(cond_fn, body_fn, init_fn(state0))
        return _AdaptiveState(
            dt_proposed=state_new.dt_proposed,
            error_norm_proposed=_inf_like(state_new.error_norm_proposed),
            proposed=_inf_like(state_new.proposed),  # meaningless?
            accepted=state_new.proposed,  # holla! New! :)
            solution=state_new.proposed,  # Overwritten by interpolate() if necessary
            previous=state0.accepted,  # holla! New! :)
            control=state_new.control,
        )

    def _attempt_step_fn(self, *, state, vector_field, t1, parameters):
        """Perform a step with an IVP solver and \
        propose a future time-step based on tolerances and error estimates."""
        # Some controllers like to clip the terminal value instead of interpolating.
        # This must happen _before_ the step.
        dt = self.control.clip_fn(state=state.accepted, dt=state.dt_proposed, t1=t1)

        # Perform the actual step.
        posterior = self.solver.step_fn(
            state=state.accepted,
            vector_field=vector_field,
            dt=dt,
            parameters=parameters,
        )
        # Normalise the error and propose a new step.
        error_normalised = self._normalise_error(
            error_estimate=posterior.error_estimate,
            u=self.reference_state_fn(posterior.u, state.accepted.u),
            atol=self.atol,
            rtol=self.rtol,
            norm_ord=self.norm_ord,
        )
        dt_proposed, state_control = self.control.control_fn(
            state=state.control,
            error_normalised=error_normalised,
            error_contraction_rate=self.error_contraction_rate,
            dt_previous=state.dt_proposed,
        )
        return _AdaptiveState(
            dt_proposed=dt_proposed,  # new
            error_norm_proposed=error_normalised,  # new
            proposed=posterior,  # new
            solution=state.solution,  # too early to accept :)
            accepted=state.accepted,  # too early to accept :)
            previous=state.previous,  # too early to accept :)
            control=state_control,  # new
        )

    @staticmethod
    def _normalise_error(*, error_estimate, u, atol, rtol, norm_ord):
        error_relative = error_estimate / (atol + rtol * jnp.abs(u))
        dim = jnp.atleast_1d(u).size
        return jnp.linalg.norm(error_relative, ord=norm_ord) / jnp.sqrt(dim)

    def _interpolate(self, *, state: _AdaptiveState[S, C], t) -> _AdaptiveState[S, C]:
        accepted, solution, previous = self.solver.interpolate_fn(
            s0=state.previous, s1=state.accepted, t=t
        )
        return _AdaptiveState(
            dt_proposed=state.dt_proposed,
            error_norm_proposed=state.error_norm_proposed,
            proposed=_inf_like(state.proposed),
            accepted=accepted,  # holla! New! :)
            solution=solution,  # holla! New! :)
            previous=previous,  # holla! New! :)
            control=state.control,
        )

    def extract_fn(self, *, state: _AdaptiveState[S, C]) -> S:
        return self.solver.extract_fn(state.solution)

    def extract_terminal_value_fn(self, *, state: _AdaptiveState[S, C]) -> S:
        return self.solver.extract_terminal_value_fn(state.solution)


def _inf_like(tree):
    return jax.tree_map(lambda x: jnp.inf * jnp.ones_like(x), tree)
