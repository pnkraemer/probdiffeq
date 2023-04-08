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
        error_norm_proposed,
        control: C,
        proposed: S,
        accepted: S,
        solution: S,
        previous: S,
    ):
        self.error_norm_proposed = error_norm_proposed
        self.control = control
        self.proposed = proposed
        self.accepted = accepted
        self.solution = solution
        self.previous = previous

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
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
            error_norm_proposed,
            control,
            proposed,
            accepted,
            solution,
            previous,
        ) = children
        return cls(
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
        control=None,
        norm_ord=None,
        numerical_zero=1e-10,
        while_loop_fn=jax.lax.while_loop,
        reference_state_fn=_reference_state_fn_max_abs,
    ):
        if control is None:
            control = controls.ProportionalIntegral()

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
    def init_fn(self, dt0, **solver_init_kwargs):
        """Initialise the IVP solver state."""
        # Initialise the components
        state_control = self.control.init_state_from_dt(dt0)
        state_solver = self.solver.init_fn(**solver_init_kwargs)

        # Initialise (prototypes for) proposed values
        error_norm_proposed = self._normalise_error(
            error_estimate=state_solver.error_estimate,
            u=state_solver.u,
            atol=self.atol,
            rtol=self.rtol,
            norm_ord=self.norm_ord,
        )
        return _AdaptiveState(
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
        state_control = self.control.clip(
            t=state.accepted.t, state=state.control, t1=t1
        )

        # Perform the actual step.
        posterior = self.solver.step_fn(
            state=state.accepted,
            vector_field=vector_field,
            dt=self.control.extract_dt_from_state(state_control),
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
        state_control = self.control.apply(
            state=state_control,
            error_normalised=error_normalised,
            error_contraction_rate=self.error_contraction_rate,
        )
        return _AdaptiveState(
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
            error_norm_proposed=state.error_norm_proposed,
            proposed=_inf_like(state.proposed),
            accepted=accepted,  # holla! New! :)
            solution=solution,  # holla! New! :)
            previous=previous,  # holla! New! :)
            control=state.control,
        )

    def extract_fn(self, state: _AdaptiveState[S, C], /) -> S:
        solver_extract = self.solver.extract_fn(state.solution)
        control_extract = self.control.extract_dt_from_state(state.control)

        # return BOTH dt & solver_extract.
        #  Usually, only the latter is necessary.
        #  but we return both because this way, extract is inverse to init,
        #  and it becomes much easier to restart the solver at a new point
        #  without losing consistency.
        return control_extract, solver_extract

    def extract_terminal_value_fn(self, state: _AdaptiveState[S, C], /) -> S:
        solver_extract = self.solver.extract_terminal_value_fn(state.solution)
        control_extract = self.control.extract_dt_from_state(state.control)
        return control_extract, solver_extract


def _inf_like(tree):
    return jax.tree_map(lambda x: jnp.inf * jnp.ones_like(x), tree)
