"""Adaptive solvers for initial value problems (IVPs)."""
from typing import Any, Generic, TypeVar

import jax
import jax.numpy as jnp

from probdiffeq import controls, ivpsolvers
from probdiffeq.backend import containers


# we could make this generic, but any path there would involve
# custom pytree-registration, which needs a lot of code for such a simple object.
class _AdaptiveState(containers.NamedTuple):
    error_norm_proposed: float
    control: Any
    proposed: Any
    accepted: Any
    solution: Any
    previous: Any


def _reference_state_fn_max_abs(sol, sol_previous):
    return jnp.maximum(jnp.abs(sol), jnp.abs(sol_previous))


T = TypeVar("T", bound=ivpsolvers.Solver)
"""A type-variable for (non-adaptive) IVP solvers."""


# todo: this is the only object in this module
#  (and I cannot imagine another one coming in the near future)
#  so why do we need a class in the first place?


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

    @jax.jit
    def init(self, t, posterior, output_scale, num_steps, dt0):
        """Initialise the IVP solver state."""
        # Initialise the components
        state_solver = self.solver.init(t, posterior, output_scale, num_steps)
        state_control = self.control.init(dt0)

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
    def step(self, state, vector_field, t1, parameters):
        """Perform a full step (including acceptance/rejection)."""
        enter_rejection_loop = state.accepted.t + self.numerical_zero < t1
        state = jax.lax.cond(
            enter_rejection_loop,
            lambda s: self._rejection_loop(
                state0=s,
                vector_field=vector_field,
                t1=t1,
                parameters=parameters,
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
            s = self._attempt_step(
                state=s,
                vector_field=vector_field,
                t1=t1,
                parameters=parameters,
            )
            proceed_iteration = s.error_norm_proposed > 1.0
            return proceed_iteration, s

        def init(s):
            return True, s

        _, state_new = self.while_loop_fn(cond_fn, body_fn, init(state0))
        return _AdaptiveState(
            error_norm_proposed=_inf_like(state_new.error_norm_proposed),
            proposed=_inf_like(state_new.proposed),  # meaningless?
            accepted=state_new.proposed,  # holla! New! :)
            solution=state_new.proposed,  # Overwritten by interpolate() if necessary
            previous=state0.accepted,  # holla! New! :)
            control=state_new.control,
        )

    def _attempt_step(self, *, state, vector_field, t1, parameters):
        """Perform a step with an IVP solver and \
        propose a future time-step based on tolerances and error estimates."""
        # Some controllers like to clip the terminal value instead of interpolating.
        # This must happen _before_ the step.
        state_control = self.control.clip(state.control, t=state.accepted.t, t1=t1)

        # Perform the actual step.
        state_proposed = self.solver.step(
            state=state.accepted,
            vector_field=vector_field,
            dt=self.control.extract(state_control),
            parameters=parameters,
        )
        # Normalise the error and propose a new step.
        error_normalised = self._normalise_error(
            error_estimate=state_proposed.error_estimate,
            u=self.reference_state_fn(state_proposed.u, state.accepted.u),
            atol=self.atol,
            rtol=self.rtol,
            norm_ord=self.norm_ord,
        )
        error_contraction_rate = self.solver.strategy.extrapolation.num_derivatives + 1
        state_control = self.control.apply(
            state_control,
            error_normalised=error_normalised,
            error_contraction_rate=error_contraction_rate,
        )
        return _AdaptiveState(
            error_norm_proposed=error_normalised,  # new
            proposed=state_proposed,  # new
            control=state_control,  # new
            solution=state.solution,  # too early to accept
            accepted=state.accepted,  # too early to accept
            previous=state.previous,  # too early to accept
        )

    @staticmethod
    def _normalise_error(*, error_estimate, u, atol, rtol, norm_ord):
        error_relative = error_estimate / (atol + rtol * jnp.abs(u))
        dim = jnp.atleast_1d(u).size
        return jnp.linalg.norm(error_relative, ord=norm_ord) / jnp.sqrt(dim)

    def _interpolate(self, *, state: _AdaptiveState, t) -> _AdaptiveState:
        accepted, solution, previous = self.solver.interpolate(
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

    def extract(self, state: _AdaptiveState, /):
        solver_extract = self.solver.extract(state.solution)
        control_extract = self.control.extract(state.control)
        return solver_extract, control_extract


# Register outside of class to declutter the AdaptiveIVPSolver source code a bit


def _asolver_flatten(asolver: AdaptiveIVPSolver):
    children = (
        asolver.solver,
        asolver.atol,
        asolver.rtol,
        asolver.control,
        asolver.numerical_zero,
    )
    aux = asolver.norm_ord, asolver.reference_state_fn, asolver.while_loop_fn
    return children, aux


def _asolver_unflatten(aux, children):
    solver, atol, rtol, control, numerical_zero = children
    norm_ord, reference_state_fn, while_loop_fn = aux
    return AdaptiveIVPSolver(
        solver=solver,
        while_loop_fn=while_loop_fn,
        atol=atol,
        rtol=rtol,
        control=control,
        numerical_zero=numerical_zero,
        norm_ord=norm_ord,
        reference_state_fn=reference_state_fn,
    )


jax.tree_util.register_pytree_node(
    nodetype=AdaptiveIVPSolver,
    flatten_func=_asolver_flatten,
    unflatten_func=_asolver_unflatten,
)


def _inf_like(tree):
    return jax.tree_map(lambda x: jnp.inf * jnp.ones_like(x), tree)
