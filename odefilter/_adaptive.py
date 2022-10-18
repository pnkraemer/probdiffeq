"""Adaptive solvers."""
from dataclasses import dataclass
from functools import partial  # noqa: F401
from typing import Any, Generic, TypeVar, Union

import jax.lax
import jax.numpy as jnp
import jax.tree_util
from jax.tree_util import register_pytree_node_class

from odefilter import controls

T = TypeVar("T")
"""A generic ODE Filter state."""
R = TypeVar("R")
"""A generic ODE Filter strategy."""


@register_pytree_node_class
@dataclass(frozen=True)
class AdaptiveODEFilterState(Generic[T]):
    """Solver state."""

    dt_proposed: float
    error_norm_proposed: float

    control: Any  # must contain field "scale_factor".
    """Controller state."""

    # All sorts of solutions types.
    # previous.t <= solution.t <= accepted.t <= proposed.t

    proposed: T
    """The most recent proposal."""

    accepted: T
    """The most recent accepted state."""

    solution: T
    """The current best solution.

    Sometimes equal to :attr:`accepted`, but not always.
    Not equal to :attr:`accepted` if the solution has been obtained
    by overstepping a boundary and subsequent interpolation.
    In this case, it equals :attr:`previous`
    (which has been the interpolation result).
    """

    previous: T
    """The penultimate (2nd most recent) accepted state. Needed for interpolation.
    """

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


@register_pytree_node_class
@dataclass(frozen=True)
class AdaptiveODEFilter(Generic[R]):
    """Make an adaptive ODE filter."""

    solver: R

    atol: float = 1e-4
    rtol: float = 1e-2

    control: Any = controls.ProportionalIntegral()
    norm_ord: Union[int, str, None] = None

    numerical_zero: float = 1e-10
    """Assume we reached the checkpoint if the distance of the current \
     state to the checkpoint is smaller than this value."""

    def tree_flatten(self):
        children = (
            self.solver,
            self.atol,
            self.rtol,
            self.control,
            self.numerical_zero,
        )
        aux = self.norm_ord
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        solver, atol, rtol, control, numerical_zero = children
        norm_ord = aux
        return cls(
            solver=solver,
            atol=atol,
            rtol=rtol,
            control=control,
            numerical_zero=numerical_zero,
            norm_ord=norm_ord,
        )

    @property
    def error_contraction_rate(self):
        """Error order."""
        return self.solver.implementation.num_derivatives + 1

    @jax.jit
    def init_fn(self, *, taylor_coefficients, t0):
        """Initialise the IVP solver state."""
        # Initialise the components
        posterior, error_estimate = self.solver.init_fn(
            taylor_coefficients=taylor_coefficients, t0=t0
        )
        state_control = self.control.init_fn()

        # Initialise (prototypes for) proposed values
        u0, f0, *_ = taylor_coefficients
        error_norm_proposed = self._normalise_error(
            error_estimate=error_estimate,
            u=u0,
            atol=self.atol,
            rtol=self.rtol,
            norm_ord=self.norm_ord,
        )
        dt_proposed = self._propose_first_dt(taylor_coefficients=taylor_coefficients)
        return AdaptiveODEFilterState(
            dt_proposed=dt_proposed,
            error_norm_proposed=error_norm_proposed,
            solution=posterior,
            proposed=posterior,
            accepted=posterior,
            previous=posterior,
            control=state_control,
        )

    @staticmethod
    def _propose_first_dt(*, taylor_coefficients, scale=0.01):
        u0, f0, *_ = taylor_coefficients
        norm_y0 = jnp.linalg.norm(u0)
        norm_dy0 = jnp.linalg.norm(f0)
        return scale * norm_y0 / norm_dy0

    @jax.jit
    def step_fn(self, state, info_op, t1):
        """Perform a step."""
        enter_rejection_loop = state.accepted.t + self.numerical_zero < t1
        state = jax.lax.cond(
            enter_rejection_loop,
            lambda s: self._rejection_loop(state0=s, info_op=info_op, t1=t1),
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

    def _rejection_loop(self, *, info_op, state0, t1):
        def cond_fn(x):
            proceed_iteration, _ = x
            return proceed_iteration

        def body_fn(x):
            _, s = x
            s = self._attempt_step_fn(state=s, info_op=info_op, t1=t1)
            proceed_iteration = s.error_norm_proposed > 1.0
            return proceed_iteration, s

        def init_fn(s):
            return True, s

        _, state_new = jax.lax.while_loop(cond_fn, body_fn, init_fn(state0))
        return AdaptiveODEFilterState(
            dt_proposed=state_new.dt_proposed,
            error_norm_proposed=_inf_like(state_new.error_norm_proposed),
            proposed=_inf_like(state_new.proposed),  # meaningless?
            accepted=state_new.proposed,  # holla! New! :)
            solution=state_new.proposed,  # Overwritten by interpolate() if necessary
            previous=state0.accepted,  # holla! New! :)
            control=state_new.control,
        )

    def _attempt_step_fn(self, *, state, info_op, t1):
        """Perform a step with an IVP solver and \
        propose a future time-step based on tolerances and error estimates."""
        # Some controllers like to clip the terminal value instead of interpolating.
        # This must happen _before_ the step.
        dt = self.control.clip_fn(state=state.accepted, dt=state.dt_proposed, t1=t1)

        # Perform the actual step.
        posterior, error_estimate = self.solver.step_fn(
            state=state.accepted, info_op=info_op, dt=dt
        )

        # Normalise the error and propose a new step.
        error_normalised = self._normalise_error(
            error_estimate=error_estimate,
            u=jnp.abs(posterior.u),
            # todo: allow a switch to
            #  u=jnp.maximum(jnp.abs(posterior.u), jnp.abs(state.accepted.u)),
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
        return AdaptiveODEFilterState(
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

    def _interpolate(self, *, state, t):
        accepted, solution, previous = self.solver.interpolate_fn(
            s0=state.previous, s1=state.accepted, t=t
        )
        return AdaptiveODEFilterState(
            dt_proposed=state.dt_proposed,
            error_norm_proposed=state.error_norm_proposed,
            proposed=_inf_like(state.proposed),
            accepted=accepted,  # holla! New! :)
            solution=solution,  # holla! New! :)
            previous=previous,  # holla! New! :)
            control=state.control,
        )

    @jax.jit
    def extract_fn(self, *, state):  # noqa: D102
        return self.solver.extract_fn(state=state.solution)


def _empty_like(tree):
    return jax.tree_util.tree_map(jnp.empty_like, tree)


def _nan_like(tree):
    return jax.tree_map(lambda x: jnp.nan * jnp.ones_like(x), tree)


def _inf_like(tree):
    return jax.tree_map(lambda x: jnp.inf * jnp.ones_like(x), tree)
