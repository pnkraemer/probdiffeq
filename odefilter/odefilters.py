"""Initial value problem solvers."""
from dataclasses import dataclass
from functools import partial  # noqa: F401
from typing import Any, Generic, TypeVar, Union

import jax.lax
import jax.numpy as jnp
import jax.tree_util
from jax.tree_util import register_pytree_node_class

T = TypeVar("T")
"""A generic ODE Filter state."""


@register_pytree_node_class
@dataclass(frozen=True)
class AdaptiveODEFilterState(Generic[T]):
    """Solver state."""

    dt_proposed: float
    error_norm_proposed: float

    control: Any  # must contain field "scale_factor".
    """Controller."""

    # All sorts of solution objects.
    # todo: sort according to previous <= solution <= accepted <= proposed

    solution: T
    """The current best solution.

    Sometimes equal to :attr:`accepted`, but not always.
    Not equal to :attr:`accepted` if the solution has been obtained
    by overstepping a boundary and subsequent interpolation.
    In this case, it equals :attr:`previous`
    (which has been the interpolation result).
    """

    proposed: T
    """The most recent proposal."""

    accepted: T
    """The most recent accepted state."""

    previous: T
    """The penultimate (2nd most recent) accepted state. Needed for interpolation.
    """

    def tree_flatten(self):
        children = (
            self.dt_proposed,
            self.error_norm_proposed,
            self.control,
            self.solution,
            self.proposed,
            self.accepted,
            self.previous,
        )
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)


@register_pytree_node_class
@dataclass(frozen=True)
class AdaptiveODEFilter:
    """Make an adaptive ODE filter."""

    strategy: Any

    atol: float
    rtol: float

    control: Any
    norm_ord: Union[int, str, None] = None

    numerical_zero: float = 1e-10
    """Assume we reached the checkpoint if the distance of the current \
     state to the checkpoint is smaller than this value."""

    def tree_flatten(self):
        children = (
            self.strategy,
            self.atol,
            self.rtol,
            self.control,
            self.numerical_zero,
        )
        aux = self.norm_ord
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        strategy, atol, rtol, control, numerical_zero = children
        norm_ord = aux
        return cls(
            strategy=strategy,
            atol=atol,
            rtol=rtol,
            control=control,
            numerical_zero=numerical_zero,
            norm_ord=norm_ord,
        )

    @property
    def error_order(self):
        """Error order."""
        return self.strategy.implementation.num_derivatives + 1

    @jax.jit
    def init_fn(self, *, taylor_coefficients, t0):
        """Initialise the IVP solver state."""
        # Initialise the components
        posterior, error_estimate = self.strategy.init_fn(
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

    @partial(jax.jit, static_argnames=["info_op"])
    def step_fn(self, state, info_op, t1):
        """Perform a step."""
        enter_rejection_loop = state.accepted.t < t1
        state = jax.lax.cond(
            enter_rejection_loop,
            lambda s: self._rejection_loop(state0=s, info_op=info_op),
            lambda s: s,
            state,
        )

        enter_interpolation = state.accepted.t >= t1
        return jax.lax.cond(
            enter_interpolation,
            lambda s: self._interpolate(state=s, t=t1),
            lambda s: s,
            state,
        )

    def _rejection_loop(self, *, info_op, state0):
        def cond_fn(x):
            proceed_iteration, _ = x
            return proceed_iteration

        def body_fn(x):
            _, s = x
            s = self._attempt_step_fn(state=s, info_op=info_op)
            proceed_iteration = s.error_norm_proposed > 1.0
            return proceed_iteration, s

        def init_fn(s):
            return True, s

        _, state_new = jax.lax.while_loop(cond_fn, body_fn, init_fn(state0))

        solution = state_new.proposed
        accepted = state_new.proposed

        return AdaptiveODEFilterState(
            dt_proposed=state_new.dt_proposed,
            error_norm_proposed=state_new.error_norm_proposed,
            proposed=_nan_like(state_new.proposed),  # meaningless?
            accepted=accepted,  # holla! New! :)
            solution=solution,  # Overwritten by interpolate() if necessary
            previous=state0.accepted,  # holla! New! :)
            control=state_new.control,
        )

    def _attempt_step_fn(self, *, state, info_op):
        """Perform a step with an IVP solver and \
        propose a future time-step based on tolerances and error estimates."""
        posterior, error_estimate = self.strategy.step_fn(
            state=state.accepted, info_op=info_op, dt=state.dt_proposed
        )

        error_normalised = self._normalise_error(
            error_estimate=error_estimate,
            u=posterior.u,
            atol=self.atol,
            rtol=self.rtol,
            norm_ord=self.norm_ord,
        )
        state_control = self.control.control_fn(
            state=state.control,
            error_normalised=error_normalised,
            error_order=self.error_order,
        )
        dt_proposed = state.dt_proposed * state_control.scale_factor

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
        return jnp.linalg.norm(error_relative, ord=norm_ord)

    #
    # def _reset_at_checkpoint_fn(self, *, state, t1):
    #     new_accepted, new_solution = self.strategy.reset_at_checkpoint_fn(
    #         accepted=state.accepted, solution=state.solution, t1=t1
    #     )
    #     return AdaptiveODEFilterState(
    #         dt_proposed=state.dt_proposed,
    #         error_norm_proposed=state.error_norm_proposed,
    #         accepted=new_accepted,
    #         solution=new_solution,
    #         proposed=_nan_like(new_accepted),  # irrelevant?
    #         previous=_nan_like(new_accepted),  # irrelevant?
    #         control=state.control,
    #     )

    def _interpolate(self, *, state, t):
        accepted_new, interpolated = self.strategy.interpolate_fn(
            s0=state.previous, s1=state.accepted, t=t
        )

        # Either one of those two...
        # # new_previous = interpolated
        # new_previous, _ = self.strategy.reset_at_checkpoint_fn(
        #     accepted=interpolated, solution=interpolated, t1=t
        # )

        return AdaptiveODEFilterState(
            dt_proposed=state.dt_proposed,
            error_norm_proposed=state.error_norm_proposed,
            proposed=_nan_like(state.proposed),
            accepted=accepted_new,  # holla! New! :)
            solution=interpolated,  # holla! New! :)
            previous=interpolated,  # holla! New! :)
            control=state.control,
        )

    @jax.jit
    def extract_fn(self, *, state):  # noqa: D102
        return self.strategy.extract_fn(state=state.solution)


def _empty_like(tree):
    return jax.tree_util.tree_map(jnp.nan * jnp.ones_like, tree)


def _nan_like(tree):
    return jax.tree_map(lambda x: jnp.nan * jnp.ones_like(x), tree)
