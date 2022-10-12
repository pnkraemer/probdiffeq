"""Initial value problem solvers."""
from functools import partial
from typing import Any, Generic, TypeVar, Union

import equinox as eqx
import jax.lax
import jax.numpy as jnp
import jax.tree_util

T = TypeVar("T")
"""A generic ODE Filter state."""


class AdaptiveODEFilterState(Generic[T], eqx.Module):
    """Solver state."""

    dt_proposed: float
    error_norm_proposed: float

    control: Any  # must contain field "scale_factor".
    """Controller."""

    # All sorts of solution objects.
    # Maybe we can simplify here. But not yet.
    # Edit: I am pretty sure one of them can go.
    # (I am looking at you, "previous".)

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


class AdaptiveODEFilter(eqx.Module):
    """Make an adaptive ODE filter."""

    strategy: Any

    atol: float
    rtol: float

    error_order: int
    control: Any
    norm_ord: Union[int, str, None] = None

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

    @partial(jax.jit, static_argnames=("vector_field",))
    def step_fn(self, *, state, vector_field, t1):
        """Perform a step."""
        # Part I/II: accept-reject loop unless we have gathered ODE
        # evaluations _beyond_ t1.

        enter_accept_reject_loop = state.accepted.t < t1

        def true_fn1(s):
            return self._accept_reject_loop(state0=s, vector_field=vector_field)

        def false_fn1(s):
            return s

        state_new = jax.lax.cond(enter_accept_reject_loop, true_fn1, false_fn1, state)

        # Part II/II: If we have previously overstepped
        # the boundary, interpolate to t1

        interpolate = state_new.accepted.t > t1

        def true_fn(s):
            return self._interpolate(state=s, t=t1)

        def false_fn(s):
            return s

        return jax.lax.cond(interpolate, true_fn, false_fn, state_new)

    def _accept_reject_loop(self, *, vector_field, state0):
        def cond_fn(x):
            proceed_iteration, _ = x
            return proceed_iteration

        def body_fn(x):
            _, s = x
            s = self._attempt_step_fn(state=s, vector_field=vector_field)
            proceed_iteration = s.error_norm_proposed > 1.0
            return proceed_iteration, s

        def init_fn(s):
            return True, s

        _, state_new = jax.lax.while_loop(cond_fn, body_fn, init_fn(state0))

        return AdaptiveODEFilterState(
            dt_proposed=state_new.dt_proposed,
            error_norm_proposed=state_new.error_norm_proposed,
            proposed=state_new.proposed,
            accepted=state_new.proposed,  # holla! New! :)
            solution=state_new.accepted,  # Overwritten by interpolate() if necessary
            previous=state0.accepted,  # holla! New! :)
            control=state_new.control,
        )

    def _attempt_step_fn(self, *, state, vector_field):
        """Perform a step with an IVP solver and \
        propose a future time-step based on tolerances and error estimates."""

        def vf(*y):  # todo: this should not happen here?!
            return vector_field(state.accepted.t + state.dt_proposed, *y)

        posterior, error_estimate = self.strategy.step_fn(
            state=state.accepted, vector_field=vf, dt=state.dt_proposed
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

    def _interpolate(self, *, state, t):
        accepted_new, interpolated = self.strategy.interpolate_fn(
            s0=state.previous, s1=state.accepted, t=t
        )
        return AdaptiveODEFilterState(
            dt_proposed=state.dt_proposed,
            error_norm_proposed=state.error_norm_proposed,
            proposed=state.proposed,
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
