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

    @partial(jax.jit, static_argnames=("vector_field",))
    def step_fn(self, *, state, vector_field, t1):
        """Perform a step."""

        def cond_fn(x):
            proceed_iteration, _ = x
            return proceed_iteration

        def body_fn(x):
            _, s = x
            s = self.attempt_step_fn(state=s, vector_field=vector_field, t1=t1)
            proceed_iteration = s.error_norm_proposed > 1.0
            return proceed_iteration, s

        def init_fn(s):
            return True, s

        init_val = init_fn(state)
        _proceed, state_new = jax.lax.while_loop(cond_fn, body_fn, init_val)
        return AdaptiveODEFilterState(
            dt_proposed=state_new.dt_proposed,
            error_norm_proposed=state_new.error_norm_proposed,
            proposed=state_new.proposed,  # meaningless
            solution=state_new.proposed,  # holla! New! :)
            accepted=state_new.proposed,  # holla! New! :)
            previous=state_new.accepted,  # holla! New! :)
            control=state_new.control,
        )

    def attempt_step_fn(self, *, state, vector_field, t1):
        """Perform a step with an IVP solver and \
        propose a future time-step based on tolerances and error estimates."""

        def vf(*y):  # todo: this should not happen here?!
            return vector_field(*y, t=state.accepted.t + state.dt_proposed)

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

    @staticmethod
    def _propose_first_dt(*, taylor_coefficients, scale=0.01):
        u0, f0, *_ = taylor_coefficients
        norm_y0 = jnp.linalg.norm(u0)
        norm_dy0 = jnp.linalg.norm(f0)
        return scale * norm_y0 / norm_dy0

    @jax.jit
    def extract_fn(self, *, state):  # noqa: D102
        posterior_new = self.strategy.extract_fn(state=state.solution)
        return posterior_new

    @jax.jit
    def interpolate_fn(self, *, state, t):  # noqa: D102
        """Interpolate between state.recent and state.accepted.

        t must be in between state.recent.t and state.accepted.t
        """
        # todo: the time-points seem to be inappropriately assigned
        accepted_new, interpolated = self.strategy.interpolate_fn(
            s0=state.previous,
            s1=state.accepted,
            t=t,
            t0=state.previous.t,
            t1=state.accepted.t,
        )
        return AdaptiveODEFilterState(
            dt_proposed=state.dt_proposed,
            error_norm_proposed=state.error_norm_proposed,
            proposed=state.proposed,
            accepted=accepted_new,
            solution=interpolated,
            previous=interpolated,
            control=state.control,
        )


def _empty_like(tree):
    return jax.tree_util.tree_map(jnp.nan * jnp.ones_like, tree)
