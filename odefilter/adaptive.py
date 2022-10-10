"""Initial value problem solvers."""
from typing import Any, Generic, TypeVar, Union

import equinox as eqx
import jax.lax
import jax.numpy as jnp
import jax.tree_util

T = TypeVar("T")
"""A generic ODE Filter state."""


class AdaptiveSolverState(Generic[T], eqx.Module):
    """Solver state."""

    dt_proposed: float
    error_normalised: float
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

    proposed: T  # must contain fields "t" and "u".
    """The most recent proposal."""

    accepted: T  # must contain fields "t" and "u".
    """The most recent accepted state."""

    previous: T  # must contain fields "t" and "u".
    """The penultimate (2nd most recent) accepted state."""


class Adaptive(eqx.Module):
    """Make an adaptive ODE solver."""

    # Take a solver, normalise its error estimate,
    # and propose time-steps based on tolerances.

    atol: float
    rtol: float

    error_order: int
    odefilter: Any
    control: Any
    norm_ord: Union[int, str, None] = None

    def init_fn(self, *, vector_field, initial_values, t0):
        """Initialise the IVP solver state."""
        state_odefilter = self.odefilter.init_fn(
            vector_field=vector_field, initial_values=initial_values, t0=t0
        )
        state_control = self.control.init_fn()

        error_normalised = self._normalise_error(
            error_estimate=state_odefilter.error_estimate,
            u=state_odefilter.u,
            atol=self.atol,
            rtol=self.rtol,
            norm_ord=self.norm_ord,
        )
        dt_proposed = self._propose_first_dt_per_tol(
            f=lambda *x: vector_field(*x, t=t0),
            u0=initial_values,
            error_order=self.error_order,
            atol=self.atol,
            rtol=self.rtol,
        )
        return AdaptiveSolverState(
            dt_proposed=dt_proposed,
            error_normalised=error_normalised,
            solution=state_odefilter,
            proposed=state_odefilter,
            accepted=state_odefilter,
            previous=state_odefilter,
            control=state_control,
        )

    def step_fn(self, *, state, vector_field, t1):
        """Perform a step."""

        def cond_fn(x):
            proceed_iteration, _ = x
            return proceed_iteration

        def body_fn(x):
            _, s = x
            s = self.attempt_step_fn(state=s, vector_field=vector_field, t1=t1)
            proceed_iteration = s.error_normalised > 1.0
            return proceed_iteration, s

        def init_fn(s):
            return True, s

        init_val = init_fn(state)
        _, state_new = jax.lax.while_loop(cond_fn, body_fn, init_val)
        return AdaptiveSolverState(
            dt_proposed=state_new.dt_proposed,
            error_normalised=state_new.error_normalised,
            proposed=state_new.proposed,
            solution=state_new.proposed,  # holla! New! :)
            accepted=state_new.proposed,  # holla! New! :)
            previous=state_new.accepted,  # holla! New! :)
            control=state_new.control,
        )

    def attempt_step_fn(self, *, state, vector_field, t1):
        """Perform a step with an IVP solver and \
        propose a future time-step based on tolerances and error estimates."""
        # todo: should this be at the end of this function?
        #  or even happen inside the controller?

        state_proposed = self.odefilter.step_fn(
            state=state.accepted, vector_field=vector_field, dt0=state.dt_proposed
        )
        error_normalised = self._normalise_error(
            error_estimate=state_proposed.error_estimate,
            u=state_proposed.u,
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
        return AdaptiveSolverState(
            dt_proposed=dt_proposed,
            error_normalised=error_normalised,
            proposed=state_proposed,
            solution=state.solution,  # too early to accept :)
            accepted=state.accepted,  # too early to accept :)
            previous=state.previous,  # too early to accept :)
            control=state_control,
        )

    @staticmethod
    def _normalise_error(*, error_estimate, u, atol, rtol, norm_ord):
        error_relative = error_estimate / (atol + rtol * jnp.abs(u))
        return jnp.linalg.norm(error_relative, ord=norm_ord)

    @staticmethod
    def _propose_first_dt_per_tol(*, f, u0, error_order, rtol, atol):
        # Taken from:
        # https://github.com/google/jax/blob/main/jax/experimental/ode.py
        #
        # which uses the algorithm from
        #
        # E. Hairer, S. P. Norsett G. Wanner,
        # Solving Ordinary Differential Equations I: Nonstiff Problems, Sec. II.4.
        assert len(u0) == 1
        f0 = f(*u0)
        scale = atol + u0[0] * rtol
        a = jnp.linalg.norm(u0[0] / scale)
        b = jnp.linalg.norm(f0 / scale)
        dt0 = jnp.where((a < 1e-5) | (b < 1e-5), 1e-6, 0.01 * a / b)

        u1 = u0[0] + dt0 * f0
        f1 = f(u1)
        c = jnp.linalg.norm((f1 - f0) / scale) / dt0
        dt1 = jnp.where(
            (b <= 1e-15) & (c <= 1e-15),
            jnp.maximum(1e-6, dt0 * 1e-3),
            (0.01 / jnp.max(b + c)) ** (1.0 / error_order),
        )
        return jnp.minimum(100.0 * dt0, dt1)

    def reset_fn(self, *, state):  # noqa: D102
        return AdaptiveSolverState(
            dt_proposed=state.dt_proposed,
            error_normalised=state.error_normalised,
            solution=state.solution,  # reset this one too?????
            proposed=state.proposed,  # reset this one too?
            accepted=self.odefilter.reset_fn(state=state.accepted),
            previous=state.previous,
            control=state.control,  # reset this one too?
        )

    def extract_fn(self, *, state):  # noqa: D102
        return self.odefilter.extract_fn(state=state.solution)

    def interpolate_fn(self, *, state, t):  # noqa: D102
        """Interpolate between state.recent and state.accepted.

        t must be in between state.recent.t and state.accepted.t
        """
        # todo: the time-points seem to be inappropriately assigned
        accepted_new, interpolated = self.odefilter.interpolate_fn(
            state0=state.previous, state1=state.accepted, t=t
        )
        return AdaptiveSolverState(
            dt_proposed=state.dt_proposed,
            error_normalised=state.error_normalised,
            proposed=state.proposed,  # reset this one too?
            accepted=accepted_new,
            solution=interpolated,
            previous=interpolated,
            control=state.control,  # reset this one too?
        )


def _empty_like(tree):
    return jax.tree_util.tree_map(jnp.empty_like, tree)
