"""Initial value problem solvers."""
import abc
from typing import Any, Union

import equinox as eqx
import jax.lax
import jax.numpy as jnp
import jax.tree_util


class AbstractIVPSolver(eqx.Module, abc.ABC):
    """Abstract solver for IVPs."""

    @abc.abstractmethod
    def init_fn(self, *, vector_field, initial_values, t0):
        """Initialise the IVP solver state."""
        raise NotImplementedError

    @abc.abstractmethod
    def step_fn(self, state, *, vector_field, dt0):
        """Perform a step."""
        raise NotImplementedError

    @abc.abstractmethod
    def reset_fn(self, state):
        """Reset the solver state."""
        raise NotImplementedError


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# warning: rejected steps do not have their "stepping" attribute reset!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


class Adaptive(AbstractIVPSolver):
    """Make an adaptive ODE solver."""

    # Take a solver, normalise its error estimate,
    # and propose time-steps based on tolerances.

    atol: float
    rtol: float

    error_order: int
    stepping: Any
    control: Any
    norm_ord: Union[int, str, None] = None

    class State(eqx.Module):
        """Solver state."""

        dt_proposed: float
        error_normalised: float

        proposed: Any  # must contain fields "t" and "u".
        accepted: Any  # must contain fields "t" and "u".

        control: Any  # must contain field "scale_factor".

    def init_fn(self, *, vector_field, initial_values, t0):
        """Initialise the IVP solver state."""
        state_stepping = self.stepping.init_fn(
            vector_field=vector_field, initial_values=initial_values, t0=t0
        )
        state_control = self.control.init_fn()

        error_normalised = self._normalise_error(
            error_estimate=state_stepping.error_estimate,
            u=state_stepping.u,
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
        return self.State(
            dt_proposed=dt_proposed,
            error_normalised=error_normalised,
            proposed=state_stepping,
            accepted=state_stepping,
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
        return self.State(
            dt_proposed=state_new.dt_proposed,
            error_normalised=state_new.error_normalised,
            proposed=state_new.proposed,
            accepted=state_new.proposed,  # holla! New! :)
            control=state_new.control,
        )

    def attempt_step_fn(self, *, state, vector_field, t1):
        """Perform a step with an IVP solver and \
        propose a future time-step based on tolerances and error estimates."""
        # todo: should this be at the end of this function?
        #  or even happen inside the controller?
        dt_proposed = jnp.minimum(t1 - state.accepted.t, state.dt_proposed)

        state_proposed = self.stepping.step_fn(
            state=state.accepted, vector_field=vector_field, dt0=dt_proposed
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
        return self.State(
            dt_proposed=dt_proposed,
            error_normalised=error_normalised,
            proposed=state_proposed,
            accepted=state.accepted,  # too early to accept :)
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

    def reset_fn(self, *, state):
        return self.State(
            dt_proposed=state.dt_proposed,
            error_normalised=state.error_normalised,
            proposed=state.proposed,  # reset this one too?
            accepted=self.stepping.reset_fn(state=state.accepted),
            control=state.control,  # reset this one too?
        )
