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
    def init_fn(self, *, ivp):
        """Initialise the IVP solver state."""
        raise NotImplementedError

    @abc.abstractmethod
    def step_fn(self, state, *, vector_field, dt0):
        """Perform a step."""
        raise NotImplementedError


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

        stepping: Any  # must contain fields "t" and "u".
        control: Any  # must contain field "scale_factor".

    def init_fn(self, *, ivp):
        """Initialise the IVP solver state."""
        state_stepping = self.stepping.init_fn(ivp=ivp)
        state_control = self.control.init_fn()

        error_normalised = self._normalise_error(
            error_estimate=state_stepping.error_estimate,
            u=state_stepping.u,
            atol=self.atol,
            rtol=self.rtol,
            norm_ord=self.norm_ord,
        )
        dt_proposed = self._propose_first_dt_per_tol(
            f=lambda *x: ivp.vector_field(*x, ivp.t0, *ivp.parameters),
            u0=ivp.initial_values,
            error_order=self.error_order,
            atol=self.atol,
            rtol=self.rtol,
        )
        return self.State(
            dt_proposed=dt_proposed,
            error_normalised=error_normalised,
            stepping=state_stepping,
            control=state_control,
        )

    def step_fn(self, *, state, vector_field, dt0):
        """Perform a step."""

        def cond_fn(x):
            proceed_iteration, _ = x
            return proceed_iteration

        def body_fn(x):
            _, s = x
            s = self.attempt_step_fn(state=s, vector_field=vector_field, dt0=dt0)
            proceed_iteration = s.error_normalised > 1.0
            return proceed_iteration, s

        def init_fn(s):
            return True, s

        init_val = init_fn(state)
        _, state_new = jax.lax.while_loop(cond_fn, body_fn, init_val)
        return state_new

    def attempt_step_fn(self, *, state, vector_field, dt0):
        """Perform a step with an IVP solver and \
        propose a future time-step based on tolerances and error estimates."""
        state_stepping = self.stepping.step_fn(
            state=state.stepping, vector_field=vector_field, dt0=dt0
        )
        error_normalised = self._normalise_error(
            error_estimate=state_stepping.error_estimate,
            u=state_stepping.u,
            atol=self.atol,
            rtol=self.rtol,
            norm_ord=self.norm_ord,
        )
        state_control = self.control.control_fn(
            state=state.control,
            error_normalised=error_normalised,
            error_order=self.error_order,
        )
        dt_proposed = dt0 * state_control.scale_factor
        return self.State(
            dt_proposed=dt_proposed,
            error_normalised=error_normalised,
            stepping=state_stepping,
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
        f0 = f(u0)
        scale = atol + u0 * rtol
        a = jnp.linalg.norm(u0 / scale)
        b = jnp.linalg.norm(f0 / scale)
        dt0 = jnp.where((a < 1e-5) | (b < 1e-5), 1e-6, 0.01 * a / b)

        u1 = u0 + dt0 * f0
        f1 = f(u1)
        c = jnp.linalg.norm((f1 - f0) / scale) / dt0
        dt1 = jnp.where(
            (b <= 1e-15) & (c <= 1e-15),
            jnp.maximum(1e-6, dt0 * 1e-3),
            (0.01 / jnp.max(b + c)) ** (1.0 / error_order),
        )
        return jnp.minimum(100.0 * dt0, dt1)
