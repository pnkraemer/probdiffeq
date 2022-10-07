"""Initial value problem solvers."""
import abc
from typing import Any, NamedTuple, Union

import jax.lax
import jax.numpy as jnp
import jax.tree_util

from odefilter import sqrtm
from odefilter.prob import ibm, rv


class AbstractIVPSolver(abc.ABC):
    """Abstract solver for IVPs."""

    @abc.abstractmethod
    def init_fn(self, *, ivp, params):
        """Initialise the IVP solver state."""
        raise NotImplementedError

    @abc.abstractmethod
    def step_fn(self, state, *, ode_function, dt0, params):
        """Perform a step."""
        raise NotImplementedError


def ek0_non_adaptive(*, derivative_init_fn, num_derivatives, information_fn):
    """EK0 solver."""
    alg = _NonAdaptiveEK0(
        derivative_init_fn=derivative_init_fn,
        information_fn=information_fn,
        num_derivatives=num_derivatives,
    )

    a, q_sqrtm = ibm.system_matrices_1d(num_derivatives=num_derivatives)
    params = _NonAdaptiveEK0.Params(a=a, q_sqrtm_upper=q_sqrtm.T)
    return alg, params


# todo: this is not really an EK0, it is a KroneckerODEFilter
#  with an EK0 information operator
#  Turn this into something like
#  _KroneckerSolver(
#   filter_implementation=
#   EvaluateExtrapolateTimeVaryingDiffusion(information_fn=information_fn)
#  )
class _NonAdaptiveEK0(AbstractIVPSolver):
    """EK0."""

    class State(NamedTuple):

        # Mandatory
        t: float
        u: Any
        error_estimate: Any

        # split this into something like
        # extrapolated, observed, corrected, and linearised model?
        # Is this its own data structure?

        rv_extrapolated: rv.Normal
        rv_corrected: rv.Normal

    class Params(NamedTuple):

        a: Any
        q_sqrtm_upper: Any

        @property
        def q_sqrtm_lower(self):
            return self.q_sqrtm_upper.T

    def __init__(self, *, derivative_init_fn, information_fn, num_derivatives):
        self.derivative_init_fn = derivative_init_fn
        self.information_fn = information_fn

        # static parameter, therefore a class attribute instead of a parameter
        self.num_derivatives = num_derivatives

    def init_fn(self, *, ivp, params):
        f, u0, t0 = ivp.ode_function.vector_field, ivp.initial_values, ivp.t0

        m0_corrected = self.derivative_init_fn(
            vector_field=f, initial_values=(u0,), num=self.num_derivatives
        )
        m0_corrected = jnp.stack(m0_corrected)
        if m0_corrected.ndim == 1:
            m0_corrected = m0_corrected[:, None]

        c_sqrtm0_corrected = jnp.zeros(
            (self.num_derivatives + 1, self.num_derivatives + 1)
        )
        rv_corrected = rv.Normal(mean=m0_corrected, cov_sqrtm_upper=c_sqrtm0_corrected)

        m0_extrapolated = jnp.zeros_like(m0_corrected)
        c_sqrtm0_extrapolated = jnp.eye(*c_sqrtm0_corrected.shape)
        rv_extrapolated = rv.Normal(
            mean=m0_extrapolated, cov_sqrtm_upper=c_sqrtm0_extrapolated
        )

        return self.State(
            t=t0,
            u=u0,
            error_estimate=jnp.nan,
            rv_corrected=rv_corrected,
            rv_extrapolated=rv_extrapolated,
        )

    def step_fn(self, *, state, ode_function, dt0, params):

        # Turn this into a state = update_fn() thingy?
        # Do we want an init() method, too? (We probably need one.)
        # Is this its own class then? If so, what are the state and the params?
        x = self._evaluate_and_extrapolate_fn(
            dt0=dt0, ode_function=ode_function, params=params, state=state
        )
        (bias, linear_fn), error_estimate, rv_extrapolated = x

        # Final observation
        s_sqrtm = linear_fn(rv_extrapolated.cov_sqrtm_upper.T)  # shape (n,)
        s = jnp.dot(s_sqrtm, s_sqrtm)
        rv_observed = rv.Normal(mean=bias, cov_sqrtm_upper=jnp.sqrt(s))
        g = (rv_extrapolated.cov_sqrtm_upper.T @ s_sqrtm.T) / s  # shape (n,)

        # Final correction
        m_cor = rv_extrapolated.mean - g[:, None] * rv_observed.mean[None, :]
        c_sqrtm_cor = rv_extrapolated.cov_sqrtm_upper.T - g[:, None] * s_sqrtm[None, :]
        rv_corrected = rv.Normal(mean=m_cor, cov_sqrtm_upper=c_sqrtm_cor.T)

        t_new = state.t + dt0
        return self.State(
            t=t_new,
            u=jnp.squeeze(rv_corrected.mean[0]),
            error_estimate=error_estimate,
            rv_extrapolated=rv_extrapolated,
            rv_corrected=rv_corrected,
        )

    def _evaluate_and_extrapolate_fn(self, *, dt0, ode_function, params, state):
        # Compute preconditioner
        p, p_inv = ibm.preconditioner_diagonal(
            dt=dt0, num_derivatives=self.num_derivatives
        )
        # Extract previous correction
        (m0, c_sqrtm0) = state.rv_corrected

        # Extrapolate the mean and linearise the differential equation.
        m_extrapolated = p[:, None] * (params.a @ (p_inv[:, None] * m0))
        bias, linear_fn = self.information_fn(ode_function.vector_field, m_extrapolated)

        # Observe the error-free state and calibrate some parameters
        s_sqrtm_lower = linear_fn(p_inv[:, None] * params.q_sqrtm_lower)
        s = jnp.dot(s_sqrtm_lower, s_sqrtm_lower)
        residual_white = bias / jnp.sqrt(s)
        diffusion_sqrtm = jnp.sqrt(
            jnp.dot(residual_white, residual_white) / residual_white.size
        )
        error_estimate = dt0 * diffusion_sqrtm * jnp.sqrt(s)

        # Full extrapolation
        c_sqrtm_extrapolated_p = sqrtm.sum_of_sqrtm_factors(
            R1=(params.a @ (p_inv[:, None] * c_sqrtm0)).T,
            R2=diffusion_sqrtm * params.q_sqrtm_lower,
        ).T
        c_sqrtm_extrapolated = p[:, None] * c_sqrtm_extrapolated_p

        rv_extrapolated = rv.Normal(
            mean=m_extrapolated, cov_sqrtm_upper=c_sqrtm_extrapolated
        )
        return (bias, linear_fn), error_estimate, rv_extrapolated


def adaptive(*, non_adaptive_solver, control, atol, rtol, error_order):
    """Turn a non-adaptive IVP solver into an adaptive IVP solver."""
    solver_alg, solver_params = non_adaptive_solver
    control_alg, control_params = control

    params = _SolverWithControl.Params(
        atol=atol,
        rtol=rtol,
        error_order=error_order,
        solver=solver_params,
        control=control_params,
    )

    alg = _SolverWithControl(solver=solver_alg, control=control_alg)
    return alg, params


class _SolverWithControl(AbstractIVPSolver):
    # Take a solver, normalise its error estimate,
    # and propose time-steps based on tolerances.

    class State(NamedTuple):

        dt_proposed: float
        error_normalised: float

        solver: Any  # must contain fields "t" and "u".
        control: Any  # must contain field "scale_factor".

        @property
        def t(self):
            return self.solver.t

        @property
        def u(self):
            return self.solver.u

    class Params(NamedTuple):

        atol: float
        rtol: float

        error_order: int
        solver: Any
        control: Any
        norm_ord: Union[int, str, None] = None

    def __init__(self, *, solver, control):
        self.solver = solver
        self.control = control

    def init_fn(self, *, ivp, params):

        state_solver = self.solver.init_fn(ivp=ivp, params=params.solver)
        state_control = self.control.init_fn()

        error_normalised = self._normalise_error(
            error_estimate=state_solver.error_estimate,
            u=state_solver.u,
            atol=params.atol,
            rtol=params.rtol,
            norm_ord=params.norm_ord,
        )
        dt_proposed = self._propose_first_dt_per_tol(
            f=ivp.ode_function.vector_field,
            u0=ivp.initial_values,
            error_order=params.error_order,
            rtol=params.rtol,
            atol=params.atol,
        )
        return self.State(
            dt_proposed=dt_proposed,
            error_normalised=error_normalised,
            solver=state_solver,
            control=state_control,
        )

    def step_fn(self, *, state, ode_function, dt0, params):
        def cond_fn(x):
            proceed_iteration, _ = x
            return proceed_iteration

        def body_fn(x):
            _, s = x
            s = self.attempt_step_fn(
                state=s, ode_function=ode_function, dt0=dt0, params=params
            )
            proceed_iteration = s.error_normalised > 1.0
            return proceed_iteration, s

        def init_fn(s):
            return True, s

        init_val = init_fn(state)
        _, state_new = jax.lax.while_loop(cond_fn, body_fn, init_val)
        return state_new

    def attempt_step_fn(self, *, state, ode_function, dt0, params):
        """Perform a step with an IVP solver and \
        propose a future time-step based on tolerances and error estimates."""
        state_solver = self.solver.step_fn(
            state=state.solver, ode_function=ode_function, dt0=dt0, params=params.solver
        )
        error_normalised = self._normalise_error(
            error_estimate=state_solver.error_estimate,
            u=state_solver.u,
            atol=params.atol,
            rtol=params.rtol,
            norm_ord=params.norm_ord,
        )
        state_control = self.control.control_fn(
            state=state.control,
            error_normalised=error_normalised,
            error_order=params.error_order,
            params=params.control,
        )
        dt_proposed = dt0 * state_control.scale_factor
        return self.State(
            dt_proposed=dt_proposed,
            error_normalised=error_normalised,
            solver=state_solver,
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
