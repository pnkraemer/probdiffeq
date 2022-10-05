"""Initial value problem solvers."""
import abc
from typing import Any, NamedTuple, Union

import jax.lax
import jax.numpy as jnp
import jax.tree_util

from odefilter import information, sqrtm
from odefilter.prob import ibm


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


def ek0_non_adaptive(*, derivative_init_fn, num_derivatives):
    """EK0 solver."""

    alg = _NonAdaptiveEK0(
        derivative_init_fn=derivative_init_fn,
        information_fn=information.linearize_ek0_kron_1st,
        num_derivatives=num_derivatives,
    )

    a, q_sqrtm = ibm.system_matrices_1d(num_derivatives=num_derivatives)
    params = _NonAdaptiveEK0.Params(a=a, q_sqrtm=q_sqrtm)
    return alg, params


class _NonAdaptiveEK0(AbstractIVPSolver):
    """EK0."""

    class State(NamedTuple):
        t: float
        u: Any
        error_estimate: Any

        hidden_state: Any

    class Params(NamedTuple):

        a: Any
        q_sqrtm: Any

    def __init__(self, *, derivative_init_fn, information_fn, num_derivatives):
        self.derivative_init_fn = derivative_init_fn
        self.information_fn = information_fn

        # static parameter, therefore a class attribute instead of a parameter
        self.num_derivatives = num_derivatives

    def init_fn(self, *, ivp, params):
        f, u0, t0 = ivp.ode_function.f, ivp.initial_values, ivp.t0
        m0_mat = self.derivative_init_fn(
            f=f, u0=u0, num_derivatives=self.num_derivatives
        )
        m0_mat = m0_mat[:, None]
        c_sqrtm0 = jnp.zeros((self.num_derivatives + 1, self.num_derivatives + 1))

        return self.State(
            t=t0, u=u0, hidden_state=(m0_mat, c_sqrtm0), error_estimate=jnp.nan
        )

    def step_fn(self, *, state, ode_function, dt0, params):
        t, (m0, c_sqrtm0) = state.t, state.hidden_state

        p, p_inv = ibm.preconditioner_diagonal(
            dt=dt0, num_derivatives=self.num_derivatives
        )

        u_new, _, error_estimate = self._attempt_step_forward_only(
            f=ode_function.f,
            m=m0,
            c_sqrtm=c_sqrtm0,
            p=p,
            p_inv=p_inv,
            a=params.a,
            q_sqrtm=params.q_sqrtm,
        )
        error_estimate = dt0 * error_estimate

        t_new = t + dt0
        return self.State(
            t=t_new,
            u=jnp.squeeze(u_new[0][0]),
            hidden_state=u_new,
            error_estimate=error_estimate,
        )

    def _attempt_step_forward_only(self, *, f, m, c_sqrtm, p, p_inv, a, q_sqrtm):
        """Step with the 'KroneckerEK0'.

        Includes error estimation.
        Includes time-varying, scalar diffusion.
        """
        # m is an (nu+1,d) array. c_sqrtm is a (nu+1,nu+1) array.

        # Apply the pre-conditioner
        m, c_sqrtm = p_inv[:, None] * m, p_inv[:, None] * c_sqrtm

        # Predict the mean.
        # Immediately undo the preconditioning,
        # because it's served its purpose for the mean.
        # (It is not really necessary for the mean, to be honest.)
        m_ext = p[:, None] * (a @ m)

        bias, linear_fn = self.information_fn(f, m_ext)

        # Compute the error estimate
        m_obs = bias
        err, diff_sqrtm = self._estimate_error(
            m_res=m_obs, q_sqrtm=p[:, None] * q_sqrtm, linear_fn=linear_fn
        )

        # The full extrapolation:
        c_sqrtm_ext = sqrtm.sum_of_sqrtm_factors(
            R1=(a @ c_sqrtm).T, R2=diff_sqrtm * q_sqrtm.T
        ).T

        # Un-apply the pre-conditioner.
        # Now it is also done serving its purpose for the covariance.
        c_sqrtm_ext = p[:, None] * c_sqrtm_ext

        # The final correction
        c_sqrtm_obs, (m_cor, c_sqrtm_cor) = self._final_correction(
            m_obs=m_obs, m_ext=m_ext, c_sqrtm_ext=c_sqrtm_ext, linear_fn=linear_fn
        )

        return (m_cor, c_sqrtm_cor), (m_obs, c_sqrtm_obs), err

    @staticmethod
    def _final_correction(*, m_obs, m_ext, c_sqrtm_ext, linear_fn):
        # no fancy QR/sqrtm-stuff, because
        # the observation matrices have shape (): they are scalars.
        # The correction is almost free.
        s_sqrtm = linear_fn(c_sqrtm_ext)  # shape (n,)
        s = s_sqrtm @ s_sqrtm.T

        g = (c_sqrtm_ext @ s_sqrtm.T) / s  # shape (n,)
        c_sqrtm_cor = c_sqrtm_ext - g[:, None] * s_sqrtm[None, :]
        m_cor = m_ext - g[:, None] * m_obs[None, :]

        c_sqrtm_obs = jnp.sqrt(s)
        return c_sqrtm_obs, (m_cor, c_sqrtm_cor)

    @staticmethod
    def _estimate_error(*, m_res, q_sqrtm, linear_fn):
        s_sqrtm = linear_fn(q_sqrtm)
        s = s_sqrtm @ s_sqrtm.T
        diff = m_res.T @ m_res / (m_res.size * s)
        diff_sqrtm = jnp.sqrt(diff)
        error_estimate = diff_sqrtm * jnp.sqrt(s)
        return error_estimate, diff_sqrtm


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
            f=ivp.ode_function.f,
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
