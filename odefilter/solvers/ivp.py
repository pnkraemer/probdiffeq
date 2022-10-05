"""Initial value problem solvers."""
import abc
from typing import Any, NamedTuple

import jax.numpy as jnp

from odefilter import sqrtm
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


def ek0(*, init, num_derivatives):
    """EK0 solver."""
    init_alg, init_params = init

    alg = _EK0(init=init_alg)

    a, q_sqrtm = ibm.system_matrices_1d(num_derivatives=num_derivatives)
    params = _EK0.Params(
        num_derivatives=num_derivatives, init=init_params, a=a, q_sqrtm=q_sqrtm
    )
    return alg, params


class _EK0(AbstractIVPSolver):
    """EK0."""

    class State(NamedTuple):
        t: float
        u: Any
        error_estimate: Any

        hidden_state: Any

    class Params(NamedTuple):
        num_derivatives: int
        init: Any

        a: Any
        q_sqrtm: Any

    def __init__(self, *, init):
        self.init = init

    def init_fn(self, *, ivp, params):
        f, u0, t0 = ivp.ode_function.f, ivp.initial_values, ivp.t0
        m0_mat = self.init(f=f, u0=u0, num_derivatives=params.num_derivatives)
        m0_mat = m0_mat[:, None]
        c_sqrtm0 = jnp.zeros((params.num_derivatives + 1, params.num_derivatives + 1))

        return self.State(
            t=t0, u=u0, hidden_state=(m0_mat, c_sqrtm0), error_estimate=jnp.nan
        )

    def step_fn(self, *, state, ode_function, dt0, params):
        t, (m0, c_sqrtm0) = state.t, state.hidden_state

        p, p_inv = ibm.preconditioner_diagonal(
            dt=dt0, num_derivatives=params.num_derivatives
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

        # Compute the error estimate
        m_obs = m_ext[1, :] - f(m_ext[0, :])
        err, diff_sqrtm = self._estimate_error(
            m_res=m_obs, q_sqrtm=p[:, None] * q_sqrtm
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
            m_obs=m_obs, m_ext=m_ext, c_sqrtm_ext=c_sqrtm_ext
        )

        return (m_cor, c_sqrtm_cor), (m_obs, c_sqrtm_obs), err

    @staticmethod
    def _final_correction(*, m_obs, m_ext, c_sqrtm_ext):
        # no fancy QR/sqrtm-stuff, because
        # the observation matrices have shape (): they are scalars.
        # The correction is almost free.
        s_sqrtm = c_sqrtm_ext[1, :]  # shape (n,)
        s = s_sqrtm @ s_sqrtm.T

        g = (c_sqrtm_ext @ s_sqrtm.T) / s  # shape (n,)
        c_sqrtm_cor = c_sqrtm_ext - g[:, None] * s_sqrtm[None, :]
        m_cor = m_ext - g[:, None] * m_obs[None, :]

        c_sqrtm_obs = jnp.sqrt(s)
        return c_sqrtm_obs, (m_cor, c_sqrtm_cor)

    @staticmethod
    def _estimate_error(*, m_res, q_sqrtm):
        s_sqrtm = q_sqrtm[1, :]
        s = s_sqrtm @ s_sqrtm.T
        diff = m_res.T @ m_res / (m_res.size * s)
        diff_sqrtm = jnp.sqrt(diff)
        error_estimate = diff_sqrtm * jnp.sqrt(s)
        return error_estimate, diff_sqrtm
