"""Solvers for IVPs."""

import abc
from collections import namedtuple
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from odefilter import sqrtm
from odefilter.prob import ibm


class AbstractIVPSolver(abc.ABC):
    """Abstract solver for IVPs."""

    @abc.abstractmethod
    def init_fn(
        self,
        ivp,
        params,
    ):
        """Initialise the IVP solver state."""
        raise NotImplementedError

    @abc.abstractmethod
    def perform_step_fn(self, state0, *, ode_function, t1, params):
        """Perform a step."""
        raise NotImplementedError


def ek0(*, num_derivatives, control, init):
    """Create an ODEFilter that implements the EK0."""
    controller, control_params = control
    init_algorithm, init_params = init

    # Assemble solver
    alg = _EK0(
        num_derivatives=num_derivatives,
        control=controller,
        init=init_algorithm,
    )

    # Assemble parameters
    a, q_sqrtm = ibm.system_matrices_1d(num_derivatives=num_derivatives)
    params = _EK0Params(a=a, q_sqrtm=q_sqrtm, control=control_params, init=init_params)
    return alg, params


class _EK0Params(NamedTuple):
    a: Any
    q_sqrtm: Any

    control: Any
    init: Any


_KroneckerEK0State = namedtuple(
    "_KroneckerEK0State", ("t", "u", "dt_proposed", "error_norm", "stats")
)
"""EK0 State.

state.t
state.u
state.params
state.step.dt_proposed
state.step.params
state.step.control.error_norm
state.step.control.error_norm_previously_accepted
state.step.control.params
"""


class _EK0:
    """The Kronecker EK0, but only for computing the terminal value.

    Uses adaptive steps and proportional control.
    Uses time-varying, scalar-valued diffusion.
    Uses Taylor-mode initialisation for num_derivatives >= 5,
    and forward-mode initialisation otherweise.
    """

    def __init__(self, *, control, init, num_derivatives=5):

        self.num_derivatives = num_derivatives
        self.control = control
        self.init = init

    def init_fn(self, *, ivp, params):

        f, u0, t0 = ivp.ode_function.f, ivp.y0, ivp.t0
        m0_mat = self.init(f=f, u0=u0, num_derivatives=self.num_derivatives)
        m0_mat = m0_mat[:, None]
        c_sqrtm0 = jnp.zeros((self.num_derivatives + 1, self.num_derivatives + 1))
        dt0 = self.control.propose_first_dt(f=f, u0=u0, params=params.control)

        stats = {
            "f_evaluation_count": 0,
            "steps_accepted_count": 0,
            "steps_attempted_count": 0,
            "dt_min": jnp.inf,
            "dt_max": 0.0,
        }
        state = _KroneckerEK0State(
            t=t0,
            u=(m0_mat, c_sqrtm0),
            dt_proposed=dt0,
            error_norm=1.0,
            stats=stats,
        )
        return state

    def perform_step_fn(self, state0, *, ode_function, t1, params):
        """Perform a successful step."""
        larger_than_1 = 1.1
        init_val = _KroneckerEK0State(
            t=state0.t,
            u=state0.u,
            dt_proposed=state0.dt_proposed,
            error_norm=larger_than_1,
            stats=state0.stats,
        )
        state = jax.lax.while_loop(
            cond_fun=lambda s: s.error_norm > 1,
            body_fun=lambda s: self.attempt_step_fn(
                s,
                ode_function=ode_function,
                state0=state0,
                t1=t1,
                params=params,
            ),
            init_val=init_val,
        )
        stats = state.stats
        stats["steps_accepted_count"] += 1
        stats["dt_min"] = jnp.minimum(stats["dt_min"], state.t - state0.t)
        stats["dt_max"] = jnp.maximum(stats["dt_max"], state.t - state0.t)
        state = _KroneckerEK0State(
            t=state.t,
            u=state.u,
            dt_proposed=state.dt_proposed,
            error_norm=state.error_norm,
            stats=stats,
        )
        return state

    def attempt_step_fn(self, s_prev, *, ode_function, state0, t1, params):

        m0, c_sqrtm0 = state0.u
        error_norm_previously_accepted = state0.error_norm

        # Never exceed the terminal value
        dt_clipped = jnp.minimum(s_prev.dt_proposed, t1 - s_prev.t)
        t_new = s_prev.t + dt_clipped

        # Compute preconditioner
        p, p_inv = ibm.preconditioner_diagonal(
            dt=dt_clipped, num_derivatives=self.num_derivatives
        )

        # Attempt step
        u_new, _, error = _attempt_step_forward_only(
            f=ode_function.f,
            m=m0,
            c_sqrtm=c_sqrtm0,
            p=p,
            p_inv=p_inv,
            a=params.a,
            q_sqrtm=params.q_sqrtm,
        )
        error = dt_clipped * error

        # Normalise the error
        m_new, _ = u_new
        u1_ref = jnp.abs(jnp.maximum(m_new[0, :], m0[0, :]))
        error_norm = self.control.normalise_error(
            error=error, u1_ref=u1_ref, params=params.control
        )

        # Propose a new time-step
        scale_factor = self.control.scale_factor(
            error_norm=error_norm,
            error_norm_previously_accepted=error_norm_previously_accepted,
            params=params.control,
        )
        dt_proposed = scale_factor * dt_clipped

        stats = s_prev.stats
        stats["f_evaluation_count"] += 1
        stats["steps_attempted_count"] += 1

        state = _KroneckerEK0State(
            t=t_new,
            u=u_new,
            dt_proposed=dt_proposed,
            error_norm=error_norm,
            stats=stats,
        )
        return state


def _attempt_step_forward_only(*, f, m, c_sqrtm, p, p_inv, a, q_sqrtm):
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
    err, diff_sqrtm = _estimate_error(m_res=m_obs, q_sqrtm=p[:, None] * q_sqrtm)

    # The full extrapolation:
    c_sqrtm_ext = sqrtm.sum_of_sqrtm_factors(
        R1=(a @ c_sqrtm).T, R2=diff_sqrtm * q_sqrtm.T
    ).T

    # Un-apply the pre-conditioner.
    # Now it is also done serving its purpose for the covariance.
    c_sqrtm_ext = p[:, None] * c_sqrtm_ext

    # The final correction
    c_sqrtm_obs, (m_cor, c_sqrtm_cor) = _final_correction(
        m_obs=m_obs, m_ext=m_ext, c_sqrtm_ext=c_sqrtm_ext
    )

    return (m_cor, c_sqrtm_cor), (m_obs, c_sqrtm_obs), err


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


def _estimate_error(*, m_res, q_sqrtm):
    s_sqrtm = q_sqrtm[1, :]
    s = s_sqrtm @ s_sqrtm.T
    diff = m_res.T @ m_res / (m_res.size * s)
    diff_sqrtm = jnp.sqrt(diff)
    error_estimate = diff_sqrtm * jnp.sqrt(s)
    return error_estimate, diff_sqrtm
