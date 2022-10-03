from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jnp

from odefilter import ibm, inits, sqrtm, stepsizes

EK0State = namedtuple("KroneckerEK0State", ("u", "dt_proposed", "error_norm", "stats"))


def ek0(*, num_derivatives=5):
    """The Kronecker EK0, but only for computing the terminal value.

    Uses adaptive steps and proportional control.
    Uses time-varying, scalar-valued diffusion.
    Uses Taylor-mode initialisation for num_derivatives >= 5,
    and forward-mode initialisation otherweise.
    """

    # Create the identity matrix required for Kronecker-type things below.
    a, q_sqrtm = ibm.system_matrices_1d(num_derivatives=num_derivatives)

    # Initialisation with autodiff
    if num_derivatives <= 5:
        autodiff_fun = inits.ad_first_order.forwardmode_jvp
    else:
        autodiff_fun = inits.ad_first_order.taylormode

    def init_fn(
        *,
        f,
        tspan,
        u0,
        rtol,
        atol,
        safety=0.95,
        factor_min=0.2,
        factor_max=10.0,
        power_integral_unscaled=0.3,
        power_proportional_unscaled=0.4,
    ):

        m0_mat = autodiff_fun(f=f, u0=u0, num_derivatives=num_derivatives)
        m0_mat = m0_mat[:, None]
        c_sqrtm0 = jnp.zeros((num_derivatives + 1, num_derivatives + 1))
        dt0 = stepsizes.propose_first_dt_per_tol(
            f=f, u0=u0, atol=atol, rtol=rtol, num_derivatives=num_derivatives
        )

        stats = {
            "f_evaluation_count": 0,
            "steps_accepted_count": 0,
            "steps_attempted_count": 0,
            "dt_min": jnp.inf,
            "dt_max": 0.0,
        }
        state = KroneckerEK0State(
            u=(m0_mat, c_sqrtm0), dt_proposed=dt0, error_norm=1.0, stats=stats
        )
        return tspan[0], state

    def perform_step_fn(
        t0,
        state0,
        *,
        f,
        t1,
        rtol,
        atol,
        safety=0.95,
        factor_min=0.2,
        factor_max=10.0,
        power_integral_unscaled=0.3,
        power_proportional_unscaled=0.4,
    ):
        """Perform a successful step."""

        m0, c_sqrtm0 = state0.u
        error_norm_previously_accepted = state0.error_norm

        def cond_fun(s):
            _, state = s
            return state.error_norm > 1

        def body_fun(s):
            _, s_prev = s

            # Never exceed the terminal value
            dt_clipped = jnp.minimum(s_prev.dt_proposed, t1 - t0)
            t_new = t0 + dt_clipped

            # Compute preconditioner
            p, p_inv = ibm.preconditioner_diagonal(
                dt=dt_clipped, num_derivatives=num_derivatives
            )

            # Attempt step
            u_new, _, error = attempt_step_forward_only(
                f=f,
                m=m0,
                c_sqrtm=c_sqrtm0,
                p=p,
                p_inv=p_inv,
                a=a,
                q_sqrtm=q_sqrtm,
            )
            error = dt_clipped * error

            # Normalise the error
            m_new, _ = u_new
            print(m_new.shape, m0.shape)
            u1_ref = jnp.abs(jnp.maximum(m_new[0, :], m0[0, :]))
            error_rel = error / (atol + rtol * u1_ref)
            error_norm = jnp.linalg.norm(error_rel) / jnp.sqrt(error.size)

            # Propose a new time-step
            scale_factor = stepsizes.scale_factor_pi_control(
                error_norm=error_norm,
                error_order=num_derivatives + 1,
                safety=safety,
                factor_min=factor_min,
                factor_max=factor_max,
                error_norm_previously_accepted=error_norm_previously_accepted,
                power_integral_unscaled=power_integral_unscaled,
                power_proportional_unscaled=power_proportional_unscaled,
            )
            dt_proposed = scale_factor * dt_clipped

            stats = s_prev.stats
            stats["f_evaluation_count"] += 1
            stats["steps_attempted_count"] += 1

            state = KroneckerEK0State(
                u=u_new, dt_proposed=dt_proposed, error_norm=error_norm, stats=stats
            )
            return t_new, state

        larger_than_1 = 1.1
        init_val = KroneckerEK0State(
            u=state0.u,
            dt_proposed=state0.dt_proposed,
            error_norm=larger_than_1,
            stats=state0.stats,
        )
        t, state = jax.lax.while_loop(
            cond_fun=cond_fun,
            body_fun=body_fun,
            init_val=(t0, init_val),
        )
        stats = state.stats
        stats["steps_accepted_count"] += 1
        stats["dt_min"] = jnp.minimum(stats["dt_min"], t - t0)
        stats["dt_max"] = jnp.maximum(stats["dt_max"], t - t0)
        state = KroneckerEK0State(
            u=state.u,
            dt_proposed=state.dt_proposed,
            error_norm=state.error_norm,
            stats=stats,
        )
        return t, state

    def extract_qoi_fn(t, state):
        return t, state.u, state.stats

    return init_fn, perform_step_fn, extract_qoi_fn


def attempt_step_forward_only(*, f, m, c_sqrtm, p, p_inv, a, q_sqrtm):
    """A step with the 'KroneckerEK0'.

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


@jax.jit
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


@jax.jit
def _estimate_error(*, m_res, q_sqrtm):
    s_sqrtm = q_sqrtm[1, :]
    s = s_sqrtm @ s_sqrtm.T
    diff = m_res.T @ m_res / (m_res.size * s)
    diff_sqrtm = jnp.sqrt(diff)
    error_estimate = diff_sqrtm * jnp.sqrt(s)
    return error_estimate, diff_sqrtm
