"""Implementations for scalar initial value problems."""

import jax
import jax.numpy as jnp

from probdiffeq import _sqrt_util
from probdiffeq.statespace import _corr
from probdiffeq.statespace.scalar import variables


def taylor_order_zero(*args, **kwargs):
    return _TaylorZerothOrder(*args, **kwargs)


def correct_affine_qoi_noisy(rv, affine, *, stdev):
    # Read inputs
    A, b = affine

    # Apply observation model to covariance
    cov_sqrtm = rv.cov_sqrtm_lower
    cov_sqrtm_obs_nonsquare = jnp.dot(A, cov_sqrtm[0, ...])

    # Revert the conditional covariances
    cov_sqrtm_obs_upper, (
        cov_sqrtm_cor_upper,
        gain,
    ) = _sqrt_util.revert_conditional(
        R_X_F=cov_sqrtm_obs_nonsquare[None, :].T,
        R_X=rv.cov_sqrtm_lower.T,
        R_YX=jnp.ones((1, 1)) * stdev,
    )
    cov_sqrtm_obs = cov_sqrtm_obs_upper.T
    cov_sqrtm_cor = cov_sqrtm_cor_upper.T
    gain = gain[:, 0]  # "squeeze"; output shape is (), not (1,)

    # Gather the observed variable
    mean_obs = jnp.dot(A, rv.mean[0, ...]) + b
    observed = variables.NormalQOI(mean=mean_obs, cov_sqrtm_lower=cov_sqrtm_obs)

    # Gather the corrected variable
    mean_cor = rv.mean - gain * mean_obs
    corrected = variables.NormalHiddenState(
        mean=mean_cor, cov_sqrtm_lower=cov_sqrtm_cor
    )
    return observed, (corrected, gain)


def correct_affine_ode_2nd(rv, affine):
    # Read inputs
    A, b = affine

    # Apply observation model to covariance
    cov_sqrtm = rv.cov_sqrtm_lower
    cov_sqrtm_obs_nonsquare = cov_sqrtm[2, ...] - jnp.dot(A, cov_sqrtm[0, ...])

    # Revert the conditional covariances
    cov_sqrtm_obs_upper, (
        cov_sqrtm_cor_upper,
        gain,
    ) = _sqrt_util.revert_conditional_noisefree(
        R_X_F=cov_sqrtm_obs_nonsquare[None, :].T, R_X=rv.cov_sqrtm_lower.T
    )
    cov_sqrtm_obs = cov_sqrtm_obs_upper.T
    cov_sqrtm_cor = cov_sqrtm_cor_upper.T
    gain = gain[:, 0]  # "squeeze"; output shape is (), not (1,)

    # Gather the observed variable
    mean_obs = rv.mean[2, ...] - jnp.dot(A, rv.mean[0, ...]) - b
    observed = variables.NormalQOI(mean=mean_obs, cov_sqrtm_lower=cov_sqrtm_obs)

    # Gather the corrected variable
    mean_cor = rv.mean - gain * mean_obs
    corrected = variables.NormalHiddenState(
        mean=mean_cor, cov_sqrtm_lower=cov_sqrtm_cor
    )
    return observed, (corrected, gain)


@jax.tree_util.register_pytree_node_class
class _TaylorZerothOrder(_corr.Correction):
    def __repr__(self):
        return f"<TS0 with ode_order={self.ode_order}>"

    def init(self, ssv, /):
        m_like = jnp.zeros(())
        chol_like = jnp.zeros(())
        obs_like = variables.NormalQOI(m_like, chol_like)
        return ssv, obs_like

    def estimate_error(self, ssv: variables.SSV, corr, /, vector_field, t, p):
        m0, m1 = self.select_derivatives(ssv.hidden_state)
        fx = vector_field(*m0, t=t, p=p)
        cache, observed = self.marginalise_observation(fx, m1, ssv.hidden_state)
        mahalanobis_norm = observed.mahalanobis_norm(jnp.zeros(()))
        output_scale = mahalanobis_norm / jnp.sqrt(m1.size)
        error_estimate_unscaled = observed.marginal_stds()
        error_estimate = output_scale * error_estimate_unscaled
        return error_estimate, observed, cache

    def marginalise_observation(self, fx, m1, x):
        b = m1 - fx
        cov_sqrtm_lower = x.cov_sqrtm_lower[self.ode_order, :]
        l_obs_raw = _sqrt_util.triu_via_qr(cov_sqrtm_lower[:, None])
        l_obs = jnp.reshape(l_obs_raw, ())
        observed = variables.NormalQOI(b, l_obs)
        cache = (b,)
        return cache, observed

    def select_derivatives(self, x):
        m0, m1 = x.mean[: self.ode_order], x.mean[self.ode_order]
        return m0, m1

    def complete(self, ssv: variables.SSV, corr, /, vector_field, t, p):
        (b,) = corr
        m_ext, l_ext = (ssv.hidden_state.mean, ssv.hidden_state.cov_sqrtm_lower)

        l_obs_nonsquare = l_ext[self.ode_order, :]
        r_obs_mat, (r_cor, gain_mat) = _sqrt_util.revert_conditional_noisefree(
            R_X_F=l_obs_nonsquare[:, None], R_X=l_ext.T
        )
        r_obs = jnp.reshape(r_obs_mat, ())
        gain = jnp.reshape(gain_mat, (-1,))
        m_cor = m_ext - gain * b
        observed = variables.NormalQOI(mean=b, cov_sqrtm_lower=r_obs.T)

        rv_cor = variables.NormalHiddenState(mean=m_cor, cov_sqrtm_lower=r_cor.T)
        corrected = variables.SSV(m_cor[0], rv_cor)
        return corrected, observed

    def extract(self, ssv, corr):
        return ssv


def estimate_error(observed, /):
    mahalanobis_norm = observed.mahalanobis_norm(jnp.zeros(()))
    output_scale = mahalanobis_norm
    error_estimate_unscaled = observed.marginal_stds()
    return output_scale * error_estimate_unscaled
