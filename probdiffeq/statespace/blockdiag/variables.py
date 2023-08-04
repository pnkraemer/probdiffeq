"""Block-diagonal variables."""

import jax
import jax.numpy as jnp

from probdiffeq import _sqrt_util
from probdiffeq.statespace.scalar import variables as scalar_variables


def marginalise_deterministic(rv, trafo):
    A, b = trafo
    mean, cov_sqrtm_lower = rv.mean, rv.cov_sqrtm_lower

    cov_new = jax.vmap(_sqrt_util.triu_via_qr)(A(cov_sqrtm_lower)[:, :, None])
    cov_new = jnp.squeeze(cov_new, axis=(-2, -1))
    return scalar_variables.NormalQOI(A(mean) + b, cov_new)


def revert_deterministic(rv, trafo):
    # Extract information
    A, b = trafo
    mean, cov_sqrtm_lower = rv.mean, rv.cov_sqrtm_lower

    # QR-decomposition
    # (todo: rename revert_conditional_noisefree to revert_transformation_cov_sqrt())
    cov_sqrtm_upper = jnp.transpose(cov_sqrtm_lower, axes=(0, 2, 1))
    r_obs, (r_cor, gain) = jax.vmap(_sqrt_util.revert_conditional_noisefree)(
        R_X_F=A(cov_sqrtm_lower)[:, :, None], R_X=cov_sqrtm_upper
    )
    l_cor = jnp.transpose(r_cor, axes=(0, 2, 1))
    l_obs = jnp.squeeze(r_obs, axis=(-2, -1))
    gain = jnp.squeeze(gain, axis=-1)

    # Gather terms and return
    m_cor = mean - gain * (A(mean) + b)[..., None]
    corrected = scalar_variables.NormalHiddenState(m_cor, l_cor)
    observed = scalar_variables.NormalQOI(A(mean) + b, l_obs)
    return observed, (corrected, gain)
