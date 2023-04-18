"""Corrections."""
from typing import Tuple

import jax
import jax.numpy as jnp

from probdiffeq import _sqrt_util
from probdiffeq.statespace import _corr
from probdiffeq.statespace.iso import _vars


def taylor_order_zero(*args, **kwargs):
    return _IsoTaylorZerothOrder(*args, **kwargs)


@jax.tree_util.register_pytree_node_class
class _IsoTaylorZerothOrder(_corr.Correction):
    def __repr__(self):
        return f"<TS0 with ode_order={self.ode_order}>"

    def init(self, x, /):
        bias_like = jnp.empty_like(x.hidden_state.mean[0, :])
        chol_like = jnp.empty(())
        obs_like = _vars.IsoNormalQOI(bias_like, chol_like)
        return x, obs_like

    def begin(self, x: _vars.IsoSSV, c, /, vector_field, t, p):
        m = x.hidden_state.mean
        m0, m1 = m[: self.ode_order, ...], m[self.ode_order, ...]
        bias = m1 - vector_field(*m0, t=t, p=p)
        cov_sqrtm_lower = x.hidden_state.cov_sqrtm_lower[self.ode_order, ...]

        l_obs_nonscalar = _sqrt_util.sqrtm_to_upper_triangular(
            R=cov_sqrtm_lower[:, None]
        )
        l_obs = jnp.reshape(l_obs_nonscalar, ())
        obs = _vars.IsoNormalQOI(bias, l_obs)

        mahalanobis_norm = obs.mahalanobis_norm(jnp.zeros_like(m1))
        output_scale = mahalanobis_norm / jnp.sqrt(bias.size)

        error_estimate_unscaled = obs.marginal_std()
        error_estimate = error_estimate_unscaled * output_scale
        cache = (error_estimate, output_scale, (bias,))
        return x, cache

    def complete(self, x: _vars.IsoSSV, co, /, vector_field, p):
        *_, (bias,) = co

        m_ext = x.hidden_state.mean
        l_ext = x.hidden_state.cov_sqrtm_lower
        l_obs = l_ext[self.ode_order, ...]

        l_obs_nonscalar = _sqrt_util.sqrtm_to_upper_triangular(R=l_obs[:, None])
        l_obs_scalar = jnp.reshape(l_obs_nonscalar, ())

        observed = _vars.IsoNormalQOI(mean=bias, cov_sqrtm_lower=l_obs_scalar)

        g = (l_ext @ l_obs.T) / l_obs_scalar  # shape (n,)
        m_cor = m_ext - g[:, None] * bias[None, :] / l_obs_scalar
        l_cor = l_ext - g[:, None] * l_obs[None, :] / l_obs_scalar
        corrected = _vars.IsoNormalHiddenState(mean=m_cor, cov_sqrtm_lower=l_cor)
        ssv = _vars.IsoSSV(corrected)
        return ssv, observed

    def extract(self, x, c, /):
        return x
