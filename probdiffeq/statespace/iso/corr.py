"""Corrections."""

import jax
import jax.numpy as jnp

from probdiffeq import _sqrt_util
from probdiffeq.statespace import _corr
from probdiffeq.statespace.iso import variables


def taylor_order_zero(*args, **kwargs):
    return _IsoTaylorZerothOrder(*args, **kwargs)


@jax.tree_util.register_pytree_node_class
class _IsoTaylorZerothOrder(_corr.Correction):
    def __repr__(self):
        return f"<TS0 with ode_order={self.ode_order}>"

    def init(self, x, /):
        bias_like = jnp.empty_like(x.hidden_state.mean[0, :])
        chol_like = jnp.empty(())
        obs_like = variables.IsoNormalQOI(bias_like, chol_like)
        return x, obs_like

    def begin(self, x: variables.IsoSSV, c, /, vector_field, t, p):
        m = x.hidden_state.mean
        m0, m1 = m[: self.ode_order, ...], m[self.ode_order, ...]
        bias = m1 - vector_field(*m0, t=t, p=p)
        cov_sqrtm_lower = x.hidden_state.cov_sqrtm_lower[self.ode_order, ...]

        l_obs_nonscalar = _sqrt_util.triu_via_qr(cov_sqrtm_lower[:, None])
        l_obs = jnp.reshape(l_obs_nonscalar, ())
        obs = variables.IsoNormalQOI(bias, l_obs)

        mahalanobis_norm = obs.mahalanobis_norm(jnp.zeros_like(m1))
        output_scale = mahalanobis_norm / jnp.sqrt(bias.size)

        error_estimate_unscaled = obs.marginal_std() * jnp.ones_like(bias)
        error_estimate = error_estimate_unscaled * output_scale
        cache = (bias,)
        return error_estimate, obs, (x, cache)

    def complete(self, x: variables.IsoSSV, co, /, vector_field, t, p):
        (bias,) = co

        m_ext = x.hidden_state.mean
        l_ext = x.hidden_state.cov_sqrtm_lower
        l_obs = l_ext[self.ode_order, ...]

        l_obs_nonscalar = _sqrt_util.triu_via_qr(l_obs[:, None])
        l_obs_scalar = jnp.reshape(l_obs_nonscalar, ())

        observed = variables.IsoNormalQOI(mean=bias, cov_sqrtm_lower=l_obs_scalar)

        g = (l_ext @ l_obs.T) / l_obs_scalar  # shape (n,)
        m_cor = m_ext - g[:, None] * bias[None, :] / l_obs_scalar
        l_cor = l_ext - g[:, None] * l_obs[None, :] / l_obs_scalar
        corrected = variables.IsoNormalHiddenState(mean=m_cor, cov_sqrtm_lower=l_cor)
        ssv = variables.IsoSSV(m_cor[0, :], corrected)
        return ssv, observed

    def extract(self, x, c, /):
        return x
