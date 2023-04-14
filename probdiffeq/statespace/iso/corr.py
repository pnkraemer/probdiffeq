"""Corrections."""

import jax
import jax.numpy as jnp

from probdiffeq import _sqrt_util
from probdiffeq.statespace import _collections
from probdiffeq.statespace.iso import _vars


def taylor_order_zero(*args, **kwargs):
    return _IsoTaylorZerothOrder(*args, **kwargs)


@jax.tree_util.register_pytree_node_class
class _IsoTaylorZerothOrder(_collections.AbstractCorrection):
    def __repr__(self):
        return f"<TS0 with ode_order={self.ode_order}>"

    def init(self, s, /):
        m_like = jnp.zeros_like(s.hidden_state.mean[..., 0, :])
        l_like = jnp.zeros_like(s.hidden_state.cov_sqrtm_lower[..., 0, 0])
        obs_like = _vars.IsoNormalQOI(mean=m_like, cov_sqrtm_lower=l_like)

        error_estimate = jnp.zeros(())
        return _vars.IsoSSV(
            observed_state=obs_like,
            error_estimate=error_estimate,
            hidden_state=s.hidden_state,
            hidden_shape=s.hidden_shape,
            backward_model=s.backward_model,
            output_scale_dynamic=None,
            cache_extra=None,
            cache_corr=None,
        )
        return s

    def begin(self, x: _vars.IsoSSV, /, vector_field, t, p):
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

        return _vars.IsoSSV(
            x.hidden_state,
            hidden_shape=x.hidden_shape,
            observed_state=None,
            output_scale_dynamic=output_scale,
            error_estimate=error_estimate,
            cache_extra=x.cache_extra,
            cache_corr=(bias,),
            backward_model=x.backward_model,
        )

    def complete(self, x: _vars.IsoSSV, /, _vector_field, _t, _p) -> _vars.IsoSSV:
        (bias,) = x.cache_corr

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
        return _vars.IsoSSV(
            corrected,
            observed_state=observed,
            hidden_shape=x.hidden_shape,
            error_estimate=x.error_estimate,
            backward_model=x.backward_model,
            output_scale_dynamic=None,
            cache_extra=None,
            cache_corr=None,
        )
