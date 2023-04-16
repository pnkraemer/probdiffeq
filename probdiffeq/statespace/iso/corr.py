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

    def init(self, s, /):
        m_like = jnp.zeros_like(s.hidden_state.mean[..., 0, :])
        l_like = jnp.zeros_like(s.hidden_state.cov_sqrtm_lower[..., 0, 0])
        obs_like = _vars.IsoNormalQOI(mean=m_like, cov_sqrtm_lower=l_like)

        error_estimate = jnp.zeros(())
        corr = _corr.State(
            observed=obs_like,
            output_scale_dynamic=None,
            error_estimate=error_estimate,
            cache=None,
        )
        return s, corr

    def begin(self, x: _vars.IsoSSV, c: _corr.State, /, vector_field, p):
        m = x.hidden_state.mean
        m0, m1 = m[: self.ode_order, ...], m[self.ode_order, ...]
        bias = m1 - vector_field(*m0, t=x.t, p=p)
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

        u = m[self.ode_order]
        ssv = _vars.IsoSSV(
            x.t, u, x.hidden_state, num_data_points=x.num_data_points + 1
        )
        corr = _corr.State(
            observed=None,
            output_scale_dynamic=output_scale,
            error_estimate=error_estimate,
            cache=(bias,),
        )
        return ssv, corr

    def complete(
        self, x: _vars.IsoSSV, c: _corr.State, /, _vector_field, _p
    ) -> Tuple[_vars.IsoSSV, _corr.State]:
        (bias,) = c.cache

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

        u = corrected.mean[0, :]
        ssv = _vars.IsoSSV(x.t, u, corrected, num_data_points=x.num_data_points)
        corr = _corr.State(
            observed=observed,
            output_scale_dynamic=None,
            error_estimate=c.error_estimate,
            cache=None,
        )
        return ssv, corr
