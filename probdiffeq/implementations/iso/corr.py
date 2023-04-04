"""Corrections."""
from typing import Tuple

import jax
import jax.numpy as jnp

from probdiffeq.implementations import _collections, _sqrtm
from probdiffeq.implementations.iso import _vars


@jax.tree_util.register_pytree_node_class
class IsoTaylorZerothOrder(_collections.AbstractCorrection):
    def begin_correction(self, x: _vars.IsoStateSpaceVar, /, vector_field, t, p):
        m = x.hidden_state.mean
        m0, m1 = m[: self.ode_order, ...], m[self.ode_order, ...]
        bias = m1 - vector_field(*m0, t=t, p=p)
        cov_sqrtm_lower = x.hidden_state.cov_sqrtm_lower[self.ode_order, ...]

        l_obs_square = _sqrtm.sqrtm_to_upper_triangular(R=cov_sqrtm_lower[:, None])
        l_obs = jnp.reshape(l_obs_square, ())

        output_scale_sqrtm_ss = _sqrtm.sqrtm_to_upper_triangular(R=bias[:, None])
        output_scale_sqrtm_s = jnp.reshape(
            output_scale_sqrtm_ss / jnp.sqrt(bias.size), ()
        )
        output_scale_sqrtm = jnp.reshape(output_scale_sqrtm_s / l_obs, ())

        return output_scale_sqrtm_s, output_scale_sqrtm, (bias,)

    def complete_correction(
        self, extrapolated: _vars.IsoStateSpaceVar, cache
    ) -> Tuple[_vars.IsoNormal, Tuple[_vars.IsoStateSpaceVar, jax.Array]]:
        (bias,) = cache

        m_ext, l_ext = (
            extrapolated.hidden_state.mean,
            extrapolated.hidden_state.cov_sqrtm_lower,
        )
        l_obs = l_ext[self.ode_order, ...]

        l_obs_scalar = jnp.reshape(
            _sqrtm.sqrtm_to_upper_triangular(R=l_obs[:, None]), ()
        )
        observed = _vars.IsoNormal(mean=bias, cov_sqrtm_lower=l_obs_scalar)

        g = l_ext @ l_obs.T / l_obs_scalar  # shape (n,)
        m_cor = m_ext - g[:, None] * bias[None, :] / l_obs_scalar
        l_cor = l_ext - g[:, None] * l_obs[None, :] / l_obs_scalar
        corrected = _vars.IsoNormal(mean=m_cor, cov_sqrtm_lower=l_cor)
        return observed, (_vars.IsoStateSpaceVar(corrected), g)
