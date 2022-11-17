"""Isotropic-style corrections."""
import jax
import jax.numpy as jnp

from probdiffeq.implementations import _collections, _sqrtm
from probdiffeq.implementations.iso import _ssv


@jax.tree_util.register_pytree_node_class
class IsoTaylorZerothOrder(_collections.AbstractCorrection):
    def begin_correction(self, x: _ssv.IsoNormal, /, vector_field, t, p):
        m = x.mean
        m0, m1 = m[: self.ode_order, ...], m[self.ode_order, ...]
        bias = m1 - vector_field(*m0, t=t, p=p)
        cov_sqrtm_lower = x.cov_sqrtm_lower[self.ode_order, ...]

        l_obs = jnp.reshape(
            _sqrtm.sqrtm_to_upper_triangular(R=cov_sqrtm_lower[:, None]), ()
        )
        res_white = (bias / l_obs) / jnp.sqrt(bias.size)

        # jnp.sqrt(\|res_white\|^2/d) without forming the square
        output_scale_sqrtm = jnp.reshape(
            _sqrtm.sqrtm_to_upper_triangular(R=res_white[:, None]), ()
        )

        error_estimate = l_obs
        return output_scale_sqrtm * error_estimate, output_scale_sqrtm, (bias,)

    def complete_correction(self, extrapolated, cache):
        (bias,) = cache

        m_ext, l_ext = extrapolated.mean, extrapolated.cov_sqrtm_lower
        l_obs = l_ext[self.ode_order, ...]

        l_obs_scalar = jnp.reshape(
            _sqrtm.sqrtm_to_upper_triangular(R=l_obs[:, None]), ()
        )
        c_obs = l_obs_scalar**2

        observed = _ssv.IsoNormal(mean=bias, cov_sqrtm_lower=l_obs_scalar)

        g = (l_ext @ l_obs.T) / c_obs  # shape (n,)
        m_cor = m_ext - g[:, None] * bias[None, :]
        l_cor = l_ext - g[:, None] * l_obs[None, :]
        corrected = _ssv.IsoNormal(mean=m_cor, cov_sqrtm_lower=l_cor)
        return observed, (corrected, g)
