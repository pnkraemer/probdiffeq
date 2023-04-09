"""Variables."""

import jax
import jax.numpy as jnp

from probdiffeq import _sqrt_util
from probdiffeq.implementations import _collections
from probdiffeq.implementations.iso import _conds


@jax.tree_util.register_pytree_node_class
class IsoStateSpaceVar(_collections.StateSpaceVar):
    def observe_qoi(self, observation_std):
        hc = self.hidden_state.cov_sqrtm_lower[0, ...].reshape((1, -1))
        m_obs = self.hidden_state.mean[0, ...]

        r_x_f = hc.T
        r_x = self.hidden_state.cov_sqrtm_lower.T
        r_yx = observation_std * jnp.ones((1, 1))
        r_obs, (r_cor, gain) = _sqrt_util.revert_conditional(
            R_X_F=r_x_f, R_X=r_x, R_YX=r_yx
        )
        r_obs = jnp.reshape(r_obs, ())
        gain = jnp.reshape(gain, (-1,))

        m_cor = self.hidden_state.mean - gain[:, None] * m_obs[None, :]
        obs = IsoNormalQOI(m_obs, r_obs.T)
        cor = IsoNormalHiddenState(m_cor, r_cor.T)
        cond = _conds.IsoConditionalQOI(gain, noise=cor)
        return obs, cond

    def extract_qoi(self) -> jax.Array:
        return self.hidden_state.mean[..., 0, :]

    def extract_qoi_from_sample(self, u, /) -> jax.Array:
        return u[..., 0, :]

    def scale_covariance(self, output_scale):
        rv = self.hidden_state.scale_covariance(output_scale=output_scale)
        return IsoStateSpaceVar(rv, cache=self.cache)

    def marginal_nth_derivative(self, n):
        # if the variable has batch-axes, vmap the result
        if self.hidden_state.mean.ndim > 2:
            fn = IsoStateSpaceVar.marginal_nth_derivative
            vect_fn = jax.vmap(fn, in_axes=(0, None))
            return vect_fn(self, n)

        if n >= self.hidden_state.mean.shape[0]:
            msg = f"The {n}th derivative is not available in the state-variable."
            raise ValueError(msg)

        mean = self.hidden_state.mean[n, :]
        cov_sqrtm_lower_nonsquare = self.hidden_state.cov_sqrtm_lower[n, :]
        R = cov_sqrtm_lower_nonsquare[:, None]
        cov_sqrtm_lower_square = _sqrt_util.sqrtm_to_upper_triangular(R=R)
        cov_sqrtm_lower = jnp.reshape(cov_sqrtm_lower_square, ())
        return IsoNormalQOI(mean=mean, cov_sqrtm_lower=cov_sqrtm_lower)


@jax.tree_util.register_pytree_node_class
class IsoNormalHiddenState(_collections.AbstractNormal):
    def logpdf(self, u, /) -> jax.Array:
        raise NotImplementedError

    def mahalanobis_norm(self, u, /) -> jax.Array:
        raise NotImplementedError

    def scale_covariance(self, output_scale):
        cov_sqrtm_lower = output_scale[..., None, None] * self.cov_sqrtm_lower
        return IsoNormalHiddenState(mean=self.mean, cov_sqrtm_lower=cov_sqrtm_lower)

    def transform_unit_sample(self, base, /) -> jax.Array:
        return self.mean + self.cov_sqrtm_lower @ base


@jax.tree_util.register_pytree_node_class
class IsoNormalQOI(_collections.AbstractNormal):
    def logpdf(self, u, /) -> jax.Array:
        x1 = self.mahalanobis_norm_squared(u)
        x2 = u.size * 2.0 * jnp.log(jnp.abs(self.cov_sqrtm_lower))
        x3 = u.size * jnp.log(jnp.pi * 2)
        return -0.5 * (x1 + x2 + x3)

    def mahalanobis_norm_squared(self, u, /) -> jax.Array:
        r"""Compute \|x - m\|_{C^{-1}}^2."""
        # not via norm()^2, for better differentiability
        res_white = self.residual_white(u)
        return jnp.dot(res_white, res_white)

    def mahalanobis_norm(self, u, /) -> jax.Array:
        r"""Compute \|x - m\|_{C^{-1}}."""
        # sqrt(dot(res, res.T)) without forming the dot product explicitly
        res_white = self.residual_white(u)
        res_white_squeeze = jnp.linalg.qr(res_white[:, None], mode="r")
        return jnp.reshape(jnp.abs(res_white_squeeze), ())

    def residual_white(self, u, /):
        obs_pt, l_obs = self.mean, self.cov_sqrtm_lower
        res_white = (obs_pt - u) / l_obs
        return res_white

    def scale_covariance(self, output_scale):
        cov_sqrtm_lower = output_scale[..., None] * self.cov_sqrtm_lower
        return IsoNormalQOI(mean=self.mean, cov_sqrtm_lower=cov_sqrtm_lower)

    def transform_unit_sample(self, base, /) -> jax.Array:
        raise NotImplementedError

    def marginal_std(self):
        return self.cov_sqrtm_lower
