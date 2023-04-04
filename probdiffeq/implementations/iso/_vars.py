"""Variables."""

import jax
import jax.numpy as jnp

from probdiffeq.implementations import _collections, _sqrtm
from probdiffeq.implementations.iso import _conds


@jax.tree_util.register_pytree_node_class
class IsoStateSpaceVar(_collections.StateSpaceVar):
    def observe_qoi(self, observation_std):
        hc = self.hidden_state.cov_sqrtm_lower[0, ...].reshape((1, -1))
        m_obs = self.hidden_state.mean[0, ...]

        r_yx = observation_std * jnp.ones((1, 1))
        r_obs, (r_cor, gain) = _sqrtm.revert_conditional(
            R_X_F=hc.T, R_X=self.hidden_state.cov_sqrtm_lower.T, R_YX=r_yx
        )
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

    def scale_covariance(self, scale_sqrtm):
        rv = self.hidden_state.scale_covariance(scale_sqrtm=scale_sqrtm)
        return IsoStateSpaceVar(rv)

    def marginal_nth_derivative(self, n):
        if self.hidden_state.mean.ndim > 2:
            # if the variable has batch-axes, vmap the result
            fn = IsoStateSpaceVar.marginal_nth_derivative
            vect_fn = jax.vmap(fn, in_axes=(0, None))
            return vect_fn(self, n)

        if n >= self.hidden_state.mean.shape[0]:
            msg = f"The {n}th derivative not available in the state-space variable."
            raise ValueError(msg)

        mean = self.hidden_state.mean[n, :]
        cov_sqrtm_lower_nonsquare = self.hidden_state.cov_sqrtm_lower[n, :]
        cov_sqrtm_lower = _sqrtm.sqrtm_to_upper_triangular(
            R=cov_sqrtm_lower_nonsquare[:, None]
        ).T
        return IsoNormalQOI(mean=mean, cov_sqrtm_lower=jnp.reshape(cov_sqrtm_lower, ()))


@jax.tree_util.register_pytree_node_class
class IsoNormalHiddenState(_collections.AbstractNormal):
    def logpdf(self, u, /) -> jax.Array:
        raise NotImplementedError

    def mahalanobis_norm(self, u, /) -> jax.Array:
        raise NotImplementedError

    def scale_covariance(self, scale_sqrtm):
        cov_sqrtm_lower = scale_sqrtm[..., None, None] * self.cov_sqrtm_lower
        return IsoNormalHiddenState(mean=self.mean, cov_sqrtm_lower=cov_sqrtm_lower)

    def transform_unit_sample(self, base, /) -> jax.Array:
        return self.mean + self.cov_sqrtm_lower @ base

    # todo: move to conditional???
    def Ax_plus_y(self, A, x, y) -> jax.Array:
        return A @ x + y


@jax.tree_util.register_pytree_node_class
class IsoNormalQOI(_collections.AbstractNormal):
    def __init__(self, mean, cov_sqrtm_lower):
        *batch_shape, _odedim = jnp.shape(mean)
        if jnp.shape(cov_sqrtm_lower) != tuple(batch_shape):
            raise ValueError

        super().__init__(mean=mean, cov_sqrtm_lower=cov_sqrtm_lower)

    def logpdf(self, u, /) -> jax.Array:
        m_obs, l_obs = self.mean, self.cov_sqrtm_lower

        res_white = (m_obs - u) / jnp.reshape(l_obs, ())

        x1 = jnp.dot(res_white, res_white.T)
        x2 = jnp.reshape(l_obs, ()) ** 2
        x3 = res_white.size * jnp.log(jnp.pi * 2)
        return -0.5 * (x1 + x2 + x3)

    def mahalanobis_norm(self, u, /) -> jax.Array:
        obs_pt, l_obs = self.mean, self.cov_sqrtm_lower
        res_white = (obs_pt - u) / l_obs
        evidence_sqrtm = jnp.sqrt(jnp.dot(res_white, res_white.T) / res_white.size)
        return evidence_sqrtm

    def scale_covariance(self, scale_sqrtm):
        cov_sqrtm_lower = scale_sqrtm[..., None] * self.cov_sqrtm_lower
        return IsoNormalQOI(mean=self.mean, cov_sqrtm_lower=cov_sqrtm_lower)

    def transform_unit_sample(self, base, /) -> jax.Array:
        raise NotImplementedError

    # todo: move to conditional???
    def Ax_plus_y(self, A, x, y) -> jax.Array:
        raise NotImplementedError
