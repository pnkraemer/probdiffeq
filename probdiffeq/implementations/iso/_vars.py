from typing import Tuple

import jax
import jax.numpy as jnp

from probdiffeq.implementations import _collections, _sqrtm


@jax.tree_util.register_pytree_node_class
class IsoVariable(_collections.StateSpaceVariable):
    def condition_on_qoi_observation(self, u, /, observation_std):
        hc = self.hidden_state.cov_sqrtm_lower[0, ...].reshape((1, -1))
        m_obs = self.hidden_state.mean[0, ...]

        r_yx = observation_std * jnp.ones((1, 1))
        r_obs, (r_cor, gain) = _sqrtm.revert_conditional(
            R_X_F=hc.T, R_X=self.hidden_state.cov_sqrtm_lower.T, R_YX=r_yx
        )
        m_cor = self.hidden_state.mean - gain * (m_obs - u)[None, :]

        return IsoNormal(m_obs, r_obs.T), (IsoVariable(IsoNormal(m_cor, r_cor.T)), gain)

    def extract_qoi(self) -> jax.Array:
        m = self.hidden_state.mean[..., 0, :]
        return m

    def extract_qoi_from_sample(self, u, /) -> jax.Array:
        return u[..., 0, :]

    def scale_covariance(self, scale_sqrtm):
        return IsoVariable(self.hidden_state.scale_covariance(scale_sqrtm=scale_sqrtm))


@jax.tree_util.register_pytree_node_class
class IsoNormal(_collections.RandomVariable):
    def __init__(self, mean, cov_sqrtm_lower):
        self.mean = mean  # (n, d) shape
        self.cov_sqrtm_lower = cov_sqrtm_lower  # (n, n) shape

    def __repr__(self):
        name = f"{self.__class__.__name__}"
        args = f"mean={self.mean}, cov_sqrtm_lower={self.cov_sqrtm_lower}"
        return f"{name}({args})"

    def tree_flatten(self) -> Tuple:
        children = self.mean, self.cov_sqrtm_lower
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children) -> "IsoNormal":
        mean, cov_sqrtm_lower = children
        return cls(mean=mean, cov_sqrtm_lower=cov_sqrtm_lower)

    def logpdf(self, u, /) -> jax.Array:
        m_obs, l_obs = self.mean, self.cov_sqrtm_lower

        res_white = (m_obs - u) / jnp.reshape(l_obs, ())

        x1 = jnp.dot(res_white, res_white.T)
        x2 = jnp.reshape(l_obs, ()) ** 2
        x3 = res_white.size * jnp.log(jnp.pi * 2)
        return -0.5 * (x1 + x2 + x3)

    def norm_of_whitened_residual_sqrtm(self) -> jax.Array:
        obs_pt, l_obs = self.mean, self.cov_sqrtm_lower
        res_white = obs_pt / l_obs
        evidence_sqrtm = jnp.sqrt(jnp.dot(res_white, res_white.T) / res_white.size)
        return evidence_sqrtm

    # todo: split those functions into a batch and a non-batch version?

    def scale_covariance(self, scale_sqrtm) -> "IsoNormal":
        if jnp.ndim(scale_sqrtm) == 0:
            return IsoNormal(
                mean=self.mean, cov_sqrtm_lower=scale_sqrtm * self.cov_sqrtm_lower
            )
        return IsoNormal(
            mean=self.mean,
            cov_sqrtm_lower=scale_sqrtm[:, None, None] * self.cov_sqrtm_lower,
        )

    def transform_unit_sample(self, base, /) -> jax.Array:
        return self.mean + self.cov_sqrtm_lower @ base

    def Ax_plus_y(self, A, x, y) -> jax.Array:
        return A @ x + y

    @property
    def sample_shape(self) -> Tuple[int]:
        return self.mean.shape
