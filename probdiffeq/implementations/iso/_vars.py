import jax
import jax.numpy as jnp

from probdiffeq.implementations import _collections, _sqrtm


@jax.tree_util.register_pytree_node_class
class IsoNormal(_collections.StateSpaceVariable):
    def __init__(self, mean, cov_sqrtm_lower):
        self.mean = mean  # (n, d) shape
        self.cov_sqrtm_lower = cov_sqrtm_lower  # (n, n) shape

    def tree_flatten(self):
        children = self.mean, self.cov_sqrtm_lower
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        mean, cov_sqrtm_lower = children
        return cls(mean=mean, cov_sqrtm_lower=cov_sqrtm_lower)

    def logpdf(self, u, /):
        m_obs, l_obs = self.mean, self.cov_sqrtm_lower

        res_white = (m_obs - u) / jnp.reshape(l_obs, ())

        x1 = jnp.dot(res_white, res_white.T)
        x2 = jnp.reshape(l_obs, ()) ** 2
        x3 = res_white.size * jnp.log(jnp.pi * 2)
        return -0.5 * (x1 + x2 + x3)

    def norm_of_whitened_residual_sqrtm(self):
        obs_pt, l_obs = self.mean, self.cov_sqrtm_lower
        res_white = obs_pt / l_obs
        evidence_sqrtm = jnp.sqrt(jnp.dot(res_white, res_white.T) / res_white.size)
        return evidence_sqrtm

    def condition_on_qoi_observation(self, u, /, observation_std):
        hc = self.cov_sqrtm_lower[0, ...].reshape((1, -1))
        m_obs = self.mean[0, ...]

        r_yx = observation_std * jnp.ones((1, 1))
        r_obs, (r_cor, gain) = _sqrtm.revert_conditional(
            R_X_F=hc.T, R_X=self.cov_sqrtm_lower.T, R_YX=r_yx
        )
        m_cor = self.mean - gain * (m_obs - u)[None, :]

        return IsoNormal(m_obs, r_obs.T), (IsoNormal(m_cor, r_cor.T), gain)

    def extract_qoi(self):
        m = self.mean[..., 0, :]
        return m

    def extract_qoi_from_sample(self, u, /):
        return u[..., 0, :]

    # todo: split those functions into a batch and a non-batch version?

    def scale_covariance(self, scale_sqrtm):
        if jnp.ndim(scale_sqrtm) == 0:
            return IsoNormal(
                mean=self.mean, cov_sqrtm_lower=scale_sqrtm * self.cov_sqrtm_lower
            )
        return IsoNormal(
            mean=self.mean,
            cov_sqrtm_lower=scale_sqrtm[:, None, None] * self.cov_sqrtm_lower,
        )

    def transform_unit_sample(self, base, /):
        return self.mean + self.cov_sqrtm_lower @ base

    def Ax_plus_y(self, A, x, y):
        return A @ x + y

    @property
    def sample_shape(self):
        return self.mean.shape
