"""Scalar implementations."""

import jax
import jax.numpy as jnp

from odefilter.implementations import _collections, _sqrtm


@jax.tree_util.register_pytree_node_class
class ScalarNormal(_collections.StateSpaceVariable):
    # Normal RV. Shapes (), ()

    def __init__(self, mean, cov_sqrtm_lower):
        self.mean = mean
        self.cov_sqrtm_lower = cov_sqrtm_lower

    def __repr__(self):
        name = f"{self.__class__.__name__}"
        args = f"mean={self.mean}, cov_sqrtm_lower={self.cov_sqrtm_lower}"
        return f"{name}({args})"

    def tree_flatten(self):
        children = self.mean, self.cov_sqrtm_lower
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        mean, cov_sqrtm_lower = children
        return cls(mean=mean, cov_sqrtm_lower=cov_sqrtm_lower)

    @property
    def sample_shape(self):
        return self.mean.shape

    def transform_unit_sample(self, base, /):
        m, l_sqrtm = self.mean, self.cov_sqrtm_lower
        return m + l_sqrtm * base

    def Ax_plus_y(self, A, x, y):
        return A * x + y

    def condition_on_qoi_observation(self, u, /, observation_std):
        raise NotImplementedError

    def extract_qoi(self):
        raise NotImplementedError

    def extract_qoi_from_sample(self, u, /):
        raise NotImplementedError

    def scale_covariance(self, scale_sqrtm):
        return ScalarNormal(self.mean, scale_sqrtm * self.cov_sqrtm_lower)

    def logpdf(self, u, /):
        m_obs, l_obs = self.mean, self.cov_sqrtm_lower
        res_white = (m_obs - u) / l_obs
        x1 = l_obs**2
        x2 = jnp.dot(res_white, res_white.T)
        x3 = res_white.size * jnp.log(jnp.pi * 2)
        return -0.5 * (x1 + x2 + x3)

    def norm_of_whitened_residual_sqrtm(self):
        obs_pt, l_obs = self.mean, self.cov_sqrtm_lower
        res_white = obs_pt / l_obs
        evidence_sqrtm = jnp.sqrt(jnp.dot(res_white, res_white.T) / res_white.size)
        return evidence_sqrtm


@jax.tree_util.register_pytree_node_class
class Normal(_collections.StateSpaceVariable):
    # Normal RV. Shapes (n,), (n,n); zeroth state is the QOI.

    def __init__(self, mean, cov_sqrtm_lower):
        self.mean = mean
        self.cov_sqrtm_lower = cov_sqrtm_lower

    def __repr__(self):
        name = f"{self.__class__.__name__}"
        args = f"mean={self.mean}, cov_sqrtm_lower={self.cov_sqrtm_lower}"
        return f"{name}({args})"

    def tree_flatten(self):
        children = self.mean, self.cov_sqrtm_lower
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        mean, cov_sqrtm_lower = children
        return cls(mean=mean, cov_sqrtm_lower=cov_sqrtm_lower)

    @property
    def sample_shape(self):
        return self.mean.shape

    def logpdf(self, u, /):
        m_obs, l_obs = self.mean, self.cov_sqrtm_lower
        res_white = jax.scipy.linalg.solve_triangular(l_obs.T, (m_obs - u), lower=False)
        x1 = jnp.linalg.slogdet(l_obs)[1] ** 2
        x2 = jnp.dot(res_white, res_white.T)
        x3 = res_white.size * jnp.log(jnp.pi * 2)
        return -0.5 * (x1 + x2 + x3)

    def norm_of_whitened_residual_sqrtm(self):
        obs_pt, l_obs = self.mean, self.cov_sqrtm_lower
        print(l_obs.shape, obs_pt.shape)
        res_white = jax.scipy.linalg.solve_triangular(l_obs.T, obs_pt, lower=False)
        evidence_sqrtm = jnp.sqrt(jnp.dot(res_white, res_white.T) / res_white.size)
        return evidence_sqrtm

    def condition_on_qoi_observation(self, u, /, observation_std):
        hc = self.cov_sqrtm_lower[0]
        m_obs = self.mean[0]

        r_yx = observation_std  # * jnp.eye(1)
        r_obs_mat, (r_cor, gain_mat) = _sqrtm.revert_conditional(
            R_X=self.cov_sqrtm_lower.T, R_X_F=hc[:, None], R_YX=r_yx[None, None]
        )
        r_obs = jnp.reshape(r_obs_mat, ())
        gain = jnp.reshape(gain_mat, (-1,))

        m_cor = self.mean - gain * (m_obs - u)

        obs = ScalarNormal(m_obs, r_obs.T)
        cor = Normal(m_cor, r_cor.T)
        return obs, (cor, gain)

    def extract_qoi(self):
        return self.mean[..., 0]
        # if self.mean.ndim == 1:
        #     return self.mean[0]
        # return jax.vmap(self._select_derivative, in_axes=(0, None))(self.mean, 0)

    def extract_qoi_from_sample(self, u, /):

        if u.ndim == 1:
            return u[0]
        return jax.vmap(self.extract_qoi_from_sample)(u)

    def scale_covariance(self, scale_sqrtm):
        if jnp.ndim(scale_sqrtm) == 0:
            return Normal(
                mean=self.mean,
                cov_sqrtm_lower=scale_sqrtm * self.cov_sqrtm_lower,
            )
        return Normal(
            mean=self.mean,
            cov_sqrtm_lower=scale_sqrtm[:, None, None] * self.cov_sqrtm_lower,
        )

    def transform_unit_sample(self, base, /):
        m, l_sqrtm = self.mean, self.cov_sqrtm_lower
        return (m[..., None] + l_sqrtm @ base[..., None])[..., 0]

    def Ax_plus_y(self, A, x, y):
        return A @ x + y
