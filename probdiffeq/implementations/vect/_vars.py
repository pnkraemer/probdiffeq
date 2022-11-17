import functools

import jax
import jax.numpy as jnp

from probdiffeq.implementations import _collections, _sqrtm


@jax.tree_util.register_pytree_node_class
class VectNormal(_collections.StateSpaceVariable):
    """Vector-normal distribution.

    You can think of this as a traditional multivariate normal distribution.
    But in fact, it is more of a matrix-normal distribution.
    This means that the mean vector is a (d*n,)-shaped array but
    represents a (d,n)-shaped matrix.
    """

    def __init__(self, mean, cov_sqrtm_lower, target_shape):
        self.mean = mean  # (n,) shape
        self.cov_sqrtm_lower = cov_sqrtm_lower  # (n, n) shape
        self.target_shape = target_shape

    def tree_flatten(self):
        children = self.mean, self.cov_sqrtm_lower
        aux = (self.target_shape,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        mean, cov_sqrtm_lower = children
        (target_shape,) = aux
        return cls(mean, cov_sqrtm_lower, target_shape=target_shape)

    # todo: extract _whiten() method?!

    def logpdf(self, u, /):
        m_obs, l_obs = self.mean, self.cov_sqrtm_lower

        res_white = jax.scipy.linalg.solve_triangular(l_obs.T, (m_obs - u), lower=False)

        x1 = jnp.dot(res_white, res_white.T)
        x2 = jnp.linalg.slogdet(l_obs)[1] ** 2
        x3 = res_white.size * jnp.log(jnp.pi * 2)
        return -0.5 * (x1 + x2 + x3)

    def norm_of_whitened_residual_sqrtm(self):
        obs_pt, l_obs = self.mean, self.cov_sqrtm_lower
        res_white = jax.scipy.linalg.solve_triangular(l_obs.T, obs_pt, lower=False)
        evidence_sqrtm = jnp.sqrt(jnp.dot(res_white, res_white.T) / res_white.size)
        return evidence_sqrtm

    def condition_on_qoi_observation(self, u, /, observation_std):
        hc = self._select_derivative_vect(self.cov_sqrtm_lower, 0)
        m_obs = self._select_derivative(self.mean, 0)

        r_yx = observation_std * jnp.eye(u.shape[0])
        r_obs, (r_cor, gain) = _sqrtm.revert_conditional(
            R_X_F=hc.T, R_X=self.cov_sqrtm_lower.T, R_YX=r_yx
        )
        m_cor = self.mean - gain @ (m_obs - u)

        obs = VectNormal(m_obs, r_obs.T, target_shape=self.target_shape)
        cor = VectNormal(m_cor, r_cor.T, target_shape=self.target_shape)
        return obs, (cor, gain)

    def extract_qoi(self):
        if self.mean.ndim == 1:
            return self._select_derivative(self.mean, i=0)
        return jax.vmap(self._select_derivative, in_axes=(0, None))(self.mean, 0)

    def extract_qoi_from_sample(self, u, /):
        if u.ndim == 1:
            return u.reshape(self.target_shape, order="F")[0, ...]
        return jax.vmap(self.extract_qoi_from_sample)(u)

    def _select_derivative_vect(self, x, i):
        fn = functools.partial(self._select_derivative, i=i)
        select = jax.vmap(fn, in_axes=1, out_axes=1)
        return select(x)

    def _select_derivative(self, x, i):
        x_reshaped = jnp.reshape(x, self.target_shape, order="F")
        return x_reshaped[i, ...]

    def scale_covariance(self, scale_sqrtm):
        if jnp.ndim(scale_sqrtm) == 0:
            return VectNormal(
                mean=self.mean,
                cov_sqrtm_lower=scale_sqrtm * self.cov_sqrtm_lower,
                target_shape=self.target_shape,
            )
        return VectNormal(
            mean=self.mean,
            cov_sqrtm_lower=scale_sqrtm[:, None, None] * self.cov_sqrtm_lower,
            target_shape=self.target_shape,
        )

    # automatically batched because of numpy's broadcasting rules?
    def transform_unit_sample(self, base, /):
        m, l_sqrtm = self.mean, self.cov_sqrtm_lower
        return (m[..., None] + l_sqrtm @ base[..., None])[..., 0]

    def Ax_plus_y(self, A, x, y):
        return A @ x + y

    @property
    def sample_shape(self):
        return self.mean.shape
