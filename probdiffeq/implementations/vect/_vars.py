import functools

import jax
import jax.numpy as jnp

from probdiffeq.implementations import _collections, _sqrtm


@jax.tree_util.register_pytree_node_class
class VectStateSpaceVar(_collections.StateSpaceVar):
    def __init__(self, hidden_state, *, target_shape):
        super().__init__(hidden_state=hidden_state)
        self.target_shape = target_shape

    def tree_flatten(self):
        children = (self.hidden_state,)
        aux = (self.target_shape,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (hidden_state,) = children
        (target_shape,) = aux
        return cls(hidden_state=hidden_state, target_shape=target_shape)

    def __repr__(self):
        return f"{self.__class__.__name__}(hidden_state={self.hidden_state})"

    def condition_on_qoi_observation(self, u, /, observation_std):
        hc = self._select_derivative_vect(self.hidden_state.cov_sqrtm_lower, 0)
        m_obs = self._select_derivative(self.hidden_state.mean, 0)

        r_yx = observation_std * jnp.eye(u.shape[0])
        r_obs, (r_cor, gain) = _sqrtm.revert_conditional(
            R_X_F=hc.T, R_X=self.hidden_state.cov_sqrtm_lower.T, R_YX=r_yx
        )
        m_cor = self.hidden_state.mean - gain @ (m_obs - u)

        obs = VectNormal(m_obs, r_obs.T)
        cor = VectStateSpaceVar(
            VectNormal(m_cor, r_cor.T), target_shape=self.target_shape
        )
        return obs, (cor, gain)

    def extract_qoi(self):
        if self.hidden_state.mean.ndim == 1:
            return self._select_derivative(self.hidden_state.mean, i=0)
        return jax.vmap(self._select_derivative, in_axes=(0, None))(
            self.hidden_state.mean, 0
        )

    def extract_qoi_from_sample(self, u, /):
        if u.ndim == 1:
            return u.reshape(self.target_shape, order="F")[0, ...]
        return jax.vmap(self.extract_qoi_from_sample)(u)

    def scale_covariance(self, scale_sqrtm):
        hidden_state_scaled = self.hidden_state.scale_covariance(scale_sqrtm)
        return VectStateSpaceVar(hidden_state_scaled, target_shape=self.target_shape)

    def _select_derivative_vect(self, x, i):
        fn = functools.partial(self._select_derivative, i=i)
        select = jax.vmap(fn, in_axes=1, out_axes=1)
        return select(x)

    def _select_derivative(self, x, i):
        x_reshaped = jnp.reshape(x, self.target_shape, order="F")
        return x_reshaped[i, ...]


@jax.tree_util.register_pytree_node_class
class VectNormal(_collections.AbstractNormal):
    """Vector-normal distribution.

    You can think of this as a traditional multivariate normal distribution.
    But in fact, it is more of a matrix-normal distribution.
    This means that the mean vector is a (d*n,)-shaped array but
    represents a (d,n)-shaped matrix.
    """

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

    #
    # def _select_derivative_vect(self, x, i):
    #     fn = functools.partial(self._select_derivative, i=i)
    #     select = jax.vmap(fn, in_axes=1, out_axes=1)
    #     return select(x)
    #
    # def _select_derivative(self, x, i):
    #     x_reshaped = jnp.reshape(x, self.target_shape, order="F")
    #     return x_reshaped[i, ...]

    def scale_covariance(self, scale_sqrtm):
        cov_scaled = scale_sqrtm[..., None, None] * self.cov_sqrtm_lower
        return VectNormal(mean=self.mean, cov_sqrtm_lower=cov_scaled)

    # automatically batched because of numpy's broadcasting rules?
    def transform_unit_sample(self, base, /):
        m, l_sqrtm = self.mean, self.cov_sqrtm_lower
        return (m[..., None] + l_sqrtm @ base[..., None])[..., 0]

    def Ax_plus_y(self, A, x, y):
        return A @ x + y

    @property
    def sample_shape(self):
        return self.mean.shape
