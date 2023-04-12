"""Variables."""

import functools

import jax
import jax.numpy as jnp

from probdiffeq import _sqrt_util
from probdiffeq.ssm import _collections
from probdiffeq.ssm.dense import _conds


@jax.tree_util.register_pytree_node_class
class DenseStateSpaceVar(_collections.StateSpaceVar):
    """State-space variable with dense covariance structure."""

    def __init__(self, hidden_state, *, cache, target_shape):
        super().__init__(hidden_state=hidden_state, cache=cache)
        self.target_shape = target_shape

    def tree_flatten(self):
        children = (self.hidden_state, self.cache)
        aux = (self.target_shape,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (hidden_state, cache) = children
        (target_shape,) = aux
        return cls(hidden_state=hidden_state, cache=cache, target_shape=target_shape)

    def __repr__(self):
        return f"{self.__class__.__name__}(hidden_state={self.hidden_state})"

    # todo: move to _conds.DenseConditional(H=E0, noise=noise).observe()
    def observe_qoi(self, observation_std):
        hc = self._select_derivative_vect(self.hidden_state.cov_sqrtm_lower, 0)
        m_obs = self._select_derivative(self.hidden_state.mean, 0)

        r_yx = observation_std * jnp.eye(self.target_shape[1])
        r_obs, (r_cor, gain) = _sqrt_util.revert_conditional(
            R_X_F=hc.T, R_X=self.hidden_state.cov_sqrtm_lower.T, R_YX=r_yx
        )
        m_cor = self.hidden_state.mean - gain @ m_obs
        obs = DenseNormal(m_obs, r_obs.T)

        noise = DenseNormal(m_cor, r_cor.T)
        cor = _conds.DenseConditional(gain, noise=noise, target_shape=self.target_shape)
        return obs, cor

    def extract_qoi(self):
        if self.hidden_state.mean.ndim == 1:
            return self._select_derivative(self.hidden_state.mean, i=0)

        select_fn = jax.vmap(self._select_derivative, in_axes=(0, None))
        return select_fn(self.hidden_state.mean, 0)

    def extract_qoi_from_sample(self, u, /):
        if u.ndim == 1:
            return u.reshape(self.target_shape, order="F")[0, ...]
        return jax.vmap(self.extract_qoi_from_sample)(u)

    def scale_covariance(self, output_scale):
        rv = self.hidden_state.scale_covariance(output_scale)
        return DenseStateSpaceVar(rv, cache=self.cache, target_shape=self.target_shape)

    def marginal_nth_derivative(self, n):
        if self.hidden_state.mean.ndim > 1:
            # if the variable has batch-axes, vmap the result
            fn = DenseStateSpaceVar.marginal_nth_derivative
            vect_fn = jax.vmap(fn, in_axes=(0, None))
            return vect_fn(self, n)

        if n >= self.target_shape[0]:
            msg = f"The {n}th derivative not available in the state-space variable."
            raise ValueError(msg)

        mean = self._select_derivative(self.hidden_state.mean, n)
        cov_sqrtm_lower_nonsquare = self._select_derivative_vect(
            self.hidden_state.cov_sqrtm_lower, n
        )
        cov_sqrtm_lower = _sqrt_util.sqrtm_to_upper_triangular(
            R=cov_sqrtm_lower_nonsquare.T
        ).T
        return DenseNormal(mean, cov_sqrtm_lower)

    def _select_derivative_vect(self, x, i):
        fn = functools.partial(self._select_derivative, i=i)
        select = jax.vmap(fn, in_axes=1, out_axes=1)
        return select(x)

    def _select_derivative(self, x, i):
        x_reshaped = jnp.reshape(x, self.target_shape, order="F")
        return x_reshaped[i, ...]


@jax.tree_util.register_pytree_node_class
class DenseNormal(_collections.AbstractNormal):
    """Random variables with a normal distribution.

    You can think of this as a traditional multivariate normal distribution.
    However, it is more of a matrix-normal distribution:
    The mean is a (d*n,)-shaped array that represents a (d,n)-shaped matrix.
    """

    def logpdf(self, u, /):
        # todo: cache those?
        diagonal = jnp.diagonal(self.cov_sqrtm_lower, axis1=-1, axis2=-2)
        slogdet = jnp.sum(jnp.log(jnp.abs(diagonal)))

        x1 = self.mahalanobis_norm_squared(u)
        x2 = 2.0 * slogdet
        x3 = u.size * jnp.log(jnp.pi * 2)
        return -0.5 * (x1 + x2 + x3)

    def mahalanobis_norm_squared(self, u, /):
        res_white = self.residual_white(u)
        return jnp.dot(res_white, res_white)

    def mahalanobis_norm(self, u, /):
        res_white = self.residual_white(u)
        norm_square = jnp.linalg.qr(res_white[:, None], mode="r")
        return jnp.reshape(jnp.abs(norm_square), ())

    def marginal_stds(self):
        def std(x):
            std_mat = jnp.linalg.qr(x[..., None], mode="r")
            return jnp.reshape(std_mat, ())

        return jax.vmap(std)(self.cov_sqrtm_lower)

    def residual_white(self, u, /):
        obs_pt, l_obs = self.mean, self.cov_sqrtm_lower
        return jax.scipy.linalg.solve_triangular(
            l_obs.T, u - obs_pt, lower=False, trans="T"
        )

    def scale_covariance(self, output_scale):
        cov_scaled = output_scale[..., None, None] * self.cov_sqrtm_lower
        return DenseNormal(mean=self.mean, cov_sqrtm_lower=cov_scaled)

    # automatically batched because of numpy's broadcasting rules?
    def transform_unit_sample(self, base, /):
        m, l_sqrtm = self.mean, self.cov_sqrtm_lower
        return (m[..., None] + l_sqrtm @ base[..., None])[..., 0]

    @property
    def sample_shape(self):
        return self.mean.shape
