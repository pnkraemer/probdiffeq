"""Variables."""

import functools

import jax
import jax.numpy as jnp

from probdiffeq import _sqrt_util
from probdiffeq.statespace import variables


def merge_conditionals(prev: "DenseConditional", incoming: "DenseConditional", /):
    A = prev.transition
    (b, B_sqrtm_lower) = prev.noise.mean, prev.noise.cov_sqrtm_lower

    C = incoming.transition
    (d, D_sqrtm) = (incoming.noise.mean, incoming.noise.cov_sqrtm_lower)

    g = A @ C
    xi = A @ d + b
    R_stack = ((A @ D_sqrtm).T, B_sqrtm_lower.T)
    Xi = _sqrt_util.sum_of_sqrtm_factors(R_stack=R_stack).T

    noise = DenseNormal(mean=xi, cov_sqrtm_lower=Xi, target_shape=prev.target_shape)
    return DenseConditional(g, noise=noise, target_shape=prev.target_shape)


def identity_conditional(n, d):
    op = jnp.eye(n * d)

    m0 = jnp.zeros((n * d,))
    cholesky0 = jnp.zeros((n * d, n * d))
    noise = DenseNormal(m0, cholesky0, target_shape=(n, d))
    return DenseConditional(op, noise, target_shape=(n, d))


def marginalise_deterministic(rv, trafo):
    A, b = trafo
    mean, cov_sqrtm_lower = rv.mean, rv.cov_sqrtm_lower
    cov_sqrtm_lower_new = _sqrt_util.triu_via_qr(A(cov_sqrtm_lower).T).T
    return DenseNormal(A(mean) + b, cov_sqrtm_lower_new, target_shape=None)


def marginalise_stochastic(rv, conditional):
    A, noise = conditional

    R_stack = (A(rv.cov_sqrtm_lower).T, noise.cov_sqrtm_lower.T)
    cov_sqrtm_lower_new = _sqrt_util.sum_of_sqrtm_factors(R_stack=R_stack).T
    return DenseNormal(A(rv.mean) + noise.mean, cov_sqrtm_lower_new, target_shape=None)


def revert_deterministic(rv, trafo):
    # Extract information
    A, b = trafo
    mean, cov_sqrtm_lower = rv.mean, rv.cov_sqrtm_lower

    # QR-decomposition
    # (todo: rename revert_conditional_noisefree to revert_transformation_cov_sqrt())
    r_obs, (r_cor, gain) = _sqrt_util.revert_conditional_noisefree(
        R_X_F=A(cov_sqrtm_lower).T, R_X=cov_sqrtm_lower.T
    )

    # Gather terms and return
    m_cor = mean - gain @ (A(mean) + b)
    corrected = DenseNormal(m_cor, r_cor.T, target_shape=rv.target_shape)
    observed = DenseNormal(A(mean) + b, r_obs.T, target_shape=None)
    return observed, (corrected, gain)


def revert_stochastic(rv, conditional):
    # Extract information
    A, noise = conditional
    mean, cov_sqrtm_lower = rv.mean, rv.cov_sqrtm_lower

    # QR-decomposition
    # (todo: rename revert_conditional_noisefree to revert_transformation_cov_sqrt())
    r_obs, (r_cor, gain) = _sqrt_util.revert_conditional(
        R_X_F=A(cov_sqrtm_lower).T, R_X=cov_sqrtm_lower.T, R_YX=noise.cov_sqrtm_lower.T
    )

    # Gather terms and return
    m_cor = mean - gain @ (A(mean) + noise.mean)
    corrected = DenseNormal(m_cor, r_cor.T, target_shape=rv.target_shape)
    observed = DenseNormal(A(mean) + noise.mean, r_obs.T, target_shape=None)
    return observed, (corrected, gain)


@jax.tree_util.register_pytree_node_class
class DenseConditional(variables.Conditional):
    """Conditional distribution with dense covariance structure."""

    def __init__(self, *args, target_shape, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_shape = target_shape

    def __repr__(self):
        name = self.__class__.__name__
        args1 = f"transition={self.transition}, noise={self.noise}"
        args2 = f"target_shape={self.target_shape}"
        return f"{name}({args1}, {args2})"

    def tree_flatten(self):
        children = self.transition, self.noise
        aux = (self.target_shape,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        transition, noise = children
        (target_shape,) = aux
        return cls(transition=transition, noise=noise, target_shape=target_shape)

    def __call__(self, x, /):
        m = self.transition @ x + self.noise.mean
        return DenseNormal(
            m, self.noise.cov_sqrtm_lower, target_shape=self.target_shape
        )

    def scale_covariance(self, output_scale):
        noise = self.noise.scale_covariance(output_scale=output_scale)
        shape = self.target_shape
        return DenseConditional(self.transition, noise=noise, target_shape=shape)

    def marginalise(self, rv, /):
        # Pull into preconditioned space
        m0_p = rv.mean
        l0_p = rv.cov_sqrtm_lower

        # Apply transition
        m_new_p = self.transition @ m0_p + self.noise.mean
        l_new_p = _sqrt_util.sum_of_sqrtm_factors(
            R_stack=((self.transition @ l0_p).T, self.noise.cov_sqrtm_lower.T)
        ).T

        # Push back into non-preconditioned space
        m_new = m_new_p
        l_new = l_new_p

        marg = DenseNormal(m_new, l_new, target_shape=self.target_shape)
        return marg


@jax.tree_util.register_pytree_node_class
class DenseSSV(variables.SSV):
    """State-space variable with dense covariance structure."""

    def __repr__(self):
        return f"{self.__class__.__name__}(hidden_state={self.hidden_state})"

    # todo: move to DenseConditional(H=E0, noise=noise).observe()
    def observe_qoi(self, observation_std):
        hc = self._select_derivative_vect(self.hidden_state.cov_sqrtm_lower, 0)
        m_obs = self._select_derivative(self.hidden_state.mean, 0)

        r_yx = observation_std * jnp.eye(self.target_shape[1])
        r_obs, (r_cor, gain) = _sqrt_util.revert_conditional(
            R_X_F=hc.T, R_X=self.hidden_state.cov_sqrtm_lower.T, R_YX=r_yx
        )
        m_cor = self.hidden_state.mean - gain @ m_obs
        obs = DenseNormal(m_obs, r_obs.T, target_shape=None)

        noise = DenseNormal(m_cor, r_cor.T, target_shape=self.target_shape)
        cor = DenseConditional(gain, noise=noise, target_shape=self.target_shape)
        return obs, cor

    def _select_derivative_vect(self, x, i):
        fn = functools.partial(self._select_derivative, i=i)
        select = jax.vmap(fn, in_axes=1, out_axes=1)
        return select(x)

    def _select_derivative(self, x, i):
        x_reshaped = jnp.reshape(x, self.target_shape, order="F")
        return x_reshaped[i, ...]


@jax.tree_util.register_pytree_node_class
class DenseNormal(variables.Normal):
    """Random variables with a normal distribution.

    You can think of this as a traditional multivariate normal distribution.
    However, it is more of a matrix-normal distribution:
    The mean is a (d*n,)-shaped array that represents a (d,n)-shaped matrix.
    """

    def __init__(self, *args, target_shape, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_shape = target_shape

    def tree_flatten(self):
        children = self.mean, self.cov_sqrtm_lower
        aux = self.target_shape
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        target_shape = aux
        m, l_ = children
        return cls(mean=m, cov_sqrtm_lower=l_, target_shape=target_shape)

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
        return DenseNormal(
            mean=self.mean, cov_sqrtm_lower=cov_scaled, target_shape=self.target_shape
        )

    # automatically batched because of numpy's broadcasting rules?
    def transform_unit_sample(self, base, /):
        m, l_sqrtm = self.mean, self.cov_sqrtm_lower
        return (m[..., None] + l_sqrtm @ base[..., None])[..., 0]

    @property
    def sample_shape(self):
        return self.mean.shape

    def extract_qoi_from_sample(self, u, /):
        if u.ndim == 1:
            return u.reshape(self.target_shape, order="F")[0, ...]
        return jax.vmap(self.extract_qoi_from_sample)(u)

    def marginal_nth_derivative(self, n):
        if self.mean.ndim > 1:
            # if the variable has batch-axes, vmap the result
            fn = DenseNormal.marginal_nth_derivative
            vect_fn = jax.vmap(fn, in_axes=(0, None))
            return vect_fn(self, n)

        if n >= self.target_shape[0]:
            msg = f"The {n}th derivative not available in the state-space variable."
            raise ValueError(msg)

        mean = self._select_derivative(self.mean, n)
        cov_sqrtm_lower_nonsquare = self._select_derivative_vect(
            self.cov_sqrtm_lower, n
        )
        cov_sqrtm_lower = _sqrt_util.triu_via_qr(cov_sqrtm_lower_nonsquare.T).T
        return DenseNormal(mean, cov_sqrtm_lower, target_shape=None)

    def _select_derivative_vect(self, x, i):
        fn = functools.partial(self._select_derivative, i=i)
        select = jax.vmap(fn, in_axes=1, out_axes=1)
        return select(x)

    # todo: there is a lot of duplication between SSV and NormalHiddenState
    # todo: target_shape only makes sens if DenseNormal is a hidden state.
    #  should we split the normals?
    def _select_derivative(self, x, i):
        x_reshaped = jnp.reshape(x, self.target_shape, order="F")
        return x_reshaped[i, ...]

    def cov_dense(self):
        if self.cov_sqrtm_lower.ndim > 2:
            return jax.vmap(DenseNormal.cov_dense)(self)
        return self.cov_sqrtm_lower @ self.cov_sqrtm_lower.T
