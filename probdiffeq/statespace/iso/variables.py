"""Variables."""

import jax
import jax.numpy as jnp

from probdiffeq import _markov, _sqrt_util
from probdiffeq.statespace import variables


def unit_markov_sequence(**kwargs) -> _markov.MarkovSeqRev:
    rv0 = standard_normal(**kwargs)
    cond0 = identity_conditional(**kwargs)
    return _markov.MarkovSeqRev(init=rv0, conditional=cond0)


def identity_conditional(num_derivatives, ode_shape) -> "IsoConditionalHiddenState":
    assert len(ode_shape) == 1
    (d,) = ode_shape

    op = jnp.eye(num_derivatives + 1)

    m0 = jnp.zeros((num_derivatives + 1, d))
    c0 = jnp.zeros((num_derivatives + 1, num_derivatives + 1))
    noise = IsoNormalHiddenState(m0, c0)
    return IsoConditionalHiddenState(op, noise=noise)


def standard_normal(*, num_derivatives, ode_shape) -> "IsoNormalHiddenState":
    assert len(ode_shape) == 1
    (d,) = ode_shape

    m0 = jnp.zeros((num_derivatives + 1, d))
    c0 = jnp.eye(num_derivatives + 1)
    return IsoNormalHiddenState(m0, c0)


def merge_conditionals(previous, incoming, /):
    A = previous.transition
    (b, B_sqrtm_lower) = previous.noise.mean, previous.noise.cov_sqrtm_lower

    C = incoming.transition
    (d, D_sqrtm) = (incoming.noise.mean, incoming.noise.cov_sqrtm_lower)

    g = A @ C
    xi = A @ d + b
    R_stack = ((A @ D_sqrtm).T, B_sqrtm_lower.T)
    Xi = _sqrt_util.sum_of_sqrtm_factors(R_stack=R_stack).T

    noise = IsoNormalHiddenState(mean=xi, cov_sqrtm_lower=Xi)
    bw_model = IsoConditionalHiddenState(g, noise=noise)
    return bw_model


@jax.tree_util.register_pytree_node_class
class IsoConditionalHiddenState(variables.Conditional):
    # Conditional between two hidden states and QOI
    def __call__(self, x, /):
        m = self.transition @ x + self.noise.mean
        return IsoNormalHiddenState(m, self.noise.cov_sqrtm_lower)

    def marginalise(self, rv, /):
        """Marginalise the output of a linear model."""
        # Read
        m0 = rv.mean
        l0 = rv.cov_sqrtm_lower

        # Apply transition
        m_new = self.transition @ m0 + self.noise.mean
        l_new = _sqrt_util.sum_of_sqrtm_factors(
            R_stack=((self.transition @ l0).T, self.noise.cov_sqrtm_lower.T)
        ).T

        return IsoNormalHiddenState(mean=m_new, cov_sqrtm_lower=l_new)

    def scale_covariance(self, output_scale):
        noise = self.noise.scale_covariance(output_scale=output_scale)
        return IsoConditionalHiddenState(transition=self.transition, noise=noise)


@jax.tree_util.register_pytree_node_class
class IsoConditionalQOI(variables.Conditional):
    # Conditional between hidden state and QOI
    def __call__(self, x, /):
        mv = self.transition[:, None] * x[None, :]
        m = mv + self.noise.mean
        return IsoNormalHiddenState(m, self.noise.cov_sqrtm_lower)


@jax.tree_util.register_pytree_node_class
class IsoSSV(variables.SSV):
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
        cond = IsoConditionalQOI(gain, noise=cor)
        return obs, cond

    def extract_qoi_from_sample(self, u, /) -> jax.Array:
        return u[..., 0, :]

    def scale_covariance(self, output_scale):
        rv = self.hidden_state.scale_covariance(output_scale=output_scale)
        return IsoSSV(self.u, rv)

    def marginal_nth_derivative(self, n):
        # if the variable has batch-axes, vmap the result
        if self.hidden_state.mean.ndim > 2:
            fn = IsoSSV.marginal_nth_derivative
            vect_fn = jax.vmap(fn, in_axes=(0, None))
            return vect_fn(self, n)

        if n >= self.hidden_state.mean.shape[0]:
            msg = f"The {n}th derivative is not available in the state-variable."
            raise ValueError(msg)

        mean = self.hidden_state.mean[n, :]
        cov_sqrtm_lower_nonsquare = self.hidden_state.cov_sqrtm_lower[n, :]
        R = cov_sqrtm_lower_nonsquare[:, None]
        cov_sqrtm_lower_square = _sqrt_util.triu_via_qr(R)
        cov_sqrtm_lower = jnp.reshape(cov_sqrtm_lower_square, ())
        return IsoNormalQOI(mean=mean, cov_sqrtm_lower=cov_sqrtm_lower)


@jax.tree_util.register_pytree_node_class
class IsoNormalHiddenState(variables.Normal):
    def logpdf(self, u, /) -> jax.Array:
        raise NotImplementedError

    def mahalanobis_norm(self, u, /) -> jax.Array:
        raise NotImplementedError

    def scale_covariance(self, output_scale):
        cov_sqrtm_lower = output_scale[..., None, None] * self.cov_sqrtm_lower
        return IsoNormalHiddenState(mean=self.mean, cov_sqrtm_lower=cov_sqrtm_lower)

    def transform_unit_sample(self, base, /) -> jax.Array:
        return self.mean + self.cov_sqrtm_lower @ base

    def marginal_nth_derivative(self, n):
        # if the variable has batch-axes, vmap the result
        if self.mean.ndim > 2:
            fn = IsoNormalHiddenState.marginal_nth_derivative
            vect_fn = jax.vmap(fn, in_axes=(0, None))
            return vect_fn(self, n)

        if n >= self.mean.shape[0]:
            msg = f"The {n}th derivative is not available in the state-variable."
            raise ValueError(msg)

        mean = self.mean[n, :]
        cov_sqrtm_lower_nonsquare = self.cov_sqrtm_lower[n, :]
        R = cov_sqrtm_lower_nonsquare[:, None]
        cov_sqrtm_lower_square = _sqrt_util.triu_via_qr(R)
        cov_sqrtm_lower = jnp.reshape(cov_sqrtm_lower_square, ())
        return IsoNormalQOI(mean=mean, cov_sqrtm_lower=cov_sqrtm_lower)

    def extract_qoi_from_sample(self, u, /):
        return u[..., 0, :]

    def cov_dense(self):
        if self.cov_sqrtm_lower.ndim > 2:
            return jax.vmap(IsoNormalHiddenState.cov_dense)(self)
        return self.cov_sqrtm_lower @ self.cov_sqrtm_lower.T


@jax.tree_util.register_pytree_node_class
class IsoNormalQOI(variables.Normal):
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

    def marginal_nth_derivative(self, n):
        raise NotImplementedError

    def extract_qoi_from_sample(self, u, /):
        raise NotImplementedError
