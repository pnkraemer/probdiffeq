"""Implementations for scalar initial value problems."""

import jax
import jax.numpy as jnp

from probdiffeq import _sqrt_util
from probdiffeq.statespace import _vars


@jax.tree_util.register_pytree_node_class
class ConditionalHiddenState(_vars.Conditional):
    def __call__(self, x, /):
        if self.transition.ndim > 2:
            return jax.vmap(ConditionalHiddenState.__call__)(self, x)

        m = self.transition @ x + self.noise.mean
        rv = NormalHiddenState(m, self.noise.cov_sqrtm_lower)
        return SSV(rv, cache=None)

    def scale_covariance(self, output_scale):
        noise = self.noise.scale_covariance(output_scale=output_scale)
        return ConditionalHiddenState(transition=self.transition, noise=noise)

    def merge_with_incoming_conditional(self, incoming, /):
        if self.transition.ndim > 2:
            fn = ConditionalHiddenState.merge_with_incoming_conditional
            return jax.vmap(fn)(self, incoming)

        A = self.transition
        (b, B_sqrtm_lower) = self.noise.mean, self.noise.cov_sqrtm_lower

        C = incoming.transition
        (d, D_sqrtm) = (incoming.noise.mean, incoming.noise.cov_sqrtm_lower)

        g = A @ C
        xi = A @ d + b
        Xi = _sqrt_util.sum_of_sqrtm_factors(
            R_stack=((A @ D_sqrtm).T, B_sqrtm_lower.T)
        ).T

        noise = NormalHiddenState(mean=xi, cov_sqrtm_lower=Xi)
        return ConditionalHiddenState(g, noise=noise)

    def marginalise(self, rv, /):
        # Todo: this auto-batch is a bit hacky,
        #  but single-handedly replaces the entire BatchConditional class
        if rv.mean.ndim > 1:
            return jax.vmap(ConditionalHiddenState.marginalise)(self, rv)

        m0, l0 = rv.mean, rv.cov_sqrtm_lower

        m_new = self.transition @ m0 + self.noise.mean
        l_new = _sqrt_util.sum_of_sqrtm_factors(
            R_stack=((self.transition @ l0).T, self.noise.cov_sqrtm_lower.T)
        ).T

        return NormalHiddenState(m_new, l_new)


@jax.tree_util.register_pytree_node_class
class ConditionalQOI(_vars.Conditional):
    def __call__(self, x, /):
        if self.transition.ndim > 1:
            return jax.vmap(ConditionalQOI.__call__)(self, x)
        m = self.transition * x + self.noise.mean
        return NormalHiddenState(m, self.noise.cov_sqrtm_lower)


@jax.tree_util.register_pytree_node_class
class NormalQOI(_vars.Normal):
    # Normal RV. Shapes (), (). No QOI.

    def transform_unit_sample(self, base, /):
        m, l_sqrtm = self.mean, self.cov_sqrtm_lower
        return m + l_sqrtm * base

    def condition_on_qoi_observation(self, u, /, observation_std):
        raise NotImplementedError

    def extract_qoi_from_sample(self, u, /):
        raise NotImplementedError

    def scale_covariance(self, output_scale):
        return NormalQOI(self.mean, output_scale * self.cov_sqrtm_lower)

    def logpdf(self, u, /):
        if jnp.ndim(u) > 0:
            return jax.vmap(NormalQOI.logpdf)(self, u)

        x1 = 2.0 * jnp.log(self.marginal_stds())  # logdet
        x2 = self.mahalanobis_norm_squared(u)
        x3 = jnp.log(jnp.pi * 2)
        return -0.5 * (x1 + x2 + x3)

    def marginal_stds(self):
        return jnp.abs(self.cov_sqrtm_lower)

    def mahalanobis_norm_squared(self, u, /):
        res_white_scalar = self.residual_white(u)
        return res_white_scalar**2.0

    def mahalanobis_norm(self, u, /):
        res_white_scalar = self.residual_white(u)
        return jnp.abs(res_white_scalar)

    def residual_white(self, u, /):
        obs_pt, l_obs = self.mean, self.cov_sqrtm_lower
        res_white = (obs_pt - u) / l_obs
        return res_white

    def marginal_nth_derivative(self, n):
        raise NotImplementedError


@jax.tree_util.register_pytree_node_class
class SSV(_vars.SSV):
    # Normal RV. Shapes (n,), (n,n); zeroth state is the QOI.

    def observe_qoi(self, observation_std):
        # what is this for? batched calls? If so, that seems wrong.
        #  the scalar state should not worry about the context it is called in.
        if self.hidden_state.cov_sqrtm_lower.ndim > 2:
            fn = SSV.observe_qoi
            fn_vmap = jax.vmap(fn, in_axes=(0, None), out_axes=(0, 0))
            return fn_vmap(self, observation_std)

        hc = self.hidden_state.cov_sqrtm_lower[0]
        m_obs = self.hidden_state.mean[0]

        r_yx = observation_std  # * jnp.eye(1)
        r_obs_mat, (r_cor, gain_mat) = _sqrt_util.revert_conditional(
            R_X=self.hidden_state.cov_sqrtm_lower.T,
            R_X_F=hc[:, None],
            R_YX=r_yx[None, None],
        )
        r_obs = jnp.reshape(r_obs_mat, ())
        gain = jnp.reshape(gain_mat, (-1,))

        m_cor = self.hidden_state.mean - gain * m_obs

        obs = NormalQOI(m_obs, r_obs.T)
        cor = NormalHiddenState(m_cor, r_cor.T)
        return obs, ConditionalQOI(gain, cor)

    def extract_qoi_from_sample(self, u, /):
        if u.ndim == 1:
            return u[0]
        return jax.vmap(self.extract_qoi_from_sample)(u)

    def scale_covariance(self, output_scale):
        rv = self.hidden_state.scale_covariance(output_scale=output_scale)
        return SSV(self.u, rv)

    def marginal_nth_derivative(self, n):
        if self.hidden_state.mean.ndim > 1:
            # if the variable has batch-axes, vmap the result
            fn = SSV.marginal_nth_derivative
            vect_fn = jax.vmap(fn, in_axes=(0, None))
            return vect_fn(self, n)

        if n >= self.hidden_state.mean.shape[0]:
            msg = f"The {n}th derivative not available in the state-space variable."
            raise ValueError(msg)

        mean = self.hidden_state.mean[n]
        cov_sqrtm_lower_nonsquare = self.hidden_state.cov_sqrtm_lower[n, :]
        cov_sqrtm_lower = _sqrt_util.sqrtm_to_upper_triangular(
            R=cov_sqrtm_lower_nonsquare[:, None]
        ).T
        return NormalQOI(mean=mean, cov_sqrtm_lower=jnp.reshape(cov_sqrtm_lower, ()))


@jax.tree_util.register_pytree_node_class
class NormalHiddenState(_vars.Normal):
    def logpdf(self, u, /):
        raise NotImplementedError

    def mahalanobis_norm(self, u, /):
        raise NotImplementedError

    def scale_covariance(self, output_scale):
        return NormalHiddenState(
            mean=self.mean,
            cov_sqrtm_lower=output_scale[..., None, None] * self.cov_sqrtm_lower,
        )

    def transform_unit_sample(self, base, /):
        m, l_sqrtm = self.mean, self.cov_sqrtm_lower
        return (m[..., None] + l_sqrtm @ base[..., None])[..., 0]

    def marginal_nth_derivative(self, n):
        if self.mean.ndim > 1:
            # if the variable has batch-axes, vmap the result
            fn = NormalHiddenState.marginal_nth_derivative
            vect_fn = jax.vmap(fn, in_axes=(0, None))
            return vect_fn(self, n)

        if n >= self.mean.shape[0]:
            msg = f"The {n}th derivative not available in the state-space variable."
            raise ValueError(msg)

        mean = self.mean[n]
        cov_sqrtm_lower_nonsquare = self.cov_sqrtm_lower[n, :]
        cov_sqrtm_lower = _sqrt_util.sqrtm_to_upper_triangular(
            R=cov_sqrtm_lower_nonsquare[:, None]
        ).T
        return NormalQOI(mean=mean, cov_sqrtm_lower=jnp.reshape(cov_sqrtm_lower, ()))

    def extract_qoi_from_sample(self, u, /):
        if u.ndim == 1:
            return u[0]
        return jax.vmap(self.extract_qoi_from_sample)(u)
