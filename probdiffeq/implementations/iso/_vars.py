"""Variables."""

import jax
import jax.numpy as jnp

from probdiffeq.implementations import _collections, _sqrtm


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

        obs = IsoNormal(m_obs, r_obs.T)
        cond = IsoConditionalQOI(gain, noise=IsoNormal(m_cor, r_cor.T))
        return obs, cond

    def extract_qoi(self) -> jax.Array:
        return self.hidden_state.mean[..., 0, :]

    def extract_qoi_from_sample(self, u, /) -> jax.Array:
        return u[..., 0, :]

    def scale_covariance(self, scale_sqrtm):
        return IsoStateSpaceVar(
            self.hidden_state.scale_covariance(scale_sqrtm=scale_sqrtm)
        )

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
        return IsoNormal(mean=mean, cov_sqrtm_lower=jnp.reshape(cov_sqrtm_lower, ()))


@jax.tree_util.register_pytree_node_class
class IsoNormal(_collections.AbstractNormal):
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

    def scale_covariance(self, scale_sqrtm) -> "IsoNormal":
        cov_sqrtm_lower = scale_sqrtm[..., None, None] * self.cov_sqrtm_lower
        return IsoNormal(mean=self.mean, cov_sqrtm_lower=cov_sqrtm_lower)

    def transform_unit_sample(self, base, /) -> jax.Array:
        return self.mean + self.cov_sqrtm_lower @ base

    # todo: move to conditional???
    def Ax_plus_y(self, A, x, y) -> jax.Array:
        return A @ x + y


@jax.tree_util.register_pytree_node_class
class _IsoConditional(_collections.AbstractConditional):
    def __init__(self, transition, noise):
        self.transition = transition
        self.noise = noise

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(transition={self.transition}, noise={self.noise})"

    def tree_flatten(self):
        children = self.transition, self.noise
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        transition, noise = children
        return cls(transition=transition, noise=noise)


@jax.tree_util.register_pytree_node_class
class IsoConditionalHiddenState(_IsoConditional):
    # Conditional between two hidden states and QOI
    def __call__(self, x, /):
        m = self.transition @ x + self.noise.mean
        return IsoStateSpaceVar(IsoNormal(m, self.noise.cov_sqrtm_lower))

    def merge_with_incoming_conditional(self, incoming, /):
        A = self.transition
        (b, B_sqrtm) = self.noise.mean, self.noise.cov_sqrtm_lower

        C = incoming.transition
        (d, D_sqrtm) = (incoming.noise.mean, incoming.noise.cov_sqrtm_lower)

        g = A @ C
        xi = A @ d + b
        Xi = _sqrtm.sum_of_sqrtm_factors(R_stack=((A @ D_sqrtm).T, B_sqrtm.T)).T

        noise = IsoNormal(mean=xi, cov_sqrtm_lower=Xi)
        bw_model = IsoConditionalHiddenState(g, noise=noise)
        return bw_model

    def marginalise(self, rv: IsoStateSpaceVar, /) -> IsoStateSpaceVar:
        """Marginalise the output of a linear model."""
        # Read
        m0 = rv.hidden_state.mean
        l0 = rv.hidden_state.cov_sqrtm_lower

        # Apply transition
        m_new = self.transition @ m0 + self.noise.mean
        l_new = _sqrtm.sum_of_sqrtm_factors(
            R_stack=((self.transition @ l0).T, self.noise.cov_sqrtm_lower.T)
        ).T

        return IsoStateSpaceVar(IsoNormal(mean=m_new, cov_sqrtm_lower=l_new))

    def scale_covariance(self, scale_sqrtm):
        noise = self.noise.scale_covariance(scale_sqrtm=scale_sqrtm)
        return IsoConditionalHiddenState(transition=self.transition, noise=noise)


@jax.tree_util.register_pytree_node_class
class IsoConditionalQOI(_IsoConditional):
    # Conditional between hidden state and QOI
    def __call__(self, x, /):
        mv = self.transition[:, None] * x[None, :]
        m = mv + self.noise.mean
        return IsoStateSpaceVar(IsoNormal(m, self.noise.cov_sqrtm_lower))
