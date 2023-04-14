"""Implementations for scalar initial value problems."""

import jax

from probdiffeq import _sqrt_util
from probdiffeq.statespace import _collections
from probdiffeq.statespace.scalar import _vars


@jax.tree_util.register_pytree_node_class
class ConditionalHiddenState(_collections.AbstractConditional):
    def __call__(self, x, /):
        if self.transition.ndim > 2:
            return jax.vmap(ConditionalHiddenState.__call__)(self, x)

        m = self.transition @ x + self.noise.mean
        rv = _vars.NormalHiddenState(m, self.noise.cov_sqrtm_lower)
        return _vars.StateSpaceVar(rv, cache=None)

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

        noise = _vars.NormalHiddenState(mean=xi, cov_sqrtm_lower=Xi)
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

        return _vars.NormalHiddenState(m_new, l_new)


@jax.tree_util.register_pytree_node_class
class ConditionalQOI(_collections.AbstractConditional):
    def __call__(self, x, /):
        if self.transition.ndim > 1:
            return jax.vmap(ConditionalQOI.__call__)(self, x)
        m = self.transition * x + self.noise.mean
        rv = _vars.NormalHiddenState(m, self.noise.cov_sqrtm_lower)
        return _vars.StateSpaceVar(rv, cache=None)
