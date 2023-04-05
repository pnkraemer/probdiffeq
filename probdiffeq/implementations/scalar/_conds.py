"""Implementations for scalar initial value problems."""

import jax

from probdiffeq import _sqrt_util
from probdiffeq.implementations import _collections
from probdiffeq.implementations.scalar import _vars

# todo: make public and split into submodules


@jax.tree_util.register_pytree_node_class
class _Conditional(_collections.AbstractConditional):
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
class ConditionalHiddenState(_Conditional):
    def __call__(self, x, /):
        if self.transition.ndim > 2:
            return jax.vmap(ConditionalHiddenState.__call__)(self, x)

        m = self.transition @ x + self.noise.mean
        return _vars.StateSpaceVar(
            _vars.NormalHiddenState(m, self.noise.cov_sqrtm_lower)
        )

    def scale_covariance(self, scale_sqrtm):
        noise = self.noise.scale_covariance(scale_sqrtm=scale_sqrtm)
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
        if rv.hidden_state.mean.ndim > 1:
            return jax.vmap(ConditionalHiddenState.marginalise)(self, rv)

        m0, l0 = rv.hidden_state.mean, rv.hidden_state.cov_sqrtm_lower

        m_new = self.transition @ m0 + self.noise.mean
        l_new = _sqrt_util.sum_of_sqrtm_factors(
            R_stack=((self.transition @ l0).T, self.noise.cov_sqrtm_lower.T)
        ).T

        return _vars.StateSpaceVar(_vars.NormalHiddenState(m_new, l_new))


@jax.tree_util.register_pytree_node_class
class ConditionalQOI(_Conditional):
    def __call__(self, x, /):
        if self.transition.ndim > 1:
            return jax.vmap(ConditionalQOI.__call__)(self, x)
        m = self.transition * x + self.noise.mean
        return _vars.StateSpaceVar(
            _vars.NormalHiddenState(m, self.noise.cov_sqrtm_lower)
        )
