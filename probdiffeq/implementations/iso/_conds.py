"""Isotropic conditionals."""

import jax

from probdiffeq.implementations import _collections, _sqrtm
from probdiffeq.implementations.iso import _vars


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
        return _vars.IsoStateSpaceVar(_vars.IsoNormal(m, self.noise.cov_sqrtm_lower))

    def merge_with_incoming_conditional(self, incoming, /):
        A = self.transition
        (b, B_sqrtm) = self.noise.mean, self.noise.cov_sqrtm_lower

        C = incoming.transition
        (d, D_sqrtm) = (incoming.noise.mean, incoming.noise.cov_sqrtm_lower)

        g = A @ C
        xi = A @ d + b
        Xi = _sqrtm.sum_of_sqrtm_factors(R_stack=((A @ D_sqrtm).T, B_sqrtm.T)).T

        noise = _vars.IsoNormal(mean=xi, cov_sqrtm_lower=Xi)
        bw_model = IsoConditionalHiddenState(g, noise=noise)
        return bw_model

    def marginalise(self, rv, /):
        """Marginalise the output of a linear model."""
        # Read
        m0 = rv.hidden_state.mean
        l0 = rv.hidden_state.cov_sqrtm_lower

        # Apply transition
        m_new = self.transition @ m0 + self.noise.mean
        l_new = _sqrtm.sum_of_sqrtm_factors(
            R_stack=((self.transition @ l0).T, self.noise.cov_sqrtm_lower.T)
        ).T

        return _vars.IsoStateSpaceVar(
            _vars.IsoNormal(mean=m_new, cov_sqrtm_lower=l_new)
        )

    def scale_covariance(self, scale_sqrtm):
        noise = self.noise.scale_covariance(scale_sqrtm=scale_sqrtm)
        return IsoConditionalHiddenState(transition=self.transition, noise=noise)


@jax.tree_util.register_pytree_node_class
class IsoConditionalQOI(_IsoConditional):
    # Conditional between hidden state and QOI
    def __call__(self, x, /):
        mv = self.transition[:, None] * x[None, :]
        m = mv + self.noise.mean
        return _vars.IsoStateSpaceVar(_vars.IsoNormal(m, self.noise.cov_sqrtm_lower))
