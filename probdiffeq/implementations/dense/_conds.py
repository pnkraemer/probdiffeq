"""Conditionals."""


import jax

from probdiffeq import _sqrt_util
from probdiffeq.implementations import _collections
from probdiffeq.implementations.dense import _vars


@jax.tree_util.register_pytree_node_class
class DenseConditional(_collections.AbstractConditional):
    def __init__(self, transition, noise, target_shape):
        self.transition = transition
        self.noise = noise
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
        cond = _vars.DenseNormal(m, self.noise.cov_sqrtm_lower)
        return _vars.DenseStateSpaceVar(
            cond, cache=None, target_shape=self.target_shape
        )

    def scale_covariance(self, scale_sqrtm):
        noise = self.noise.scale_covariance(scale_sqrtm=scale_sqrtm)
        shape = self.target_shape
        return DenseConditional(self.transition, noise=noise, target_shape=shape)

    def merge_with_incoming_conditional(self, incoming, /):
        A = self.transition
        (b, B_sqrtm_lower) = self.noise.mean, self.noise.cov_sqrtm_lower

        C = incoming.transition
        (d, D_sqrtm) = (incoming.noise.mean, incoming.noise.cov_sqrtm_lower)

        g = A @ C
        xi = A @ d + b
        Xi = _sqrt_util.sum_of_sqrtm_factors(
            R_stack=((A @ D_sqrtm).T, B_sqrtm_lower.T)
        ).T

        noise = _vars.DenseNormal(mean=xi, cov_sqrtm_lower=Xi)
        return DenseConditional(g, noise=noise, target_shape=self.target_shape)

    def marginalise(self, rv, /):
        # Pull into preconditioned space
        m0_p = rv.hidden_state.mean
        l0_p = rv.hidden_state.cov_sqrtm_lower

        # Apply transition
        m_new_p = self.transition @ m0_p + self.noise.mean
        l_new_p = _sqrt_util.sum_of_sqrtm_factors(
            R_stack=((self.transition @ l0_p).T, self.noise.cov_sqrtm_lower.T)
        ).T

        # Push back into non-preconditioned space
        m_new = m_new_p
        l_new = l_new_p

        marg = _vars.DenseNormal(m_new, l_new)
        return _vars.DenseStateSpaceVar(marg, cache=None, target_shape=rv.target_shape)
