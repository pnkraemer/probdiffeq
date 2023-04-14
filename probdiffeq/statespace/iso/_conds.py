"""Isotropic conditionals."""

import jax

from probdiffeq import _sqrt_util
from probdiffeq.statespace import _collections
from probdiffeq.statespace.iso import _vars


@jax.tree_util.register_pytree_node_class
class IsoConditionalHiddenState(_collections.AbstractConditional):
    # Conditional between two hidden states and QOI
    def __call__(self, x, /):
        m = self.transition @ x + self.noise.mean
        rv = _vars.IsoNormalHiddenState(m, self.noise.cov_sqrtm_lower)
        return _vars.IsoSSV(rv, cache=None)

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

        noise = _vars.IsoNormalHiddenState(mean=xi, cov_sqrtm_lower=Xi)
        bw_model = IsoConditionalHiddenState(g, noise=noise)
        return bw_model

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

        return _vars.IsoNormalHiddenState(mean=m_new, cov_sqrtm_lower=l_new)

    def scale_covariance(self, output_scale):
        noise = self.noise.scale_covariance(output_scale=output_scale)
        return IsoConditionalHiddenState(transition=self.transition, noise=noise)


@jax.tree_util.register_pytree_node_class
class IsoConditionalQOI(_collections.AbstractConditional):
    # Conditional between hidden state and QOI
    def __call__(self, x, /):
        mv = self.transition[:, None] * x[None, :]
        m = mv + self.noise.mean
        rv = _vars.IsoNormalHiddenState(m, self.noise.cov_sqrtm_lower)
        return _vars.IsoSSV(rv, cache=None)
