"""Batch-style extrapolations."""
from typing import Tuple

import jax
import jax.numpy as jnp

from probdiffeq.implementations import _collections, _ibm_util, _scalar
from probdiffeq.implementations.batch import _vars

_IBMCacheType = Tuple[jax.Array]  # Cache type
"""Type of the extrapolation-cache."""


@jax.tree_util.register_pytree_node_class
class BatchConditional(_collections.AbstractConditional):
    def __init__(self, transition, noise):
        noise = _scalar.Normal(noise.mean, noise.cov_sqrtm_lower)
        self.conditional = _scalar.Conditional(transition, noise=noise)

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(transition={self.transition}, noise={self.noise})"

    @property
    def transition(self):
        return self.conditional.transition

    @property
    def noise(self):
        return self.conditional.noise

    def tree_flatten(self):
        children = (self.conditional,)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        (conditional,) = children
        return cls(transition=conditional.transition, noise=conditional.noise)

    def __call__(self, x, /):
        out = jax.vmap(_scalar.Conditional.__call__)(self.conditional, x)
        return _vars.BatchNormal(out.mean, out.cov_sqrtm_lower)

    def scale_covariance(self, scale_sqrtm):
        out = jax.vmap(_scalar.Conditional.scale_covariance)(
            self.conditional, scale_sqrtm
        )
        noise = _vars.BatchNormal(out.noise.mean, out.noise.cov_sqrtm_lower)
        return BatchConditional(transition=out.transition, noise=noise)

    def merge_with_incoming_conditional(self, incoming, /):
        fn = jax.vmap(_scalar.Conditional.merge_with_incoming_conditional)
        merged = fn(self.conditional, incoming.conditional)
        noise = _vars.BatchNormal(merged.noise.mean, merged.noise.cov_sqrtm_lower)
        return BatchConditional(transition=merged.transition, noise=noise)

    def marginalise(self, rv, /):
        marginalised = jax.vmap(_scalar.Conditional.marginalise)(self.conditional, rv)
        return _vars.BatchNormal(marginalised.mean, marginalised.cov_sqrtm_lower)


@jax.tree_util.register_pytree_node_class
class BatchIBM(_collections.AbstractExtrapolation[_vars.BatchNormal, _IBMCacheType]):
    def __init__(self, a, q_sqrtm_lower):
        self.ibm = _scalar.IBM(a, q_sqrtm_lower)

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(a={self.a}, q_sqrtm_lower={self.q_sqrtm_lower})"

    @property
    def num_derivatives(self):
        return self.ibm.a.shape[1] - 1

    @property
    def ode_shape(self):
        return (
            self.ibm.a.shape[0],
        )  # todo: this does not scale to matrix-valued problems

    def tree_flatten(self):
        children = (self.ibm,)
        return children, ()

    @classmethod
    def tree_unflatten(cls, _aux, children):
        (ibm,) = children
        return cls(a=ibm.a, q_sqrtm_lower=ibm.q_sqrtm_lower)

    @classmethod
    def from_params(cls, ode_shape, num_derivatives):
        """Create a strategy from hyperparameters."""
        assert len(ode_shape) == 1
        (n,) = ode_shape
        a, q_sqrtm = _ibm_util.system_matrices_1d(num_derivatives=num_derivatives)
        a_stack, q_sqrtm_stack = _tree_stack_duplicates((a, q_sqrtm), n=n)
        return cls(a=a_stack, q_sqrtm_lower=q_sqrtm_stack)

    def begin_extrapolation(self, m0, /, dt):
        fn = jax.vmap(_scalar.IBM.begin_extrapolation, in_axes=(0, 0, None))
        extra, cache = fn(self.ibm, m0, dt)
        extra_batch = _vars.BatchNormal(extra.mean, extra.cov_sqrtm_lower)
        return extra_batch, cache

    def complete_extrapolation(self, linearisation_pt, cache, l0, output_scale_sqrtm):
        fn = jax.vmap(_scalar.IBM.complete_extrapolation)
        ext = fn(self.ibm, linearisation_pt, cache, l0, output_scale_sqrtm)
        return _vars.BatchNormal(ext.mean, ext.cov_sqrtm_lower)

    def init_conditional(self, rv_proto):
        cond = jax.vmap(_scalar.IBM.init_conditional)(self.ibm, rv_proto)
        noise = _vars.BatchNormal(cond.noise.mean, cond.noise.cov_sqrtm_lower)
        return BatchConditional(cond.transition, noise=noise)

    def init_corrected(self, taylor_coefficients):
        cor = jax.vmap(_scalar.IBM.init_corrected)(self.ibm, taylor_coefficients)
        return _vars.BatchNormal(cor.mean, cor.cov_sqrtm_lower)

    # todo: move to correction?
    def init_error_estimate(self):
        return jax.vmap(_scalar.IBM.init_error_estimate)(self.ibm)

    # todo: move to correction?
    def init_output_scale_sqrtm(self):
        return jax.vmap(_scalar.IBM.init_output_scale_sqrtm)(self.ibm)

    def revert_markov_kernel(self, linearisation_pt, l0, output_scale_sqrtm, cache):
        fn = jax.vmap(_scalar.IBM.revert_markov_kernel)
        ext, bw_model = fn(self.ibm, linearisation_pt, cache, l0, output_scale_sqrtm)

        ext_batched = _vars.BatchNormal(ext.mean, ext.cov_sqrtm_lower)
        bw_noise = _vars.BatchNormal(
            bw_model.noise.mean, bw_model.noise.cov_sqrtm_lower
        )
        bw_model_batched = BatchConditional(bw_model.transition, bw_noise)
        return ext_batched, bw_model_batched


def _tree_stack_duplicates(tree, n):
    return jax.tree_util.tree_map(lambda s: jnp.vstack([s[None, ...]] * n), tree)
