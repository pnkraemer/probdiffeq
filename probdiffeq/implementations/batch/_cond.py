import jax

from probdiffeq.implementations import _collections, _scalar
from probdiffeq.implementations.batch import _ssv


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
        return _ssv.BatchNormal(out.mean, out.cov_sqrtm_lower)

    def scale_covariance(self, scale_sqrtm):
        out = jax.vmap(_scalar.Conditional.scale_covariance)(
            self.conditional, scale_sqrtm
        )
        noise = _ssv.BatchNormal(out.noise.mean, out.noise.cov_sqrtm_lower)
        return BatchConditional(transition=out.transition, noise=noise)

    def merge_with_incoming_conditional(self, incoming, /):
        fn = jax.vmap(_scalar.Conditional.merge_with_incoming_conditional)
        merged = fn(self.conditional, incoming.conditional)
        noise = _ssv.BatchNormal(merged.noise.mean, merged.noise.cov_sqrtm_lower)
        return BatchConditional(transition=merged.transition, noise=noise)

    def marginalise(self, rv, /):
        marginalised = jax.vmap(_scalar.Conditional.marginalise)(self.conditional, rv)
        return _ssv.BatchNormal(marginalised.mean, marginalised.cov_sqrtm_lower)
