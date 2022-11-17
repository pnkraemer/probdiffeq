import jax

from probdiffeq.implementations import _collections, _sqrtm
from probdiffeq.implementations.vect import _ssv


@jax.tree_util.register_pytree_node_class
class VectConditional(_collections.AbstractConditional):
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

    def __call__(self, x, /):
        m = self.transition @ x + self.noise.mean
        shape = self.noise.target_shape
        return _ssv.VectNormal(m, self.noise.cov_sqrtm_lower, target_shape=shape)

    def scale_covariance(self, scale_sqrtm):
        noise = self.noise.scale_covariance(scale_sqrtm=scale_sqrtm)
        return VectConditional(transition=self.transition, noise=noise)

    def merge_with_incoming_conditional(self, incoming, /):
        A = self.transition
        (b, B_sqrtm) = self.noise.mean, self.noise.cov_sqrtm_lower

        C = incoming.transition
        (d, D_sqrtm) = (incoming.noise.mean, incoming.noise.cov_sqrtm_lower)

        g = A @ C
        xi = A @ d + b
        Xi = _sqrtm.sum_of_sqrtm_factors(R1=(A @ D_sqrtm).T, R2=B_sqrtm.T).T

        shape = self.noise.target_shape
        noise = _ssv.VectNormal(mean=xi, cov_sqrtm_lower=Xi, target_shape=shape)
        return VectConditional(g, noise=noise)

    def marginalise(self, rv, /):
        # Pull into preconditioned space
        m0_p = rv.mean
        l0_p = rv.cov_sqrtm_lower

        # Apply transition
        m_new_p = self.transition @ m0_p + self.noise.mean
        l_new_p = _sqrtm.sum_of_sqrtm_factors(
            R1=(self.transition @ l0_p).T, R2=self.noise.cov_sqrtm_lower.T
        ).T

        # Push back into non-preconditioned space
        m_new = m_new_p
        l_new = l_new_p

        return _ssv.VectNormal(m_new, l_new, target_shape=rv.target_shape)
