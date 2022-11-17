from typing import Generic, TypeVar

import jax
import jax.numpy as jnp

from probdiffeq.implementations import _collections, _scalar

SSV = TypeVar("SSV")


@jax.tree_util.register_pytree_node_class
class BatchNormal(_collections.StateSpaceVariable, Generic[SSV]):
    # Shapes: (d, n), (d, n, n). QOI: n=0

    def __init__(self, mean, cov_sqrtm_lower):
        self._normal = _scalar.Normal(mean, cov_sqrtm_lower)

    def __repr__(self):
        name = f"{self.__class__.__name__}"
        args = f"mean={self.mean}, cov_sqrtm_lower={self.cov_sqrtm_lower}"
        return f"{name}({args})"

    @property
    def mean(self):
        return self._normal.mean

    @property
    def cov_sqrtm_lower(self):
        return self._normal.cov_sqrtm_lower

    def tree_flatten(self):
        children = (self._normal,)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        (normal,) = children
        return cls(normal.mean, normal.cov_sqrtm_lower)

    def to_normal(self):
        return _scalar.Normal(self.mean, self.cov_sqrtm_lower)

    @classmethod
    def from_normal(cls, normal):
        return cls(normal.mean, normal.cov_sqrtm_lower)

    @property
    def sample_shape(self):
        return self._normal.sample_shape  # mean is (d, n)

    def logpdf(self, u, /):
        batch_logpdf = jax.vmap(_scalar.Normal.logpdf)(self._normal, u)
        return jnp.sum(batch_logpdf)

    def norm_of_whitened_residual_sqrtm(self) -> jax.Array:
        fn = jax.vmap(_scalar.Normal.norm_of_whitened_residual_sqrtm)
        return fn(self._normal)

    def condition_on_qoi_observation(self, u, /, observation_std):
        fn = jax.vmap(_scalar.Normal.condition_on_qoi_observation, in_axes=(0, 0, None))
        obs, (cor, gain) = fn(self._normal, u, observation_std)

        obs_batch = BatchScalarNormal(obs.mean, obs.cov_sqrtm_lower)
        cor_batch = BatchNormal(cor.mean, cor.cov_sqrtm_lower)
        return obs_batch, (cor_batch, gain)

    def extract_qoi(self):
        return jax.vmap(_scalar.Normal.extract_qoi)(self._normal)

    def extract_qoi_from_sample(self, u, /):
        fn = jax.vmap(_scalar.Normal.extract_qoi_from_sample)
        return fn(self._normal, u)

    def Ax_plus_y(self, A, x, y):
        fn = jax.vmap(_scalar.Normal.Ax_plus_y)
        return fn(self._normal, A, x, y)

    def scale_covariance(self, scale_sqrtm):
        fn = jax.vmap(_scalar.Normal.scale_covariance)
        scaled = fn(self._normal, scale_sqrtm)
        return BatchNormal(scaled.mean, scaled.cov_sqrtm_lower)

    def transform_unit_sample(self, x, /):
        fn = jax.vmap(_scalar.Normal.transform_unit_sample)
        return fn(self._normal, x)


@jax.tree_util.register_pytree_node_class
class BatchScalarNormal(_collections.StateSpaceVariable, Generic[SSV]):
    def __init__(self, mean, cov_sqrtm_lower):
        self._normal = _scalar.ScalarNormal(mean, cov_sqrtm_lower)

    @property
    def mean(self):
        return self._normal.mean

    @property
    def cov_sqrtm_lower(self):
        return self._normal.cov_sqrtm_lower

    def tree_flatten(self):
        children = (self._normal,)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        (normal,) = children
        return cls(normal.mean, normal.cov_sqrtm_lower)

    @classmethod
    def from_scalar_normal(cls, normal):
        return cls(normal.mean, normal.cov_sqrtm_lower)

    @property
    def sample_shape(self):
        return self._normal.sample_shape  # mean is (d, n)

    def logpdf(self, u, /):
        batch_logpdf = jax.vmap(_scalar.ScalarNormal.logpdf)(self._normal, u)
        return jnp.sum(batch_logpdf)

    def norm_of_whitened_residual_sqrtm(self) -> jax.Array:
        fn = jax.vmap(_scalar.ScalarNormal.norm_of_whitened_residual_sqrtm)
        return fn(self._normal)

    def condition_on_qoi_observation(self, u, /, observation_std):
        raise NotImplementedError

    def extract_qoi(self):
        raise NotImplementedError

    def extract_qoi_from_sample(self, u, /):
        raise NotImplementedError

    def Ax_plus_y(self, A, x, y):
        raise NotImplementedError

    def scale_covariance(self, scale_sqrtm):
        raise NotImplementedError

    def transform_unit_sample(self, x, /):
        raise NotImplementedError
