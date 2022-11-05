"""Cubature rules."""
import dataclasses

import jax
import jax.numpy as jnp
import scipy.special  # type: ignore


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class _PositiveCubatureRule:
    """Cubature rule with positive weights."""

    points: jax.Array
    weights_sqrtm: jax.Array

    def __repr__(self):
        return f"{self.__class__.__name__}(k={self.points.shape[0]})"

    def tree_flatten(self):
        children = self.points, self.weights_sqrtm
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        pts, weights_sqrtm = children
        return cls(points=pts, weights_sqrtm=weights_sqrtm)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class SphericalCubatureIntegration(_PositiveCubatureRule):
    """Spherical cubature integration."""

    @classmethod
    def from_params(cls, *, ode_dimension):
        """Construct an SCI rule from the ode_dimensionension of a random variable.

        The number of cubature points is _higher_ than ``ode_dimension``.
        """
        eye_d = jnp.eye(ode_dimension) * jnp.sqrt(ode_dimension)
        pts = jnp.vstack((eye_d, -1 * eye_d))
        weights_sqrtm = jnp.ones((2 * ode_dimension,)) / jnp.sqrt(2.0 * ode_dimension)
        return cls(points=pts, weights_sqrtm=weights_sqrtm)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class UnscentedTransform(_PositiveCubatureRule):
    """Unscented transform."""

    # todo: more parameters...
    @classmethod
    def from_params(cls, *, ode_dimension, r=1.0):
        """Construct an unscented transform from parameters.

        The number of cubature points is _higher_ than ``ode_dimension``.
        """
        eye_d = jnp.eye(ode_dimension) * jnp.sqrt(ode_dimension + r)
        zeros = jnp.zeros((1, ode_dimension))
        pts = jnp.vstack((eye_d, zeros, -1 * eye_d))

        weights_sqrtm1 = jnp.ones((ode_dimension,)) / jnp.sqrt(
            2.0 * (ode_dimension + r)
        )
        weights_sqrtm2 = jnp.sqrt(r / (ode_dimension + r))
        weights_sqrtm = jnp.hstack((weights_sqrtm1, weights_sqrtm2, weights_sqrtm1))
        return cls(points=pts, weights_sqrtm=weights_sqrtm)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class GaussHermite(_PositiveCubatureRule):
    """Gauss-Hermite cubature."""

    @classmethod
    def from_params(cls, *, ode_dimension, degree=5):
        """Construct a Gauss-Hermite cubature rule.

        The number of cubature points is ode_dimension**degree.
        """
        # Roots of the probabilist/statistician's Hermite polynomials (in Numpy...)
        pts, weights, sum_of_weights = scipy.special.roots_hermitenorm(
            n=degree, mu=True
        )
        weights = weights / sum_of_weights

        # Transform into jax arrays and take square root of weights
        pts = jnp.asarray(pts)
        weights_sqrtm = jnp.sqrt(jnp.asarray(weights))

        # Build a tensor grid and return class
        tensor_pts = _tensor_points(pts, ode_dimension=ode_dimension)
        tensor_weights_sqrtm = _tensor_weights(
            weights_sqrtm, ode_dimension=ode_dimension
        )
        return cls(points=tensor_pts, weights_sqrtm=tensor_weights_sqrtm)


def _tensor_weights(*args, **kwargs):
    mesh = _tensor_points(*args, **kwargs)
    return jnp.prod(mesh, axis=1)


def _tensor_points(x, /, *, ode_dimension):
    x_mesh = jnp.meshgrid(*([x] * ode_dimension))
    y_mesh = jax.tree_util.tree_map(lambda s: jnp.reshape(s, (-1,)), x_mesh)
    return jnp.vstack(y_mesh).T
