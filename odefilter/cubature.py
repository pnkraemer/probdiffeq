"""Cubature rules."""

import jax
import jax.numpy as jnp
import scipy.special  # type: ignore

# todo: input_dimension -> input_shape. But how does the UT work in this case?


@jax.tree_util.register_pytree_node_class
class _PositiveCubatureRule:
    """Cubature rule with positive weights."""

    def __init__(self, *, points, weights_sqrtm):
        self.points = points
        self.weights_sqrtm = weights_sqrtm

    def tree_flatten(self):
        children = self.points, self.weights_sqrtm
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        pts, weights_sqrtm = children
        return cls(points=pts, weights_sqrtm=weights_sqrtm)


@jax.tree_util.register_pytree_node_class
class SphericalCubatureIntegration(_PositiveCubatureRule):
    """Spherical cubature integration."""

    @classmethod
    def from_params(cls, *, input_dimension):
        """Construct an SCI rule from the dimension of a random variable.

        The number of cubature points is _higher_ than ``input_dimension``.
        """
        _d = input_dimension  # alias for readability
        eye_d = jnp.eye(_d) * jnp.sqrt(_d)
        pts = jnp.vstack((eye_d, -1 * eye_d))

        weights_sqrtm = jnp.ones((2 * _d,)) / jnp.sqrt(2.0 * _d)
        return cls(points=pts, weights_sqrtm=weights_sqrtm)


@jax.tree_util.register_pytree_node_class
class UnscentedTransform(_PositiveCubatureRule):
    """Unscented transform."""

    # todo: more parameters...
    @classmethod
    def from_params(cls, *, input_dimension, r=1.0):
        """Construct an unscented transform from parameters.

        The number of cubature points is _higher_ than ``input_dimension``.
        """
        _d = input_dimension  # alias for readability
        eye_d = jnp.eye(_d) * jnp.sqrt(_d + r)
        zeros = jnp.zeros((1, _d))
        pts = jnp.vstack((eye_d, zeros, -1 * eye_d))

        _scale = _d + r
        weights_sqrtm1 = jnp.ones((_d,)) / jnp.sqrt(2.0 * _scale)
        weights_sqrtm2 = jnp.sqrt(r / _scale)
        weights_sqrtm = jnp.hstack((weights_sqrtm1, weights_sqrtm2, weights_sqrtm1))
        return cls(points=pts, weights_sqrtm=weights_sqrtm)


@jax.tree_util.register_pytree_node_class
class GaussHermite(_PositiveCubatureRule):
    """Gauss-Hermite cubature."""

    @classmethod
    def from_params(cls, *, input_dimension, degree=5):
        """Construct a Gauss-Hermite cubature rule.

        The number of cubature points is input_dimension**degree.
        """
        # Roots of the probabilist/statistician's Hermite polynomials (in Numpy...)
        _roots = scipy.special.roots_hermitenorm(n=degree, mu=True)
        pts, weights, sum_of_weights = _roots
        weights = weights / sum_of_weights

        # Transform into jax arrays and take square root of weights
        pts = jnp.asarray(pts)
        weights_sqrtm = jnp.sqrt(jnp.asarray(weights))

        # Build a tensor grid and return class
        tensor_pts = _tensor_points(pts, d=input_dimension)
        tensor_weights_sqrtm = _tensor_weights(weights_sqrtm, d=input_dimension)
        return cls(points=tensor_pts, weights_sqrtm=tensor_weights_sqrtm)


# how does this generalise to an input_shape instead of an input_dimension?
# via tree_map(lambda s: _tensor_points(x, s), input_shape)?


def _tensor_weights(*args, **kwargs):
    mesh = _tensor_points(*args, **kwargs)
    return jnp.prod(mesh, axis=1)


def _tensor_points(x, /, *, d):
    x_mesh = jnp.meshgrid(*([x] * d))
    y_mesh = jax.tree_util.tree_map(lambda s: jnp.reshape(s, (-1,)), x_mesh)
    return jnp.vstack(y_mesh).T
