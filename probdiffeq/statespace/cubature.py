"""Cubature rules."""

import jax
import jax.numpy as jnp
import scipy.special  # type: ignore

from probdiffeq.backend import containers


class PositiveCubatureRule(containers.NamedTuple):
    """Cubature rule with positive weights."""

    points: jax.Array
    weights_sqrtm: jax.Array


def third_order_spherical(input_shape) -> PositiveCubatureRule:
    """Third-order spherical cubature integration."""
    assert len(input_shape) <= 1
    if len(input_shape) == 1:
        (d,) = input_shape
        points_mat, weights_sqrtm = _third_order_spherical_params(d=d)
        return PositiveCubatureRule(points=points_mat, weights_sqrtm=weights_sqrtm)

    # If input_shape == (), compute weights via input_shape=(1,)
    # and 'squeeze' the points.
    points_mat, weights_sqrtm = _third_order_spherical_params(d=1)
    (S, _) = points_mat.shape
    points = jnp.reshape(points_mat, (S,))
    return PositiveCubatureRule(points=points, weights_sqrtm=weights_sqrtm)


def _third_order_spherical_params(*, d):
    eye_d = jnp.eye(d) * jnp.sqrt(d)
    pts = jnp.concatenate((eye_d, -1 * eye_d))
    weights_sqrtm = jnp.ones((2 * d,)) / jnp.sqrt(2.0 * d)
    return pts, weights_sqrtm


def unscented_transform(input_shape, r=1.0) -> PositiveCubatureRule:
    """Unscented transform."""
    assert len(input_shape) <= 1
    if len(input_shape) == 1:
        (d,) = input_shape
        points_mat, weights_sqrtm = _unscented_transform_params(d=d, r=r)
        return PositiveCubatureRule(points=points_mat, weights_sqrtm=weights_sqrtm)

    # If input_shape == (), compute weights via input_shape=(1,)
    # and 'squeeze' the points.
    points_mat, weights_sqrtm = _unscented_transform_params(d=1, r=r)
    (S, _) = points_mat.shape
    points = jnp.reshape(points_mat, (S,))
    return PositiveCubatureRule(points=points, weights_sqrtm=weights_sqrtm)


def _unscented_transform_params(d, *, r):
    eye_d = jnp.eye(d) * jnp.sqrt(d + r)
    zeros = jnp.zeros((1, d))
    pts = jnp.concatenate((eye_d, zeros, -1 * eye_d))
    _scale = d + r
    weights_sqrtm1 = jnp.ones((d,)) / jnp.sqrt(2.0 * _scale)
    weights_sqrtm2 = jnp.sqrt(r / _scale)
    weights_sqrtm = jnp.hstack((weights_sqrtm1, weights_sqrtm2, weights_sqrtm1))
    return pts, weights_sqrtm


def gauss_hermite(input_shape, degree=5) -> PositiveCubatureRule:
    """(Statistician's) Gauss-Hermite cubature.

    The number of cubature points is `prod(input_shape)**degree`.
    """
    assert len(input_shape) == 1
    (dim,) = input_shape

    # Roots of the probabilist/statistician's Hermite polynomials (in Numpy...)
    _roots = scipy.special.roots_hermitenorm(n=degree, mu=True)
    pts, weights, sum_of_weights = _roots
    weights = weights / sum_of_weights

    # Transform into jax arrays and take square root of weights
    pts = jnp.asarray(pts)
    weights_sqrtm = jnp.sqrt(jnp.asarray(weights))

    # Build a tensor grid and return class
    tensor_pts = _tensor_points(pts, d=dim)
    tensor_weights_sqrtm = _tensor_weights(weights_sqrtm, d=dim)
    return PositiveCubatureRule(points=tensor_pts, weights_sqrtm=tensor_weights_sqrtm)


# how does this generalise to an input_shape instead of an input_dimension?
# via tree_map(lambda s: _tensor_points(x, s), input_shape)?


def _tensor_weights(*args, **kwargs):
    mesh = _tensor_points(*args, **kwargs)
    return jnp.prod(mesh, axis=1)


def _tensor_points(x, /, *, d):
    x_mesh = jnp.meshgrid(*([x] * d))
    y_mesh = jax.tree_util.tree_map(lambda s: jnp.reshape(s, (-1,)), x_mesh)
    return jnp.stack(y_mesh).T
