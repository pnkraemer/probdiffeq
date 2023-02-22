"""Cubature rules."""

import jax
import jax.numpy as jnp
import scipy.special  # type: ignore

# todo: clean up the constructors
#  (there is a lot of duplication, and the *_batch logic is not really obvious)


@jax.tree_util.register_pytree_node_class
class _PositiveCubatureRule:
    """Cubature rule with positive weights."""

    def __init__(self, *, points, weights_sqrtm):
        self.points = points
        self.weights_sqrtm = weights_sqrtm

    def __repr__(self):
        name = self.__class__.__name__
        args = f"points={self.points}, weights_sqrtm={self.weights_sqrtm}"
        return f"{name}({args})"

    def tree_flatten(self):
        children = self.points, self.weights_sqrtm
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        pts, weights_sqrtm = children
        return cls(points=pts, weights_sqrtm=weights_sqrtm)

    @classmethod
    def from_params_blockdiag(cls, input_shape, **kwargs):
        # Todo: is this what _we want_?
        #  It is what we had so far, but how does the complexity of this mess
        #  scale with the dimensionality of the problem?
        #  It would be more efficient if S would not depend on the dimension anymore.
        #  Currently it does. If we simply stacked 'd' 1-dimensional rules
        #  on top of each other, the complexity reduces
        #  (but the solver seems to suffer a lot...)

        # Alright, so what do we do here?
        # Make a _PositiveCubatureRule(points.shape=(S, d), weights.shape=(S,))
        # pylint: disable=no-member
        instance = cls.from_params(input_shape=input_shape, **kwargs)

        d, *_ = input_shape
        points = instance.points.T  # (d, S)
        weights_sqrtm = jnp.stack(d * [instance.weights_sqrtm])  # (d, S)
        return cls(points=points, weights_sqrtm=weights_sqrtm)


def _tree_stack_duplicates(tree, n):
    return jax.tree_util.tree_map(lambda s: jnp.vstack([s[None, ...]] * n), tree)


def _tree_shape(tree):
    return jax.tree_util.tree_map(jnp.shape, tree)


@jax.tree_util.register_pytree_node_class
class ThirdOrderSpherical(_PositiveCubatureRule):
    """Third-order spherical cubature integration."""

    @classmethod
    def from_params(cls, input_shape):
        """Construct an SCI rule from the shape of the input of the integrand."""
        assert len(input_shape) <= 1
        if len(input_shape) == 1:
            (d,) = input_shape
            points_mat, weights_sqrtm = _sci_pts_and_weights_sqrtm(d=d)
            return cls(points=points_mat, weights_sqrtm=weights_sqrtm)

        # If input_shape == (), compute weights via input_shape=(1,)
        # and 'squeeze' the points.
        points_mat, weights_sqrtm = _sci_pts_and_weights_sqrtm(d=1)
        (S, _) = points_mat.shape
        points = jnp.reshape(points_mat, (S,))
        return cls(points=points, weights_sqrtm=weights_sqrtm)


def _sci_pts_and_weights_sqrtm(*, d):
    eye_d = jnp.eye(d) * jnp.sqrt(d)
    pts = jnp.vstack((eye_d, -1 * eye_d))
    weights_sqrtm = jnp.ones((2 * d,)) / jnp.sqrt(2.0 * d)
    return pts, weights_sqrtm


@jax.tree_util.register_pytree_node_class
class UnscentedTransform(_PositiveCubatureRule):
    """Unscented transform."""

    # todo: more parameters...
    @classmethod
    def from_params(cls, *, input_shape, r=1.0):
        """Construct an unscented transform from parameters."""
        assert len(input_shape) <= 1
        if len(input_shape) == 1:
            (d,) = input_shape
            points_mat, weights_sqrtm = _ut_points_and_weights_sqrtm(d=d, r=r)
            return cls(points=points_mat, weights_sqrtm=weights_sqrtm)

        # If input_shape == (), compute weights via input_shape=(1,)
        # and 'squeeze' the points.
        points_mat, weights_sqrtm = _ut_points_and_weights_sqrtm(d=1, r=r)
        (S, _) = points_mat.shape
        points = jnp.reshape(points_mat, (S,))
        return cls(points=points, weights_sqrtm=weights_sqrtm)


def _ut_points_and_weights_sqrtm(d, *, r):
    eye_d = jnp.eye(d) * jnp.sqrt(d + r)
    zeros = jnp.zeros((1, d))
    pts = jnp.vstack((eye_d, zeros, -1 * eye_d))
    _scale = d + r
    weights_sqrtm1 = jnp.ones((d,)) / jnp.sqrt(2.0 * _scale)
    weights_sqrtm2 = jnp.sqrt(r / _scale)
    weights_sqrtm = jnp.hstack((weights_sqrtm1, weights_sqrtm2, weights_sqrtm1))
    return pts, weights_sqrtm


@jax.tree_util.register_pytree_node_class
class GaussHermite(_PositiveCubatureRule):
    """(Statistician's) Gauss-Hermite cubature."""

    @classmethod
    def from_params(cls, *, input_shape, degree=5):
        """Construct a Gauss-Hermite cubature rule.

        The number of cubature points is prod(input_shape)**degree.
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
