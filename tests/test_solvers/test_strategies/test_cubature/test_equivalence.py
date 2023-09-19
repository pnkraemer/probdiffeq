"""Test equivalences between cubature rules."""
import jax.numpy as jnp

from probdiffeq.solvers.strategies.components import cubature


def test_third_order_spherical_vs_unscented_transform_scalar_input():
    """Assert that UT with r=0 equals the third-order spherical rule."""
    tos = cubature.third_order_spherical(input_shape=())
    ut = cubature.unscented_transform(input_shape=(), r=0.0)
    tos_points, tos_weights = tos.points, tos.weights_sqrtm
    ut_points, ut_weights = ut.points, ut.weights_sqrtm
    for x, y in [(ut_weights, tos_weights), (ut_points, tos_points)]:
        assert jnp.allclose(x[:1], y[:1])
        assert jnp.allclose(x[1], 0.0)
        assert jnp.allclose(x[2:], y[1:])


def test_third_order_spherical_vs_unscented_transform(n=4):
    """Assert that UT with r=0 equals the third-order spherical rule."""
    tos = cubature.third_order_spherical(input_shape=(n,))
    ut = cubature.unscented_transform(input_shape=(n,), r=0.0)
    tos_points, tos_weights = tos.points, tos.weights_sqrtm
    ut_points, ut_weights = ut.points, ut.weights_sqrtm
    for x, y in [(ut_weights, tos_weights), (ut_points, tos_points)]:
        print(x, y)
        assert jnp.allclose(x[:n], y[:n])
        assert jnp.allclose(x[n], 0.0)
        assert jnp.allclose(x[n + 1 :], y[n:])


# todo: test for gauss-hermite? Do we need one? (we wrap scipy's rules anyway...)
