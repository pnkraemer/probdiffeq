"""Test equivalences between cubature rules."""

from probdiffeq import probdiffeq
from probdiffeq.backend import testing


def test_third_order_spherical_vs_unscented_transform_scalar_input():
    """Assert that UT with r=0 equals the third-order spherical rule."""
    tos = probdiffeq.cubature_third_order_spherical(input_shape=())
    ut = probdiffeq.cubature_unscented_transform(input_shape=(), r=0.0)
    tos_points, tos_weights = tos.points, tos.weights_sqrtm
    ut_points, ut_weights = ut.points, ut.weights_sqrtm
    for x, y in [(ut_weights, tos_weights), (ut_points, tos_points)]:
        assert testing.allclose(x[:1], y[:1])
        assert testing.allclose(x[1], 0.0)
        assert testing.allclose(x[2:], y[1:])


@testing.parametrize("n", [4])
def test_third_order_spherical_vs_unscented_transform(n):
    """Assert that UT with r=0 equals the third-order spherical rule."""
    tos = probdiffeq.cubature_third_order_spherical(input_shape=(n,))
    ut = probdiffeq.cubature_unscented_transform(input_shape=(n,), r=0.0)
    tos_points, tos_weights = tos.points, tos.weights_sqrtm
    ut_points, ut_weights = ut.points, ut.weights_sqrtm
    for x, y in [(ut_weights, tos_weights), (ut_points, tos_points)]:
        assert testing.allclose(x[:n], y[:n])
        assert testing.allclose(x[n], 0.0)
        assert testing.allclose(x[n + 1 :], y[n:])


# todo: test for gauss-hermite? Do we need one? (we wrap scipy's rules anyway...)
