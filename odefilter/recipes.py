"""Recipes for ODE filters.

Learning about the inner workings of an ODE filter is a little too much?
We hear ya -- tt can indeed get quite complicated.
Therefore, here we provide some recipes that create our favourite,
time-tested ODE filter versions.
We still recommend to build an ODE filter yourself,
but until you do so, use one of ours.

"""
from odefilter import information
from odefilter.implementations import dense, isotropic
from odefilter.strategies import filters, fixedpoint, smoothers


def dynamic_isotropic_ekf0(*, num_derivatives=4, ode_order=1):
    """Construct the equivalent of an explicit solver with an isotropic covariance \
    structure, dynamic calibration, and optimised for terminal-value simulation.

    Suitable for high-dimensional, non-stiff problems.
    """
    _assert_num_derivatives_sufficiently_large(
        num_derivatives=num_derivatives, ode_order=ode_order
    )
    implementation = isotropic.IsotropicImplementation.from_num_derivatives(
        num_derivatives=num_derivatives
    )
    solver = filters.DynamicFilter(implementation=implementation)
    information_op = information.isotropic_ek0(ode_order=ode_order)
    return solver, information_op


def dynamic_isotropic_eks0(*, num_derivatives=4, ode_order=1):
    """Construct the equivalent of an explicit solver with an isotropic covariance \
    structure and dynamic calibration.

    Suitable for high-dimensional, non-stiff problems.
    """
    _assert_num_derivatives_sufficiently_large(
        num_derivatives=num_derivatives, ode_order=ode_order
    )
    implementation = isotropic.IsotropicImplementation.from_num_derivatives(
        num_derivatives=num_derivatives
    )
    solver = smoothers.DynamicSmoother(implementation=implementation)
    information_op = information.isotropic_ek0(ode_order=ode_order)
    return solver, information_op


def dynamic_isotropic_fixedpoint_eks0(*, num_derivatives=4, ode_order=1):
    """Construct the equivalent of an explicit solver with an isotropic covariance \
    structure and dynamic calibration.

    Suitable for high-dimensional, non-stiff problems.
    """
    _assert_num_derivatives_sufficiently_large(
        num_derivatives=num_derivatives, ode_order=ode_order
    )
    implementation = isotropic.IsotropicImplementation.from_num_derivatives(
        num_derivatives=num_derivatives
    )
    solver = fixedpoint.DynamicFixedPointSmoother(implementation=implementation)
    information_op = information.isotropic_ek0(ode_order=ode_order)
    return solver, information_op


def dynamic_ekf1(*, ode_dimension, num_derivatives=4, ode_order=1):
    """Construct the equivalent of a semi-implicit solver with dynamic calibration.

    Suitable for low-dimensional, stiff problems.
    """
    _assert_num_derivatives_sufficiently_large(
        num_derivatives=num_derivatives, ode_order=ode_order
    )
    implementation = dense.DenseImplementation.from_num_derivatives(
        num_derivatives=num_derivatives, ode_dimension=ode_dimension
    )
    solver = filters.DynamicFilter(implementation=implementation)
    information_op = information.ek1(ode_dimension=ode_dimension, ode_order=ode_order)
    return solver, information_op


def ekf1(*, ode_dimension, num_derivatives=4, ode_order=1):
    """Construct the equivalent of a semi-implicit solver.

    Suitable for low-dimensional, stiff problems.
    """
    _assert_num_derivatives_sufficiently_large(
        num_derivatives=num_derivatives, ode_order=ode_order
    )
    implementation = dense.DenseImplementation.from_num_derivatives(
        num_derivatives=num_derivatives, ode_dimension=ode_dimension
    )
    solver = filters.Filter(implementation=implementation)
    information_op = information.ek1(ode_dimension=ode_dimension, ode_order=ode_order)
    return solver, information_op


def isotropic_ekf0(*, num_derivatives=4, ode_order=1):
    """Construct the equivalent of an explicit solver with an isotropic covariance \
    structure, and optimised for terminal-value simulation.

    Suitable for high-dimensional, non-stiff problems.
    """
    _assert_num_derivatives_sufficiently_large(
        num_derivatives=num_derivatives, ode_order=ode_order
    )
    implementation = isotropic.IsotropicImplementation.from_num_derivatives(
        num_derivatives=num_derivatives
    )
    solver = filters.Filter(implementation=implementation)
    information_op = information.isotropic_ek0(ode_order=ode_order)
    return solver, information_op


def _assert_num_derivatives_sufficiently_large(*, num_derivatives, ode_order):
    assert num_derivatives >= ode_order
