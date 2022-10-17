"""Recipes for ODE filters.

Learning about the inner workings of an ODE filter is a little too much?
We hear ya -- tt can indeed get quite complicated.
Therefore, here we provide some recipes that create our favourite,
time-tested ODE filter versions.
We still recommend to build an ODE filter yourself,
but until you do so, use one of ours.

"""
from odefilter import information, solvers
from odefilter.implementations import dense, isotropic


def dynamic_isotropic_ekf0(num_derivatives):
    """Construct the equivalent of an explicit solver with an isotropic covariance \
    structure, dynamic calibration, and optimised for terminal-value simulation.

    Suitable for high-dimensional, non-stiff problems.
    """
    implementation = isotropic.IsotropicImplementation.from_num_derivatives(
        num_derivatives=num_derivatives
    )
    solver = solvers.DynamicFilter(implementation=implementation)
    information_op = information.isotropic_ek0(ode_order=1)
    return solver, information_op


def dynamic_isotropic_eks0(num_derivatives):
    """Construct the equivalent of an explicit solver with an isotropic covariance \
    structure and dynamic calibration.

    Suitable for high-dimensional, non-stiff problems.
    """
    implementation = isotropic.IsotropicImplementation.from_num_derivatives(
        num_derivatives=num_derivatives
    )
    solver = solvers.DynamicSmoother(implementation=implementation)
    information_op = information.isotropic_ek0(ode_order=1)
    return solver, information_op


def dynamic_isotropic_fixpt_eks0(num_derivatives):
    """Construct the equivalent of an explicit solver with an isotropic covariance \
    structure and dynamic calibration.

    Suitable for high-dimensional, non-stiff problems.
    """
    implementation = isotropic.IsotropicImplementation.from_num_derivatives(
        num_derivatives=num_derivatives
    )
    solver = solvers.DynamicFixedPointSmoother(implementation=implementation)
    information_op = information.isotropic_ek0(ode_order=1)
    return solver, information_op


def dynamic_ekf1(num_derivatives, ode_dimension):
    """Construct the equivalent of a semi-implicit solver with dynamic calibration.

    Suitable for low-dimensional, stiff problems.
    """
    implementation = dense.DenseImplementation.from_num_derivatives(
        num_derivatives=num_derivatives, ode_dimension=ode_dimension
    )
    solver = solvers.DynamicFilter(implementation=implementation)
    information_op = information.ek1(ode_dimension=ode_dimension, ode_order=1)
    return solver, information_op
