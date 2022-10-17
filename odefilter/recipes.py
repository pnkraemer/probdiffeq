"""Recipes for ODE filters.

Learning about the inner workings of an ODE filter is a little too much?
We hear ya -- tt can indeed get quite complicated.
Therefore, here we provide some recipes that create our favourite,
time-tested ODE filter versions.
We still recommend to build an ODE filter yourself,
but until you do so, use one of ours.

"""
from odefilter import _adaptive, controls, information, strategies
from odefilter.implementations import dense, isotropic

_ATOL_DEFAULTS = 1e-6
_RTOL_DEFAULTS = 1e-3


def dynamic_isotropic_ekf0(num_derivatives, atol=_ATOL_DEFAULTS, rtol=_RTOL_DEFAULTS):
    """Construct the equivalent of an explicit solver with an isotropic covariance \
    structure, dynamic calibration, and optimised for terminal-value simulation.

    Suitable for high-dimensional, non-stiff problems.
    """
    implementation = isotropic.IsotropicImplementation.from_num_derivatives(
        num_derivatives=num_derivatives
    )
    strategy = strategies.DynamicFilter(implementation=implementation)
    control = controls.ProportionalIntegral()
    odefilter = _adaptive.AdaptiveODEFilter(
        strategy=strategy, control=control, atol=atol, rtol=rtol
    )
    information_op = information.isotropic_ek0(ode_order=1)
    return odefilter, information_op


def dynamic_isotropic_eks0(num_derivatives, atol=_ATOL_DEFAULTS, rtol=_RTOL_DEFAULTS):
    """Construct the equivalent of an explicit solver with an isotropic covariance \
    structure and dynamic calibration.

    Suitable for high-dimensional, non-stiff problems.
    """
    implementation = isotropic.IsotropicImplementation.from_num_derivatives(
        num_derivatives=num_derivatives
    )
    strategy = strategies.DynamicSmoother(implementation=implementation)
    control = controls.ProportionalIntegral()
    odefilter = _adaptive.AdaptiveODEFilter(
        strategy=strategy, control=control, atol=atol, rtol=rtol
    )
    information_op = information.isotropic_ek0(ode_order=1)
    return odefilter, information_op


def dynamic_isotropic_fixpt_eks0(
    num_derivatives, atol=_ATOL_DEFAULTS, rtol=_RTOL_DEFAULTS
):
    """Construct the equivalent of an explicit solver with an isotropic covariance \
    structure and dynamic calibration.

    Suitable for high-dimensional, non-stiff problems.
    """
    implementation = isotropic.IsotropicImplementation.from_num_derivatives(
        num_derivatives=num_derivatives
    )
    strategy = strategies.DynamicFixedPointSmoother(implementation=implementation)
    control = controls.ProportionalIntegral()
    odefilter = _adaptive.AdaptiveODEFilter(
        strategy=strategy, control=control, atol=atol, rtol=rtol
    )
    information_op = information.isotropic_ek0(ode_order=1)
    return odefilter, information_op


def dynamic_ekf1(
    num_derivatives, ode_dimension, atol=_ATOL_DEFAULTS, rtol=_RTOL_DEFAULTS
):
    """Construct the equivalent of a semi-implicit solver with dynamic calibration.

    Suitable for low-dimensional, stiff problems.
    """
    implementation = dense.DenseImplementation.from_num_derivatives(
        num_derivatives=num_derivatives, ode_dimension=ode_dimension
    )
    strategy = strategies.DynamicFilter(implementation=implementation)
    control = controls.ProportionalIntegral()
    odefilter = _adaptive.AdaptiveODEFilter(
        strategy=strategy, control=control, atol=atol, rtol=rtol
    )
    information_op = information.ek1(ode_dimension=ode_dimension, ode_order=1)
    return odefilter, information_op
