"""State-space model recipes."""
from probdiffeq.backend import containers
from probdiffeq.impl import impl
from probdiffeq.statespace import calibration, correction, cubature, extrapolation


class _Impl(containers.NamedTuple):
    """State-space model implementation.

    Contains an extrapolation, correction, and calibration style.
    """

    extrapolation: extrapolation.ExtrapolationFactory
    """Extrapolation factory."""

    correction: correction.Correction
    """Correction method."""

    calibration: calibration.CalibrationFactory
    """Calibration factory."""


def ts0_iso(*, ode_shape, ode_order=1, num_derivatives=4) -> _Impl:
    """Zeroth-order Taylor linearisation with isotropic Kronecker structure."""
    impl.select("isotropic", ode_shape=ode_shape)

    ts0 = correction.taylor_order_zero(ode_order=ode_order)
    ibm = extrapolation.ibm_adaptive(num_derivatives=num_derivatives)
    output_scale = calibration.output_scale()
    return _Impl(correction=ts0, extrapolation=ibm, calibration=output_scale)


def ts0_blockdiag(*, ode_shape, ode_order=1, num_derivatives=4) -> _Impl:
    impl.select("blockdiag", ode_shape=ode_shape)

    ts0 = correction.taylor_order_zero(ode_order=ode_order)
    ibm = extrapolation.ibm_adaptive(num_derivatives=num_derivatives)
    output_scale = calibration.output_scale()
    return _Impl(correction=ts0, extrapolation=ibm, calibration=output_scale)


def ts1_dense(*, ode_shape, ode_order=1, num_derivatives=4) -> _Impl:
    impl.select("dense", ode_shape=ode_shape)

    ts1 = correction.taylor_order_one(ode_order=ode_order)
    ibm = extrapolation.ibm_adaptive(num_derivatives=num_derivatives)
    output_scale = calibration.output_scale()
    return _Impl(correction=ts1, extrapolation=ibm, calibration=output_scale)


def ts0_dense(*, ode_shape, ode_order=1, num_derivatives=4) -> _Impl:
    impl.select("dense", ode_shape=ode_shape)

    ts0 = correction.taylor_order_zero(ode_order=ode_order)
    ibm = extrapolation.ibm_adaptive(num_derivatives=num_derivatives)
    output_scale = calibration.output_scale()
    return _Impl(correction=ts0, extrapolation=ibm, calibration=output_scale)


def slr1_dense(
    *,
    ode_shape,
    cubature_fun=cubature.third_order_spherical,
    num_derivatives=4,
) -> _Impl:
    impl.select("dense", ode_shape=ode_shape)

    slr1 = correction.statistical_order_one(cubature_fun)
    ibm = extrapolation.ibm_adaptive(num_derivatives=num_derivatives)
    output_scale = calibration.output_scale()
    return _Impl(correction=slr1, extrapolation=ibm, calibration=output_scale)


def slr0_dense(
    *,
    ode_shape,
    cubature_fun=cubature.third_order_spherical,
    num_derivatives=4,
) -> _Impl:
    """Zeroth-order statistical linear regression.

     In state-space models with dense covariance structure.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This feature is highly experimental.
        There is no guarantee that it works correctly.
        It might be deleted tomorrow
        and without any deprecation policy.

    """
    impl.select("dense", ode_shape=ode_shape)

    slr0 = correction.statistical_order_zero(cubature_fun)
    ibm = extrapolation.ibm_adaptive(num_derivatives=num_derivatives)
    output_scale = calibration.output_scale()
    return _Impl(correction=slr0, extrapolation=ibm, calibration=output_scale)


def ts0_scalar(*, ode_order=1, num_derivatives=4) -> _Impl:
    impl.select("scalar")

    ts0 = correction.taylor_order_zero(ode_order=ode_order)
    ibm = extrapolation.ibm_adaptive(num_derivatives=num_derivatives)
    output_scale = calibration.output_scale()
    return _Impl(correction=ts0, extrapolation=ibm, calibration=output_scale)
