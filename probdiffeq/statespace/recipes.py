"""State-space model recipes."""

from typing import Any, Tuple

from probdiffeq.backend import containers
from probdiffeq.statespace import _extra, calib, cubature
from probdiffeq.statespace.blockdiag import calib as bd_calib
from probdiffeq.statespace.blockdiag import corr as bd_corr
from probdiffeq.statespace.blockdiag import extra as bd_extra
from probdiffeq.statespace.dense import calib as dense_calib
from probdiffeq.statespace.dense import corr as dense_corr
from probdiffeq.statespace.dense import extra as dense_extra
from probdiffeq.statespace.iso import calib as iso_calib
from probdiffeq.statespace.iso import corr as iso_corr
from probdiffeq.statespace.iso import extra as iso_extra
from probdiffeq.statespace.scalar import calib as scalar_calib
from probdiffeq.statespace.scalar import corr as scalar_corr
from probdiffeq.statespace.scalar import extra as scalar_extra

# todo: make strategies into factory function.
#  This allows moving the calibration up to solver-level!


class _Impl(containers.NamedTuple):
    """State-space model implementation.

    Contains an extrapolation, correction, and calibration style.
    """

    extra_factory: Tuple[_extra.ExtrapolationFactory, Any]
    """Extrapolation factory."""

    corr: Any
    """Correction method."""

    calibration_factory: calib.CalibrationFactory
    """Calibration factory."""


def ts0_iso(*, ode_order=1, num_derivatives=4) -> _Impl:
    """Zeroth-order Taylor linearisation with isotropic Kronecker structure."""
    corr = iso_corr.taylor_order_zero(ode_order=ode_order)
    extra_factory = iso_extra.ibm_iso_factory(num_derivatives=num_derivatives)
    calibration_factory = iso_calib.output_scale()
    return _Impl(
        corr=corr, extra_factory=extra_factory, calibration_factory=calibration_factory
    )


def ts0_blockdiag(*, ode_shape, ode_order=1, num_derivatives=4) -> _Impl:
    corr = bd_corr.taylor_order_zero(ode_shape=ode_shape, ode_order=ode_order)
    extra_factory = bd_extra.ibm_blockdiag_factory(
        ode_shape=ode_shape, num_derivatives=num_derivatives
    )
    calibration_factory = bd_calib.output_scale(ode_shape=ode_shape)
    return _Impl(
        corr=corr, extra_factory=extra_factory, calibration_factory=calibration_factory
    )


def ts1_dense(*, ode_shape, ode_order=1, num_derivatives=4) -> _Impl:
    corr = dense_corr.taylor_order_one(ode_shape=ode_shape, ode_order=ode_order)
    extra_factory = dense_extra.ibm_dense_factory(
        ode_shape=ode_shape, num_derivatives=num_derivatives
    )
    calibration_factory = dense_calib.output_scale()
    return _Impl(
        corr=corr, extra_factory=extra_factory, calibration_factory=calibration_factory
    )


def ts0_dense(*, ode_shape, ode_order=1, num_derivatives=4) -> _Impl:
    corr = dense_corr.taylor_order_zero(ode_shape=ode_shape, ode_order=ode_order)
    extra_factory = dense_extra.ibm_dense_factory(
        ode_shape=ode_shape, num_derivatives=num_derivatives
    )
    calibration_factory = dense_calib.output_scale()
    return _Impl(
        corr=corr, extra_factory=extra_factory, calibration_factory=calibration_factory
    )


def slr1_dense(
    *,
    ode_shape,
    cubature_rule_fn=cubature.third_order_spherical,
    ode_order=1,
    num_derivatives=4,
) -> _Impl:
    corr = dense_corr.statistical_order_one(
        ode_shape=ode_shape, ode_order=ode_order, cubature_rule_fn=cubature_rule_fn
    )
    extra_factory = dense_extra.ibm_dense_factory(
        ode_shape=ode_shape, num_derivatives=num_derivatives
    )
    calibration_factory = dense_calib.output_scale()
    return _Impl(
        corr=corr, extra_factory=extra_factory, calibration_factory=calibration_factory
    )


def slr0_dense(
    *,
    ode_shape,
    cubature_rule_fn=cubature.third_order_spherical,
    ode_order=1,
    num_derivatives=4,
) -> _Impl:
    """Zeroth-order statistical linear regression in state-space models \
     with dense covariance structure.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This feature is highly experimental.
        There is no guarantee that it works correctly.
        It might be deleted tomorrow
        and without any deprecation policy.

    """
    corr = dense_corr.statistical_order_zero(
        ode_shape=ode_shape, ode_order=ode_order, cubature_rule_fn=cubature_rule_fn
    )
    extra_factory = dense_extra.ibm_dense_factory(
        ode_shape=ode_shape, num_derivatives=num_derivatives
    )
    calibration_factory = dense_calib.output_scale()
    return _Impl(
        corr=corr, extra_factory=extra_factory, calibration_factory=calibration_factory
    )


def ts0_scalar(*, ode_order=1, num_derivatives=4) -> _Impl:
    corr = scalar_corr.taylor_order_zero(ode_order=ode_order)
    extra_factory = scalar_extra.ibm_scalar_factory(num_derivatives=num_derivatives)
    calibration_factory = scalar_calib.output_scale()
    return _Impl(
        corr=corr, extra_factory=extra_factory, calibration_factory=calibration_factory
    )
