"""State-space model recipes."""
from probdiffeq.backend import containers, statespace
from probdiffeq.statespace import calib, corr, cubature, extra

# isort: off

import probdiffeq.statespace.blockdiag.calib
import probdiffeq.statespace.blockdiag.corr
import probdiffeq.statespace.blockdiag.extra

import probdiffeq.statespace.dense.calib
import probdiffeq.statespace.dense.corr
import probdiffeq.statespace.dense.extra

import probdiffeq.statespace.iso.calib
import probdiffeq.statespace.iso.corr
import probdiffeq.statespace.iso.extra

#
# import probdiffeq.statespace.scalar.calib
# import probdiffeq.statespace.scalar.corr
import probdiffeq.statespace.scalar.extra

# isort: on


class _Impl(containers.NamedTuple):
    """State-space model implementation.

    Contains an extrapolation, correction, and calibration style.
    """

    extrapolation: extra.ExtrapolationFactory
    """Extrapolation factory."""

    correction: corr.Correction
    """Correction method."""

    calibration: calib.CalibrationFactory
    """Calibration factory."""


def ts0_iso(*, ode_order=1, num_derivatives=4) -> _Impl:
    """Zeroth-order Taylor linearisation with isotropic Kronecker structure."""
    correction = probdiffeq.statespace.iso.corr.taylor_order_zero(ode_order=ode_order)
    ibm = probdiffeq.statespace.iso.extra.ibm_factory(num_derivatives=num_derivatives)
    calibration = probdiffeq.statespace.calib.output_scale()
    return _Impl(correction=correction, extrapolation=ibm, calibration=calibration)


def ts0_blockdiag(*, ode_shape, ode_order=1, num_derivatives=4) -> _Impl:
    correction = probdiffeq.statespace.blockdiag.corr.taylor_order_zero(
        ode_shape=ode_shape, ode_order=ode_order
    )
    ibm = probdiffeq.statespace.blockdiag.extra.ibm_factory(
        ode_shape=ode_shape, num_derivatives=num_derivatives
    )
    calibration = probdiffeq.statespace.calib.output_scale()
    return _Impl(correction=correction, extrapolation=ibm, calibration=calibration)


def ts1_dense(*, ode_shape, ode_order=1, num_derivatives=4) -> _Impl:
    statespace.select("dense", ode_shape=ode_shape)

    correction = probdiffeq.statespace.corr.taylor_order_one(ode_order=ode_order)
    ibm = probdiffeq.statespace.extra.ibm_factory(num_derivatives=num_derivatives)
    calibration = probdiffeq.statespace.calib.output_scale()
    return _Impl(correction=correction, extrapolation=ibm, calibration=calibration)


def ts0_dense(*, ode_shape, ode_order=1, num_derivatives=4) -> _Impl:
    statespace.select("dense", ode_shape=ode_shape)

    correction = probdiffeq.statespace.corr.taylor_order_zero(ode_order=ode_order)
    ibm = probdiffeq.statespace.extra.ibm_factory(num_derivatives=num_derivatives)
    calibration = probdiffeq.statespace.calib.output_scale()
    return _Impl(correction=correction, extrapolation=ibm, calibration=calibration)


def slr1_dense(
    *,
    ode_shape,
    cubature_rule_fn=cubature.third_order_spherical,
    ode_order=1,
    num_derivatives=4,
) -> _Impl:
    correction = probdiffeq.statespace.dense.corr.statistical_order_one(
        ode_shape=ode_shape, ode_order=ode_order, cubature_rule_fn=cubature_rule_fn
    )
    ibm = probdiffeq.statespace.dense.extra.ibm_factory(
        ode_shape=ode_shape, num_derivatives=num_derivatives
    )
    calibration = probdiffeq.statespace.calib.output_scale()
    return _Impl(correction=correction, extrapolation=ibm, calibration=calibration)


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
    correction = probdiffeq.statespace.dense.corr.statistical_order_zero(
        ode_shape=ode_shape, ode_order=ode_order, cubature_rule_fn=cubature_rule_fn
    )
    ibm = probdiffeq.statespace.dense.extra.ibm_factory(
        ode_shape=ode_shape, num_derivatives=num_derivatives
    )
    calibration = probdiffeq.statespace.calib.output_scale()
    return _Impl(correction=correction, extrapolation=ibm, calibration=calibration)


def ts0_scalar(*, ode_order=1, num_derivatives=4) -> _Impl:
    statespace.select("scalar")

    correction = probdiffeq.statespace.corr.taylor_order_zero(ode_order=ode_order)
    ibm = probdiffeq.statespace.extra.ibm_factory(num_derivatives=num_derivatives)
    calibration = probdiffeq.statespace.calib.output_scale()
    return _Impl(correction=correction, extrapolation=ibm, calibration=calibration)
