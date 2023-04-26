"""State-space model recipes."""

from typing import Any

from probdiffeq.backend import containers
from probdiffeq.statespace import cubature
from probdiffeq.statespace.blockdiag import corr as blockdiag_corr
from probdiffeq.statespace.blockdiag import extra as blockdiag_extra
from probdiffeq.statespace.dense import calib as dense_calib
from probdiffeq.statespace.dense import corr as dense_corr
from probdiffeq.statespace.dense import extra as dense_extra
from probdiffeq.statespace.iso import corr as iso_corr
from probdiffeq.statespace.iso import extra as iso_extra
from probdiffeq.statespace.scalar import corr as scalar_corr
from probdiffeq.statespace.scalar import extra as scalar_extra

# todo: make strategies into factory function.
#  This allows moving the calibration up to solver-level!


class _Impl(containers.NamedTuple):
    """State-space model implementation.

    Contains an extra style and a corr style.
    """

    extra: Any
    """Extrapolation method."""

    corr: Any
    """Correction method."""

    calib: Any
    """Calibration method."""


def ts0_iso(*, ode_order=1, num_derivatives=4) -> _Impl:
    """Zeroth-order Taylor linearisation with isotropic Kronecker structure."""
    corr = iso_corr.taylor_order_zero(ode_order=ode_order)
    extra = iso_extra.ibm_iso(num_derivatives=num_derivatives)
    return _Impl(corr=corr, extra=extra)


def slr1_blockdiag(*, ode_shape, ode_order=1, num_derivatives=4) -> _Impl:
    """First-order statistical linear regression in state-space models \
     with a block-diagonal structure.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This feature is highly experimental.
        There is no guarantee that it works correctly.
        It might be deleted tomorrow
        and without any deprecation policy.

    """
    corr = blockdiag_corr.statistical_order_one(
        ode_shape=ode_shape, ode_order=ode_order
    )
    extra = blockdiag_extra.ibm_blockdiag(
        ode_shape=ode_shape, num_derivatives=num_derivatives
    )
    return _Impl(corr=corr, extra=extra)


def ts0_blockdiag(*, ode_shape, ode_order=1, num_derivatives=4) -> _Impl:
    corr = blockdiag_corr.taylor_order_zero(ode_order=ode_order)
    extra = blockdiag_extra.ibm_blockdiag(
        ode_shape=ode_shape, num_derivatives=num_derivatives
    )
    return _Impl(corr=corr, extra=extra)


def ts1_dense(*, ode_shape, ode_order=1, num_derivatives=4) -> _Impl:
    corr = dense_corr.taylor_order_one(ode_shape=ode_shape, ode_order=ode_order)
    extra = dense_extra.ibm_dense(ode_shape=ode_shape, num_derivatives=num_derivatives)
    calib = dense_calib.output_scale_scalar()
    return _Impl(corr=corr, extra=extra, calib=calib)


def ts0_dense(*, ode_shape, ode_order=1, num_derivatives=4) -> _Impl:
    corr = dense_corr.taylor_order_zero(ode_shape=ode_shape, ode_order=ode_order)
    extra = dense_extra.ibm_dense(ode_shape=ode_shape, num_derivatives=num_derivatives)
    calib = dense_calib.output_scale_scalar()
    return _Impl(corr=corr, extra=extra, calib=calib)


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
    extra = dense_extra.ibm_dense(ode_shape=ode_shape, num_derivatives=num_derivatives)
    calib = dense_calib.output_scale_scalar()
    return _Impl(corr=corr, extra=extra, calib=calib)


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
    extra = dense_extra.ibm_dense(ode_shape=ode_shape, num_derivatives=num_derivatives)
    calib = dense_calib.output_scale_scalar()
    return _Impl(corr=corr, extra=extra, calib=calib)


def ts0_scalar(*, ode_order=1, num_derivatives=4) -> _Impl:
    corr = scalar_corr.taylor_order_zero(ode_order=ode_order)
    extra = scalar_extra.ibm_scalar(num_derivatives=num_derivatives)
    return _Impl(corr=corr, extra=extra)
