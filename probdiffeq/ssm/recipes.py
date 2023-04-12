"""State-space model ssm."""

from typing import Any, NamedTuple

from probdiffeq.ssm import cubature
from probdiffeq.ssm.blockdiag import corr as blockdiag_corr
from probdiffeq.ssm.blockdiag import extra as blockdiag_extra
from probdiffeq.ssm.dense import corr as dense_corr
from probdiffeq.ssm.dense import extra as dense_extra
from probdiffeq.ssm.iso import corr as iso_corr
from probdiffeq.ssm.iso import extra as iso_extra
from probdiffeq.ssm.scalar import corr as scalar_corr
from probdiffeq.ssm.scalar import extra as scalar_extra


class Implementation(NamedTuple):
    """State-space model implementation.

    Contains an extrapolation style and a correction style.
    """

    correction: Any
    extrapolation: Any

    def __repr__(self):
        name = self.__class__.__name__
        n = self.extrapolation.num_derivatives
        o = self.correction.ode_order
        return f"<{name} with num_derivatives={n}, ode_order={o}>"


def ts0_iso(*, ode_order=1, num_derivatives=4) -> Implementation:
    """Zeroth-order Taylor linearisation with isotropic Kronecker structure."""
    correction = iso_corr.taylor_order_zero(ode_order=ode_order)
    extrapolation = iso_extra.ibm_iso(num_derivatives=num_derivatives)
    return Implementation(correction=correction, extrapolation=extrapolation)


def slr1_blockdiag(*, ode_shape, ode_order=1, num_derivatives=4) -> Implementation:
    """First-order statistical linear regression in state-space models \
     with a block-diagonal structure.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This feature is highly experimental.
        There is no guarantee that it works correctly.
        It might be deleted tomorrow
        and without any deprecation policy.

    """
    correction = blockdiag_corr.statistical_order_one(
        ode_shape=ode_shape, ode_order=ode_order
    )
    extrapolation = blockdiag_extra.ibm_blockdiag(
        ode_shape=ode_shape, num_derivatives=num_derivatives
    )
    return Implementation(correction=correction, extrapolation=extrapolation)


def ts0_blockdiag(*, ode_shape, ode_order=1, num_derivatives=4) -> Implementation:
    correction = blockdiag_corr.taylor_order_zero(ode_order=ode_order)
    extrapolation = blockdiag_extra.ibm_blockdiag(
        ode_shape=ode_shape, num_derivatives=num_derivatives
    )
    return Implementation(correction=correction, extrapolation=extrapolation)


def ts1_dense(*, ode_shape, ode_order=1, num_derivatives=4) -> Implementation:
    correction = dense_corr.taylor_order_one(ode_shape=ode_shape, ode_order=ode_order)
    extrapolation = dense_extra.ibm_dense(
        ode_shape=ode_shape, num_derivatives=num_derivatives
    )
    return Implementation(correction=correction, extrapolation=extrapolation)


def ts0_dense(*, ode_shape, ode_order=1, num_derivatives=4) -> Implementation:
    correction = dense_corr.taylor_order_zero(ode_shape=ode_shape, ode_order=ode_order)
    extrapolation = dense_extra.ibm_dense(
        ode_shape=ode_shape, num_derivatives=num_derivatives
    )
    return Implementation(correction=correction, extrapolation=extrapolation)


def slr1_dense(
    *,
    ode_shape,
    cubature_rule_fn=cubature.third_order_spherical,
    ode_order=1,
    num_derivatives=4,
) -> Implementation:
    correction = dense_corr.statistical_order_one(
        ode_shape=ode_shape, ode_order=ode_order, cubature_rule_fn=cubature_rule_fn
    )
    extrapolation = dense_extra.ibm_dense(
        ode_shape=ode_shape, num_derivatives=num_derivatives
    )
    return Implementation(correction=correction, extrapolation=extrapolation)


def slr0_dense(
    *,
    ode_shape,
    cubature_rule_fn=cubature.third_order_spherical,
    ode_order=1,
    num_derivatives=4,
) -> Implementation:
    """Zeroth-order statistical linear regression in state-space models \
     with dense covariance structure.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This feature is highly experimental.
        There is no guarantee that it works correctly.
        It might be deleted tomorrow
        and without any deprecation policy.

    """
    correction = dense_corr.statistical_order_zero(
        ode_shape=ode_shape, ode_order=ode_order, cubature_rule_fn=cubature_rule_fn
    )
    extrapolation = dense_extra.ibm_dense(
        ode_shape=ode_shape, num_derivatives=num_derivatives
    )
    return Implementation(correction=correction, extrapolation=extrapolation)


def ts0_scalar(*, ode_order=1, num_derivatives=4) -> Implementation:
    correction = scalar_corr.taylor_order_zero(ode_order=ode_order)
    extrapolation = scalar_extra.ibm_scalar(num_derivatives=num_derivatives)
    return Implementation(correction=correction, extrapolation=extrapolation)
