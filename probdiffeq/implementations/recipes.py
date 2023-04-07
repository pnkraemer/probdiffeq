"""State-space model implementations."""

from typing import Any, NamedTuple

from probdiffeq.implementations import cubature
from probdiffeq.implementations.blockdiag import corr as blockdiag_corr
from probdiffeq.implementations.blockdiag import extra as blockdiag_extra
from probdiffeq.implementations.dense import corr as dense_corr
from probdiffeq.implementations.dense import extra as dense_extra
from probdiffeq.implementations.iso import corr as iso_corr
from probdiffeq.implementations.iso import extra as iso_extra
from probdiffeq.implementations.scalar import corr as scalar_corr
from probdiffeq.implementations.scalar import extra as scalar_extra


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


def ts0_iso(*, ode_order=1, num_derivatives=4):
    """Zeroth-order Taylor linearisation with isotropic Kronecker structure."""
    correction = iso_corr.taylor_order_zero(ode_order=ode_order)
    extrapolation = iso_extra.ibm_iso(num_derivatives=num_derivatives)
    return Implementation(correction=correction, extrapolation=extrapolation)


def slr1_blockdiag(*, ode_shape, cubature_rule=None, ode_order=1, num_derivatives=4):
    """First-order statistical linear regression in state-space models \
     with a block-diagonal structure.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This feature is highly experimental.
        There is no guarantee that it works correctly.
        It might be deleted tomorrow
        and without any deprecation policy.

    """
    if cubature_rule is None:
        correction = blockdiag_corr.statistical_order_one(
            ode_shape=ode_shape, ode_order=ode_order
        )
    else:
        correction = blockdiag_corr.statistical_order_one(
            ode_shape=ode_shape, ode_order=ode_order, cubature_rule=cubature_rule
        )
    extrapolation = blockdiag_extra.ibm_blockdiag(
        ode_shape=ode_shape, num_derivatives=num_derivatives
    )
    return Implementation(correction=correction, extrapolation=extrapolation)


class BlockDiagTS0(Implementation):
    @classmethod
    def from_params(cls, *, ode_shape, ode_order=1, num_derivatives=4):
        correction = blockdiag_corr.taylor_order_zero(ode_order=ode_order)
        extrapolation = blockdiag_extra.ibm_blockdiag(
            ode_shape=ode_shape, num_derivatives=num_derivatives
        )
        return cls(correction=correction, extrapolation=extrapolation)


class DenseTS1(Implementation):
    @classmethod
    def from_params(cls, *, ode_shape, ode_order=1, num_derivatives=4):
        correction = dense_corr.taylor_order_one(
            ode_shape=ode_shape, ode_order=ode_order
        )
        extrapolation = dense_extra.ibm_dense(
            ode_shape=ode_shape, num_derivatives=num_derivatives
        )
        return cls(correction=correction, extrapolation=extrapolation)


class DenseTS0(Implementation):
    @classmethod
    def from_params(cls, *, ode_shape, ode_order=1, num_derivatives=4):
        correction = dense_corr.taylor_order_zero(
            ode_shape=ode_shape, ode_order=ode_order
        )
        extrapolation = dense_extra.ibm_dense(
            ode_shape=ode_shape, num_derivatives=num_derivatives
        )
        return cls(correction=correction, extrapolation=extrapolation)


class DenseSLR1(Implementation):
    @classmethod
    def from_params(
        cls,
        *,
        ode_shape,
        cubature_rule_fn=cubature.third_order_spherical,
        ode_order=1,
        num_derivatives=4,
    ):
        correction = dense_corr.statistical_order_one(
            ode_shape=ode_shape, ode_order=ode_order, cubature_rule_fn=cubature_rule_fn
        )
        extrapolation = dense_extra.ibm_dense(
            ode_shape=ode_shape, num_derivatives=num_derivatives
        )
        return cls(correction=correction, extrapolation=extrapolation)


class DenseSLR0(Implementation):
    """Zeroth-order statistical linear regression in state-space models \
     with dense covariance structure.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This feature is highly experimental.
        There is no guarantee that it works correctly.
        It might be deleted tomorrow
        and without any deprecation policy.

    """

    @classmethod
    def from_params(
        cls,
        *,
        ode_shape,
        cubature_rule_fn=cubature.third_order_spherical,
        ode_order=1,
        num_derivatives=4,
    ):
        correction = dense_corr.statistical_order_zero(
            ode_shape=ode_shape, ode_order=ode_order, cubature_rule_fn=cubature_rule_fn
        )
        extrapolation = dense_extra.ibm_dense(
            ode_shape=ode_shape, num_derivatives=num_derivatives
        )
        return cls(correction=correction, extrapolation=extrapolation)


class ScalarTS0(Implementation):
    @classmethod
    def from_params(cls, *, ode_order=1, num_derivatives=4):
        correction = scalar_corr.taylor_order_zero(ode_order=ode_order)
        extrapolation = scalar_extra.ibm_scalar(num_derivatives=num_derivatives)
        return cls(correction=correction, extrapolation=extrapolation)
