"""State-space model implementations."""

from typing import Generic, TypeVar

import jax

from probdiffeq.implementations import _collections
from probdiffeq.implementations.blockdiag import corr as blockdiag_corr
from probdiffeq.implementations.blockdiag import extra as blockdiag_extra
from probdiffeq.implementations.dense import corr as dense_corr
from probdiffeq.implementations.dense import extra as dense_extra
from probdiffeq.implementations.iso import corr as iso_corr
from probdiffeq.implementations.iso import extra as iso_extra

ExtraType = TypeVar("ExtraType", bound=_collections.AbstractExtrapolation)
"""A type-variable for an extrapolation style."""


CorrType = TypeVar("CorrType", bound=_collections.AbstractCorrection)
"""A type-variable for a correction style."""


class AbstractImplementation(Generic[CorrType, ExtraType]):
    """State-space model implementation.

    Mostly a container for an extrapolation style and a correction style.
    """

    def __init__(self, *, correction: CorrType, extrapolation: ExtraType):
        self.correction = correction
        self.extrapolation = extrapolation

    def tree_flatten(self):
        children = (self.correction, self.extrapolation)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        correction, extrapolation = children
        return cls(correction=correction, extrapolation=extrapolation)


@jax.tree_util.register_pytree_node_class
class IsoTS0(AbstractImplementation[iso_corr.IsoTaylorZerothOrder, iso_extra.IsoIBM]):
    @classmethod
    def from_params(cls, *, ode_order=1, num_derivatives=4):
        correction = iso_corr.IsoTaylorZerothOrder(ode_order=ode_order)
        extrapolation = iso_extra.IsoIBM.from_params(num_derivatives=num_derivatives)
        return cls(correction=correction, extrapolation=extrapolation)


@jax.tree_util.register_pytree_node_class
class BlockDiagSLR1(
    AbstractImplementation[
        blockdiag_corr.BlockDiagStatisticalFirstOrder, blockdiag_extra.BlockDiagIBM
    ]
):
    """First-order statistical linear regression in state-space models \
     with a block-diagonal structure.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This feature is highly experimental.
        There is no guarantee that it works correctly.
        It might be deleted tomorrow
        and without any deprecation policy.

    """

    @classmethod
    def from_params(cls, *, ode_shape, cubature=None, ode_order=1, num_derivatives=4):
        if cubature is None:
            correction = blockdiag_corr.BlockDiagStatisticalFirstOrder.from_params(
                ode_shape=ode_shape, ode_order=ode_order
            )
        else:
            correction = blockdiag_corr.BlockDiagStatisticalFirstOrder(
                ode_shape=ode_shape, ode_order=ode_order, cubature=cubature
            )
        extrapolation = blockdiag_extra.BlockDiagIBM.from_params(
            ode_shape=ode_shape, num_derivatives=num_derivatives
        )
        return cls(correction=correction, extrapolation=extrapolation)


@jax.tree_util.register_pytree_node_class
class BlockDiagTS0(
    AbstractImplementation[
        blockdiag_corr.BlockDiagStatisticalFirstOrder, blockdiag_extra.BlockDiagIBM
    ]
):
    @classmethod
    def from_params(cls, *, ode_shape, ode_order=1, num_derivatives=4):
        correction = blockdiag_corr.BlockDiagTaylorZerothOrder(ode_order=ode_order)
        extrapolation = blockdiag_extra.BlockDiagIBM.from_params(
            ode_shape=ode_shape, num_derivatives=num_derivatives
        )
        return cls(correction=correction, extrapolation=extrapolation)


@jax.tree_util.register_pytree_node_class
class DenseTS1(
    AbstractImplementation[dense_corr.DenseTaylorFirstOrder, dense_extra.DenseIBM]
):
    @classmethod
    def from_params(cls, *, ode_shape, ode_order=1, num_derivatives=4):
        correction = dense_corr.DenseTaylorFirstOrder(
            ode_shape=ode_shape, ode_order=ode_order
        )
        extrapolation = dense_extra.DenseIBM.from_params(
            ode_shape=ode_shape, num_derivatives=num_derivatives
        )
        return cls(correction=correction, extrapolation=extrapolation)


@jax.tree_util.register_pytree_node_class
class DenseTS0(
    AbstractImplementation[dense_corr.DenseTaylorZerothOrder, dense_extra.DenseIBM]
):
    @classmethod
    def from_params(cls, *, ode_shape, ode_order=1, num_derivatives=4):
        correction = dense_corr.DenseTaylorZerothOrder(
            ode_shape=ode_shape, ode_order=ode_order
        )
        extrapolation = dense_extra.DenseIBM.from_params(
            ode_shape=ode_shape, num_derivatives=num_derivatives
        )
        return cls(correction=correction, extrapolation=extrapolation)


@jax.tree_util.register_pytree_node_class
class DenseSLR1(
    AbstractImplementation[dense_corr.DenseStatisticalFirstOrder, dense_extra.DenseIBM]
):
    @classmethod
    def from_params(cls, *, ode_shape, cubature=None, ode_order=1, num_derivatives=4):
        correction = dense_corr.DenseStatisticalFirstOrder.from_params(
            ode_shape=ode_shape, ode_order=ode_order, cubature=cubature
        )
        extrapolation = dense_extra.DenseIBM.from_params(
            ode_shape=ode_shape, num_derivatives=num_derivatives
        )
        return cls(correction=correction, extrapolation=extrapolation)


@jax.tree_util.register_pytree_node_class
class DenseSLR0(
    AbstractImplementation[dense_corr.DenseStatisticalZerothOrder, dense_extra.DenseIBM]
):
    """Zeroth-order statistical linear regression in state-space models \
     with dense covariance structure.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This feature is highly experimental.
        There is no guarantee that it works correctly.
        It might be deleted tomorrow
        and without any deprecation policy.

    """

    @classmethod
    def from_params(cls, *, ode_shape, cubature=None, ode_order=1, num_derivatives=4):
        correction = dense_corr.DenseStatisticalZerothOrder.from_params(
            ode_shape=ode_shape, ode_order=ode_order, cubature=cubature
        )
        extrapolation = dense_extra.DenseIBM.from_params(
            ode_shape=ode_shape, num_derivatives=num_derivatives
        )
        return cls(correction=correction, extrapolation=extrapolation)
