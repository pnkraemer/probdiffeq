"""Implementations."""

from typing import Generic, TypeVar

import jax

from probdiffeq.implementations import _collections
from probdiffeq.implementations.batch import corr as batch_corr
from probdiffeq.implementations.batch import extra as batch_extra
from probdiffeq.implementations.dense import corr as dense_corr
from probdiffeq.implementations.dense import extra as dense_extra
from probdiffeq.implementations.iso import corr as iso_corr
from probdiffeq.implementations.iso import extra as iso_extra

ExtraType = TypeVar("ExtraType", bound=_collections.AbstractExtrapolation)
"""Extrapolation style."""


CorrType = TypeVar("CorrType", bound=_collections.AbstractCorrection)
"""Correction style."""


class AbstractImplementation(Generic[CorrType, ExtraType]):
    """Solver / inference implementations.

    Mostly a container for an extrapolation method and a correction method.
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
class BatchSLR1(
    AbstractImplementation[batch_corr.BatchStatisticalFirstOrder, batch_extra.BatchIBM]
):
    @classmethod
    def from_params(cls, *, ode_shape, cubature=None, ode_order=1, num_derivatives=4):
        if cubature is None:
            correction = batch_corr.BatchStatisticalFirstOrder.from_params(
                ode_shape=ode_shape, ode_order=ode_order
            )
        else:
            correction = batch_corr.BatchStatisticalFirstOrder(
                ode_shape=ode_shape, ode_order=ode_order, cubature=cubature
            )
        extrapolation = batch_extra.BatchIBM.from_params(
            ode_shape=ode_shape, num_derivatives=num_derivatives
        )
        return cls(correction=correction, extrapolation=extrapolation)


@jax.tree_util.register_pytree_node_class
class BatchTS0(
    AbstractImplementation[batch_corr.BatchStatisticalFirstOrder, batch_extra.BatchIBM]
):
    @classmethod
    def from_params(cls, *, ode_shape, ode_order=1, num_derivatives=4):
        correction = batch_corr.BatchTaylorZerothOrder(ode_order=ode_order)
        extrapolation = batch_extra.BatchIBM.from_params(
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
    @classmethod
    def from_params(cls, *, ode_shape, cubature=None, ode_order=1, num_derivatives=4):
        correction = dense_corr.DenseStatisticalZerothOrder.from_params(
            ode_shape=ode_shape, ode_order=ode_order, cubature=cubature
        )
        extrapolation = dense_extra.DenseIBM.from_params(
            ode_shape=ode_shape, num_derivatives=num_derivatives
        )
        return cls(correction=correction, extrapolation=extrapolation)
