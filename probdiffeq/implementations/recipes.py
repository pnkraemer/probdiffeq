"""Implementations."""

from typing import Generic, TypeVar

import jax

from probdiffeq.implementations import _collections, batch, iso, vect

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
class IsoTS0(AbstractImplementation[iso.IsoTaylorZerothOrder, iso.IsoIBM]):
    @classmethod
    def from_params(cls, *, ode_order=1, num_derivatives=4):
        correction = iso.IsoTaylorZerothOrder(ode_order=ode_order)
        extrapolation = iso.IsoIBM.from_params(num_derivatives=num_derivatives)
        return cls(correction=correction, extrapolation=extrapolation)


@jax.tree_util.register_pytree_node_class
class BatchMM1(AbstractImplementation[batch.BatchMomentMatching, batch.BatchIBM]):
    @classmethod
    def from_params(cls, *, ode_shape, cubature=None, ode_order=1, num_derivatives=4):
        if cubature is None:
            correction = batch.BatchMomentMatching.from_params(
                ode_shape=ode_shape, ode_order=ode_order
            )
        else:
            correction = batch.BatchMomentMatching(
                ode_shape=ode_shape, ode_order=ode_order, cubature=cubature
            )
        extrapolation = batch.BatchIBM.from_params(
            ode_shape=ode_shape, num_derivatives=num_derivatives
        )
        return cls(correction=correction, extrapolation=extrapolation)


@jax.tree_util.register_pytree_node_class
class BatchTS0(AbstractImplementation[batch.BatchMomentMatching, batch.BatchIBM]):
    @classmethod
    def from_params(cls, *, ode_shape, ode_order=1, num_derivatives=4):
        correction = batch.BatchTaylorZerothOrder(ode_order=ode_order)
        extrapolation = batch.BatchIBM.from_params(
            ode_shape=ode_shape, num_derivatives=num_derivatives
        )
        return cls(correction=correction, extrapolation=extrapolation)


@jax.tree_util.register_pytree_node_class
class VectTS1(AbstractImplementation[vect.VectTaylorFirstOrder, vect.VectIBM]):
    @classmethod
    def from_params(cls, *, ode_shape, ode_order=1, num_derivatives=4):
        correction = vect.VectTaylorFirstOrder(ode_shape=ode_shape, ode_order=ode_order)
        extrapolation = vect.VectIBM.from_params(
            ode_shape=ode_shape, num_derivatives=num_derivatives
        )
        return cls(correction=correction, extrapolation=extrapolation)


@jax.tree_util.register_pytree_node_class
class VectTS0(AbstractImplementation[vect.VectTaylorZerothOrder, vect.VectIBM]):
    @classmethod
    def from_params(cls, *, ode_shape, ode_order=1, num_derivatives=4):
        correction = vect.VectTaylorZerothOrder(
            ode_shape=ode_shape, ode_order=ode_order
        )
        extrapolation = vect.VectIBM.from_params(
            ode_shape=ode_shape, num_derivatives=num_derivatives
        )
        return cls(correction=correction, extrapolation=extrapolation)


@jax.tree_util.register_pytree_node_class
class VectMM1(AbstractImplementation[vect.VectMomentMatching, vect.VectIBM]):
    @classmethod
    def from_params(cls, *, ode_shape, cubature=None, ode_order=1, num_derivatives=4):
        if cubature is None:
            correction = vect.VectMomentMatching.from_params(
                ode_shape=ode_shape, ode_order=ode_order
            )
        else:
            correction = vect.VectMomentMatching(
                ode_shape=ode_shape, ode_order=ode_order, cubature=cubature
            )
        extrapolation = vect.VectIBM.from_params(
            ode_shape=ode_shape, num_derivatives=num_derivatives
        )
        return cls(correction=correction, extrapolation=extrapolation)
