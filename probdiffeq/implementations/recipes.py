"""Implementations."""

from typing import Generic, TypeVar

import jax

from probdiffeq.implementations import _batch, _collections, _iso, _vect

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
class IsoTS0(AbstractImplementation[_iso.IsoTaylorZerothOrder, _iso.IsoIBM]):
    @classmethod
    def from_params(cls, *, ode_order=1, num_derivatives=4):
        correction = _iso.IsoTaylorZerothOrder(ode_order=ode_order)
        extrapolation = _iso.IsoIBM.from_params(num_derivatives=num_derivatives)
        return cls(correction=correction, extrapolation=extrapolation)


@jax.tree_util.register_pytree_node_class
class BatchMM1(AbstractImplementation[_batch.BatchMomentMatching, _batch.BatchIBM]):
    @classmethod
    def from_params(cls, *, ode_shape, cubature=None, ode_order=1, num_derivatives=4):
        if cubature is None:
            correction = _batch.BatchMomentMatching.from_params(
                ode_shape=ode_shape, ode_order=ode_order
            )
        else:
            correction = _batch.BatchMomentMatching(
                ode_shape=ode_shape, ode_order=ode_order, cubature=cubature
            )
        extrapolation = _batch.BatchIBM.from_params(
            ode_shape=ode_shape, num_derivatives=num_derivatives
        )
        return cls(correction=correction, extrapolation=extrapolation)


@jax.tree_util.register_pytree_node_class
class BatchTS0(AbstractImplementation[_batch.BatchMomentMatching, _batch.BatchIBM]):
    @classmethod
    def from_params(cls, *, ode_shape, ode_order=1, num_derivatives=4):
        correction = _batch.BatchTaylorZerothOrder(ode_order=ode_order)
        extrapolation = _batch.BatchIBM.from_params(
            ode_shape=ode_shape, num_derivatives=num_derivatives
        )
        return cls(correction=correction, extrapolation=extrapolation)


@jax.tree_util.register_pytree_node_class
class VectTS1(AbstractImplementation[_vect.VectTaylorFirstOrder, _vect.VectIBM]):
    @classmethod
    def from_params(cls, *, ode_shape, ode_order=1, num_derivatives=4):
        correction = _vect.VectTaylorFirstOrder(
            ode_shape=ode_shape, ode_order=ode_order
        )
        extrapolation = _vect.VectIBM.from_params(
            ode_shape=ode_shape, num_derivatives=num_derivatives
        )
        return cls(correction=correction, extrapolation=extrapolation)


@jax.tree_util.register_pytree_node_class
class VectTS0(AbstractImplementation[_vect.VectTaylorZerothOrder, _vect.VectIBM]):
    @classmethod
    def from_params(cls, *, ode_shape, ode_order=1, num_derivatives=4):
        correction = _vect.VectTaylorZerothOrder(
            ode_shape=ode_shape, ode_order=ode_order
        )
        extrapolation = _vect.VectIBM.from_params(
            ode_shape=ode_shape, num_derivatives=num_derivatives
        )
        return cls(correction=correction, extrapolation=extrapolation)


@jax.tree_util.register_pytree_node_class
class VectMM1(AbstractImplementation[_vect.VectMomentMatching, _vect.VectIBM]):
    @classmethod
    def from_params(cls, *, ode_shape, cubature=None, ode_order=1, num_derivatives=4):
        if cubature is None:
            correction = _vect.VectMomentMatching.from_params(
                ode_shape=ode_shape, ode_order=ode_order
            )
        else:
            correction = _vect.VectMomentMatching(
                ode_shape=ode_shape, ode_order=ode_order, cubature=cubature
            )
        extrapolation = _vect.VectIBM.from_params(
            ode_shape=ode_shape, num_derivatives=num_derivatives
        )
        return cls(correction=correction, extrapolation=extrapolation)
