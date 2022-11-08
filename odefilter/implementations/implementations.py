"""Implementations."""

from typing import Generic, TypeVar

import jax

from odefilter.implementations import batch
from odefilter.implementations import correction as correction_module
from odefilter.implementations import dense
from odefilter.implementations import extrapolation as extrapolation_module
from odefilter.implementations import isotropic

ExtraType = TypeVar("ExtraType", bound=extrapolation_module.Extrapolation)
"""Extrapolation style."""


CorrType = TypeVar(
    "CorrType", bound=correction_module.Correction
)  # think: Correction style
"""Correction style."""


class AbstractImplementation(Generic[CorrType, ExtraType]):
    """Implementations.

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
class IsoTS0(AbstractImplementation[isotropic.IsoTaylorZerothOrder, isotropic.IsoIBM]):
    @classmethod
    def from_params(cls, *, ode_order=1, num_derivatives=4):
        correction = isotropic.IsoTaylorZerothOrder(ode_order=ode_order)
        extrapolation = isotropic.IsoIBM.from_params(num_derivatives=num_derivatives)
        return cls(correction=correction, extrapolation=extrapolation)


@jax.tree_util.register_pytree_node_class
class BatchMM1(AbstractImplementation[batch.BatchMomentMatching, batch.BatchIBM]):
    @classmethod
    def from_params(
        cls, *, ode_dimension, cubature=None, ode_order=1, num_derivatives=4
    ):
        if cubature is None:
            correction = batch.BatchMomentMatching.from_params(
                ode_dimension=ode_dimension, ode_order=ode_order
            )
        else:
            correction = batch.BatchMomentMatching(
                ode_dimension=ode_dimension, ode_order=ode_order, cubature=cubature
            )
        extrapolation = batch.BatchIBM.from_params(
            ode_dimension=ode_dimension, num_derivatives=num_derivatives
        )
        return cls(correction=correction, extrapolation=extrapolation)


@jax.tree_util.register_pytree_node_class
class BatchTS0(AbstractImplementation[batch.BatchMomentMatching, batch.BatchIBM]):
    @classmethod
    def from_params(cls, *, ode_dimension, ode_order=1, num_derivatives=4):
        correction = batch.BatchTaylorZerothOrder(ode_order=ode_order)
        extrapolation = batch.BatchIBM.from_params(
            ode_dimension=ode_dimension, num_derivatives=num_derivatives
        )
        return cls(correction=correction, extrapolation=extrapolation)


@jax.tree_util.register_pytree_node_class
class TS1(AbstractImplementation[dense.TaylorFirstOrder, dense.IBM]):
    @classmethod
    def from_params(cls, *, ode_dimension, ode_order=1, num_derivatives=4):
        correction = dense.TaylorFirstOrder(
            ode_dimension=ode_dimension, ode_order=ode_order
        )
        extrapolation = dense.IBM.from_params(
            ode_dimension=ode_dimension, num_derivatives=num_derivatives
        )
        return cls(correction=correction, extrapolation=extrapolation)


@jax.tree_util.register_pytree_node_class
class TS0(AbstractImplementation[dense.TaylorZerothOrder, dense.IBM]):
    @classmethod
    def from_params(cls, *, ode_dimension, ode_order=1, num_derivatives=4):
        correction = dense.TaylorZerothOrder(
            ode_dimension=ode_dimension, ode_order=ode_order
        )
        extrapolation = dense.IBM.from_params(
            ode_dimension=ode_dimension, num_derivatives=num_derivatives
        )
        return cls(correction=correction, extrapolation=extrapolation)


@jax.tree_util.register_pytree_node_class
class MM1(AbstractImplementation[dense.MomentMatching, dense.IBM]):
    @classmethod
    def from_params(
        cls, *, ode_dimension, cubature=None, ode_order=1, num_derivatives=4
    ):
        if cubature is None:
            correction = dense.MomentMatching.from_params(
                ode_dimension=ode_dimension, ode_order=ode_order
            )
        else:
            correction = dense.MomentMatching(
                ode_dimension=ode_dimension, ode_order=ode_order, cubature=cubature
            )
        extrapolation = dense.IBM.from_params(
            ode_dimension=ode_dimension, num_derivatives=num_derivatives
        )
        return cls(correction=correction, extrapolation=extrapolation)
