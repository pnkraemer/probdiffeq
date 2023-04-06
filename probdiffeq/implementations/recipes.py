"""State-space model implementations."""

import jax

from probdiffeq.implementations import cubature
from probdiffeq.implementations.blockdiag import corr as blockdiag_corr
from probdiffeq.implementations.blockdiag import extra as blockdiag_extra
from probdiffeq.implementations.dense import corr as dense_corr
from probdiffeq.implementations.dense import extra as dense_extra
from probdiffeq.implementations.iso import corr as iso_corr
from probdiffeq.implementations.iso import extra as iso_extra
from probdiffeq.implementations.scalar import corr as scalar_corr
from probdiffeq.implementations.scalar import extra as scalar_extra

# todo: why are these classes? Why not plain functions?
#  nothing is happening in here, really.


class AbstractImplementation:
    """State-space model implementation.

    Mostly a container for an extrapolation style and a correction style.
    """

    def __init__(self, *, correction, extrapolation):
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

    def __repr__(self):
        name = self.__class__.__name__
        n = self.extrapolation.num_derivatives
        return f"<{name} with num_derivatives={n}>"


@jax.tree_util.register_pytree_node_class
class IsoTS0(AbstractImplementation):
    @classmethod
    def from_params(cls, *, ode_order=1, num_derivatives=4):
        correction = iso_corr.IsoTaylorZerothOrder(ode_order=ode_order)
        extrapolation = iso_extra.ibm_iso(num_derivatives=num_derivatives)
        return cls(correction=correction, extrapolation=extrapolation)


@jax.tree_util.register_pytree_node_class
class BlockDiagSLR1(AbstractImplementation):
    """First-order statistical linear regression in state-space models \
     with a block-diagonal structure.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This feature is highly experimental.
        There is no guarantee that it works correctly.
        It might be deleted tomorrow
        and without any deprecation policy.

    """

    @classmethod
    def from_params(
        cls, *, ode_shape, cubature_rule=None, ode_order=1, num_derivatives=4
    ):
        if cubature_rule is None:
            correction = blockdiag_corr.BlockDiagStatisticalFirstOrder.from_params(
                ode_shape=ode_shape, ode_order=ode_order
            )
        else:
            correction = blockdiag_corr.BlockDiagStatisticalFirstOrder(
                ode_shape=ode_shape, ode_order=ode_order, cubature_rule=cubature_rule
            )
        extrapolation = blockdiag_extra.ibm_blockdiag(
            ode_shape=ode_shape, num_derivatives=num_derivatives
        )
        return cls(correction=correction, extrapolation=extrapolation)


@jax.tree_util.register_pytree_node_class
class BlockDiagTS0(AbstractImplementation):
    @classmethod
    def from_params(cls, *, ode_shape, ode_order=1, num_derivatives=4):
        correction = blockdiag_corr.taylor_order_zero(ode_order=ode_order)
        extrapolation = blockdiag_extra.ibm_blockdiag(
            ode_shape=ode_shape, num_derivatives=num_derivatives
        )
        return cls(correction=correction, extrapolation=extrapolation)


@jax.tree_util.register_pytree_node_class
class DenseTS1(AbstractImplementation):
    @classmethod
    def from_params(cls, *, ode_shape, ode_order=1, num_derivatives=4):
        correction = dense_corr.taylor_order_one(
            ode_shape=ode_shape, ode_order=ode_order
        )
        extrapolation = dense_extra.ibm_dense(
            ode_shape=ode_shape, num_derivatives=num_derivatives
        )
        return cls(correction=correction, extrapolation=extrapolation)


@jax.tree_util.register_pytree_node_class
class DenseTS0(AbstractImplementation):
    @classmethod
    def from_params(cls, *, ode_shape, ode_order=1, num_derivatives=4):
        correction = dense_corr.taylor_order_zero(
            ode_shape=ode_shape, ode_order=ode_order
        )
        extrapolation = dense_extra.ibm_dense(
            ode_shape=ode_shape, num_derivatives=num_derivatives
        )
        return cls(correction=correction, extrapolation=extrapolation)


@jax.tree_util.register_pytree_node_class
class DenseSLR1(AbstractImplementation):
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


@jax.tree_util.register_pytree_node_class
class DenseSLR0(AbstractImplementation):
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


@jax.tree_util.register_pytree_node_class
class ScalarTS0(AbstractImplementation):
    @classmethod
    def from_params(cls, *, ode_order=1, num_derivatives=4):
        correction = scalar_corr.taylor_order_zero(ode_order=ode_order)
        extrapolation = scalar_extra.ibm_scalar(num_derivatives=num_derivatives)
        return cls(correction=correction, extrapolation=extrapolation)
