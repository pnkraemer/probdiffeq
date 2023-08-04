"""Calibration."""
import jax
import jax.numpy as jnp

from probdiffeq.statespace import calib
from probdiffeq.statespace.scalar import calib as scalar_calib


def output_scale(ode_shape):
    """Construct (a buffet of) isotropic calibration strategies."""
    return BlockDiagFactory(ode_shape=ode_shape)


class BlockDiag(calib.Calibration):
    def __init__(self, wraps, *, ode_shape):
        self.wraps = wraps
        self.ode_shape = ode_shape

    def init(self, prior):
        if jnp.ndim(prior) == 0:
            raise ValueError

        return jax.vmap(self.wraps.init)(prior)

    def update(self, state, /, observed):
        print(observed.mean.shape)
        print(observed.cov_sqrtm_lower.shape)
        return jax.vmap(self.wraps.update)(state, observed)

    def extract(self, state, /):
        return jax.vmap(self.wraps.extract)(state)


class BlockDiagFactory(calib.CalibrationFactory):
    def __init__(self, *, ode_shape):
        self.ode_shape = ode_shape

    def most_recent(self):
        wraps = scalar_calib.ScalarMostRecent()
        return BlockDiag(wraps, ode_shape=self.ode_shape)

    def running_mean(self):
        wraps = scalar_calib.ScalarRunningMean()
        return BlockDiag(wraps, ode_shape=self.ode_shape)


# Register objects as (empty) pytrees. todo: temporary?!
def _flatten(node):
    return (node.wraps,), node.ode_shape


def _unflatten(nodetype, ode_shape, wraps):
    return nodetype(*wraps, ode_shape=ode_shape)


jax.tree_util.register_pytree_node(
    nodetype=BlockDiag,
    flatten_func=_flatten,
    unflatten_func=lambda *a: _unflatten(BlockDiag, *a),
)
