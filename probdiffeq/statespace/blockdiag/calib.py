"""Calibration."""
import jax
import jax.numpy as jnp

from probdiffeq.statespace import _calib


def output_scale(output_scale_scalar, *, ode_shape):
    """Construct (a buffet of) isotropic calibration strategies."""
    mle = _blockdiag(output_scale_scalar.mle, ode_shape=ode_shape)
    dynamic = _blockdiag(output_scale_scalar.dynamic, ode_shape=ode_shape)
    free = _blockdiag(output_scale_scalar.free, ode_shape=ode_shape)

    return _calib.CalibrationBundle(mle=mle, dynamic=dynamic, free=free)


def _blockdiag(output_scale_scalar, *, ode_shape):
    @jax.tree_util.Partial
    def init(s, /):
        s_arr = s * jnp.ones(ode_shape)
        return jax.vmap(output_scale_scalar.init)(s_arr)

    @jax.tree_util.Partial
    def update(s, /):
        return jax.vmap(output_scale_scalar.update)(s)

    @jax.tree_util.Partial
    def extract(s):
        if s.ndim > 1:
            return s[-1, :]
        return s

    return _calib.Calib(init=init, update=update, extract=extract)
