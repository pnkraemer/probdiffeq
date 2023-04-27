"""Calibration."""
import jax
import jax.numpy as jnp

from probdiffeq.statespace import _calib


def output_scale(output_scale_scalar, *, ode_shape):
    @jax.tree_util.Partial
    def init(s, /):
        s_arr = s * jnp.ones(ode_shape)
        return jax.vmap(output_scale_scalar.init)(s_arr)

    @jax.tree_util.Partial
    def extract(s):
        if s.ndim > 1:
            return s[-1, :]
        return s

    return _calib.Calib(init=init, extract=extract)
