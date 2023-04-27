"""Calibration tools."""

import jax

from probdiffeq.statespace import _calib


def output_scale():
    """Scalar output scale."""

    @jax.tree_util.Partial
    def init(s, /):
        return s

    @jax.tree_util.Partial
    def extract(s, /):
        if s.ndim > 0:
            return s[-1]
        return s

    return _calib.Calib(init=init, extract=extract)
