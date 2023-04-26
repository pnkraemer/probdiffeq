"""Calibration tools."""

from typing import Any

import jax

from probdiffeq.backend import containers


class _Calib(containers.NamedTuple):
    # todo: no "apply" method yet. Maybe in the future.

    init: Any
    extract: Any


def output_scale_scalar():
    """Scalar output-scale."""

    @jax.tree_util.Partial
    def init(output_scale):
        return output_scale

    @jax.tree_util.Partial
    def extract(output_scale):
        if output_scale.ndim > 0:
            return output_scale[-1]
        return output_scale

    return _Calib(init=init, extract=extract)
