"""Calibration."""
import jax
import jax.numpy as jnp

from probdiffeq import _sqrt_util
from probdiffeq.statespace import calib


def output_scale():
    """Construct (a buffet of) isotropic calibration strategies."""
    return CalibrationFactory()


class ScalarMostRecent(calib.Calibration):
    def init(self, prior):
        return prior

    def update(self, _state, /, observed):
        zero_data = jnp.zeros_like(observed.mean)
        mahalanobis_norm = observed.mahalanobis_norm(zero_data)
        calibrated = mahalanobis_norm / jnp.sqrt(zero_data.size)
        return calibrated

    def extract(self, state, /):
        return state, state


class ScalarRunningMean(calib.Calibration):
    def init(self, prior):
        return prior, prior, 0.0

    def update(self, state, /, observed):
        prior, calibrated, num_data = state

        zero_data = jnp.zeros_like(observed.mean)
        mahalanobis_norm = observed.mahalanobis_norm(zero_data)
        new_term = mahalanobis_norm / jnp.sqrt(zero_data.size)

        calibrated = _update_running_mean(calibrated, new_term, num=num_data)
        return prior, calibrated, num_data + 1.0

    def extract(self, state, /):
        prior, calibrated, _num_data = state
        return prior, calibrated


def _update_running_mean(mean, x, /, num):
    sum_updated = _sqrt_util.sqrt_sum_square_scalar(jnp.sqrt(num) * mean, x)
    return sum_updated / jnp.sqrt(num + 1)


class CalibrationFactory(calib.CalibrationFactory):
    pass


# todo: run tests for scalar solvers

# Register objects as (empty) pytrees. todo: temporary?!
jax.tree_util.register_pytree_node(
    nodetype=ScalarRunningMean,
    flatten_func=lambda _: ((), ()),
    unflatten_func=lambda *a: ScalarRunningMean(),
)
jax.tree_util.register_pytree_node(
    nodetype=ScalarMostRecent,
    flatten_func=lambda _: ((), ()),
    unflatten_func=lambda *a: ScalarMostRecent(),
)
