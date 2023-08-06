"""Calibration API."""


import abc

import jax
import jax.numpy as jnp

from probdiffeq.backend import statespace


def output_scale():
    """Construct (a buffet of) isotropic calibration strategies."""
    return CalibrationFactory()


class Calibration(abc.ABC):
    """Calibration implementation."""

    @abc.abstractmethod
    def init(self, prior):
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, state, /, observed):
        raise NotImplementedError

    @abc.abstractmethod
    def extract(self, state, /):
        raise NotImplementedError


class MostRecent(Calibration):
    def init(self, prior):
        return prior

    def update(self, _state, /, observed):
        zero_data = jnp.zeros_like(statespace.random.mean(observed))
        mahalanobis_norm = statespace.random.mahalanobis_norm(zero_data, observed)
        calibrated = mahalanobis_norm / jnp.sqrt(zero_data.size)
        return calibrated

    def extract(self, state, /):
        return state, state


class RunningMean(Calibration):
    def init(self, prior):
        return prior, prior, 0.0

    def update(self, state, /, observed):
        prior, calibrated, num_data = state

        zero_data = jnp.zeros_like(statespace.random.mean(observed))
        mahalanobis_norm = statespace.random.mahalanobis_norm(zero_data, observed)
        new_term = mahalanobis_norm / jnp.sqrt(zero_data.size)

        calibrated = statespace.ssm_util.update_mean(calibrated, new_term, num=num_data)
        return prior, calibrated, num_data + 1.0

    def extract(self, state, /):
        prior, calibrated, _num_data = state
        return prior, calibrated


class CalibrationFactory:
    """Calibration factory.

    Calibration implementations are tied to state-space model factorisations,
    but at the time of choosing the factorisation, it is too early to choose a method.
    This factory allows delaying this decision to later.
    """

    def running_mean(self):
        return RunningMean()

    def most_recent(self):
        return MostRecent()


# Register objects as (empty) pytrees. todo: temporary?!
jax.tree_util.register_pytree_node(
    nodetype=RunningMean,
    flatten_func=lambda _: ((), ()),
    unflatten_func=lambda *a: RunningMean(),
)
jax.tree_util.register_pytree_node(
    nodetype=MostRecent,
    flatten_func=lambda _: ((), ()),
    unflatten_func=lambda *a: MostRecent(),
)
