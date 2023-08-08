"""Calibration API."""


import abc

import jax

from probdiffeq.impl import impl


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
        calibrated = impl.random.mahalanobis_norm_relative(0.0, observed)
        return calibrated

    def extract(self, state, /):
        return state, state


class RunningMean(Calibration):
    def init(self, prior):
        return prior, prior, 0.0

    def update(self, state, /, observed):
        prior, calibrated, num_data = state

        new_term = impl.random.mahalanobis_norm_relative(0.0, observed)
        calibrated = impl.ssm_util.update_mean(calibrated, new_term, num=num_data)
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
