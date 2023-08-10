"""Calibration functionality."""


import abc

import jax

from probdiffeq.impl import impl


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
        return impl.random.mahalanobis_norm_relative(0.0, observed)

    def extract(self, state, /):
        return state, state


# todo: if we pass the mahalanobis_relative term to the update() function,
#  it reduces to a generic stats() module that can also be used for e.g.
#  marginal likelihoods. In this case, the MostRecent() stuff becomes void.
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


def _unflatten_func(nodetype):
    return lambda *_a: nodetype()


# Register objects as (empty) pytrees. todo: temporary?!
for node in [RunningMean, MostRecent]:
    jax.tree_util.register_pytree_node(
        nodetype=node,
        flatten_func=lambda _: ((), ()),
        unflatten_func=_unflatten_func(node),
    )


def output_scale() -> CalibrationFactory:
    """Construct (a buffet of) calibration strategies."""
    return CalibrationFactory()
