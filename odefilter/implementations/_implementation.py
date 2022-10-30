"""Interface for implementations."""

import abc
from dataclasses import dataclass
from typing import Generic, Tuple, TypeVar

from jax import Array

R = TypeVar("R")  # think: random variables
C = TypeVar("C")  # think: my personal cache


@dataclass(frozen=True)
class Implementation(abc.ABC, Generic[R, C]):
    """Implementation interface."""

    @abc.abstractmethod
    def init_corrected(self, *, taylor_coefficients) -> R:
        raise NotImplementedError

    @abc.abstractmethod
    def init_output_scale_sqrtm(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def init_error_estimate(self) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def begin_extrapolation(self, m0, /, *, dt) -> Tuple[R, C]:
        raise NotImplementedError

    @abc.abstractmethod
    def complete_extrapolation(
        self, *, linearisation_pt: R, l0, cache: C, output_scale_sqrtm
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def revert_markov_kernel(
        self, *, linearisation_pt: R, l0, cache: C, output_scale_sqrtm
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def extract_sol(self, *, rv: R):
        raise NotImplementedError

    @abc.abstractmethod
    def condense_backward_models(self, *, bw_init, bw_state):
        raise NotImplementedError

    @abc.abstractmethod
    def init_backward_transition(self):
        raise NotImplementedError

    @abc.abstractmethod
    def init_backward_noise(self, *, rv_proto):
        raise NotImplementedError

    @abc.abstractmethod
    def scale_covariance(self, *, rv: R, scale_sqrtm):
        raise NotImplementedError

    @abc.abstractmethod
    def marginalise_backwards(self, *, init: R, linop, noise: R):
        raise NotImplementedError

    @abc.abstractmethod
    def marginalise_model(self, *, init: R, linop, noise: R):
        raise NotImplementedError

    @abc.abstractmethod
    def sample_backwards(self, init: R, linop, noise: R, base_samples):
        raise NotImplementedError

    # todo: make the extract_*_from_* functions use this one?
    @abc.abstractmethod
    def extract_mean_from_marginals(self, mean):
        raise NotImplementedError
