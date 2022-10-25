"""Interface for implementations."""

import abc
from dataclasses import dataclass


@dataclass(frozen=True)
class Implementation(abc.ABC):
    """Implementation interface."""

    @abc.abstractmethod
    def init_corrected(self, *, taylor_coefficients):
        raise NotImplementedError

    @abc.abstractmethod
    def init_output_scale_sqrtm(self):
        raise NotImplementedError

    @abc.abstractmethod
    def init_error_estimate(self):
        raise NotImplementedError

    @abc.abstractmethod
    def assemble_preconditioner(self, *, dt):
        raise NotImplementedError

    @abc.abstractmethod
    def extrapolate_mean(self, m0, /, *, p, p_inv):
        raise NotImplementedError

    @abc.abstractmethod
    def estimate_error(self, *, info_op, cache_obs, m_obs, p):
        raise NotImplementedError

    @abc.abstractmethod
    def complete_extrapolation(self, *, m_ext, l0, p_inv, p, output_scale_sqrtm):
        raise NotImplementedError

    @abc.abstractmethod
    def revert_markov_kernel(
        self, *, m_ext, l0, p, p_inv, output_scale_sqrtm, m0_p, m_ext_p
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def final_correction(self, *, info_op, extrapolated, cache_obs, m_obs):
        raise NotImplementedError

    @abc.abstractmethod
    def extract_sol(self, *, rv):
        raise NotImplementedError

    @abc.abstractmethod
    def condense_backward_models(self, *, bw_init, bw_state):  # noqa: D102
        raise NotImplementedError

    @abc.abstractmethod
    def init_preconditioner(self):
        raise NotImplementedError

    @abc.abstractmethod
    def init_backward_transition(self):
        raise NotImplementedError

    @abc.abstractmethod
    def init_backward_noise(self, *, rv_proto):
        raise NotImplementedError

    @abc.abstractmethod
    def evidence_sqrtm(self, *, observed):
        raise NotImplementedError

    @abc.abstractmethod
    def scale_covariance(self, *, rv, scale_sqrtm):
        raise NotImplementedError

    @abc.abstractmethod
    def marginalise_backwards(self, *, init, linop, noise):
        raise NotImplementedError

    @abc.abstractmethod
    def marginalise_model(self, *, init, linop, noise):
        raise NotImplementedError

    @abc.abstractmethod
    def sample_backwards(self, init, linop, noise, base_samples):
        raise NotImplementedError

    # todo: make the extract_*_from_* functions use this one?
    @abc.abstractmethod
    def extract_mean_from_marginals(self, mean):
        raise NotImplementedError
