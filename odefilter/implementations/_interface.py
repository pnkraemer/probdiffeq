"""Interface for implementations."""

import abc
from dataclasses import dataclass


@dataclass(frozen=True)
class Implementation(abc.ABC):
    """Implementation interface."""

    @classmethod
    def from_num_derivatives(cls, *, num_derivatives, ode_dimension):
        raise NotImplementedError

    @abc.abstractmethod
    def init_corrected(self, *, taylor_coefficients):
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
    def estimate_error(self, *, linear_fn, m_obs, p):
        raise NotImplementedError

    @abc.abstractmethod
    def complete_extrapolation(self, *, m_ext, l0, p_inv, p, diffusion_sqrtm):
        raise NotImplementedError

    @abc.abstractmethod
    def revert_markov_kernel(
        self, *, m_ext, l0, p, p_inv, diffusion_sqrtm, m0_p, m_ext_p
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def final_correction(self, *, extrapolated, linear_fn, m_obs):
        raise NotImplementedError

    @abc.abstractmethod
    def extract_sol(self, *, rv):
        raise NotImplementedError

    @abc.abstractmethod
    def init_preconditioner(self):
        raise NotImplementedError

    @abc.abstractmethod
    def init_backward_transition(self):
        raise NotImplementedError

    @abc.abstractmethod
    def init_backward_noise(self, rv_proto):
        raise NotImplementedError
