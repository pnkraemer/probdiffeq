"""Interface for implementations."""

import abc
from dataclasses import dataclass

import jax.numpy as jnp

from odefilter.implementations import _sqrtm


@dataclass(frozen=True)
class Implementation(abc.ABC):
    """Implementation interface."""

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

    @abc.abstractmethod
    def evidence_sqrtm(self, *, observed):
        raise NotImplementedError

    @abc.abstractmethod
    def scale_covariance(self, *, rv, scale_sqrtm):
        raise NotImplementedError

    @staticmethod
    def sum_sqrt_scalars(a, b):
        R = jnp.asarray([[a], [b]])
        diffsqrtm = _sqrtm.sqrtm_to_upper_triangular(R=R).T
        return jnp.reshape(diffsqrtm, ())
