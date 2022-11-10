"""State-space model variable API.

Essentially, the variables are random variables with a specific structure.
"""

import abc

# todo: make "u" a property?
# todo: move extract_sol functions here?


class StateSpaceVariable(abc.ABC):
    @abc.abstractmethod
    def logpdf(self, u, /):
        raise NotImplementedError

    @abc.abstractmethod
    def norm_of_whitened_residual_sqrtm(self):
        raise NotImplementedError

    @abc.abstractmethod
    def condition_on_qoi_observation(self, u, /, *, observation_std):
        raise NotImplementedError

    @abc.abstractmethod
    def extract_qoi(self):
        raise NotImplementedError

    @abc.abstractmethod
    def extract_qoi_from_sample(self, u, /):
        raise NotImplementedError

    @abc.abstractmethod
    def scale_covariance(self, *, scale_sqrtm):
        raise NotImplementedError

    @abc.abstractmethod
    def transform_unit_sample(self, x, /):
        raise NotImplementedError

    @abc.abstractmethod
    def Ax_plus_y(self, *, A, x, y):
        raise NotImplementedError
