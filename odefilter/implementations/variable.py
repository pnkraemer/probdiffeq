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

    # todo: make this a property?
    @abc.abstractmethod
    def extract_qoi(self):
        raise NotImplementedError
