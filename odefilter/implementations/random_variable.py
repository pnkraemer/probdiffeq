"""Random variable API."""

import abc

# todo: make "u" a property?
# todo: move evidence_sqrtm here and rename accordingly?
# todo: make self.whiten() property?
# todo: move correct_sol_observation here?


class RandomVariable(abc.ABC):
    @abc.abstractmethod
    def logpdf(self, u, /):
        raise NotImplementedError

    @abc.abstractmethod
    def norm_of_whitened_residual_sqrtm(self):
        raise NotImplementedError
