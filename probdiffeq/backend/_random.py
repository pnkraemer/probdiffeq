import abc


class RandomVariableBackEnd(abc.ABC):
    @abc.abstractmethod
    def qoi_like(self):
        raise NotImplementedError

    @abc.abstractmethod
    def mahalanobis_norm(self, u, /, rv):
        raise NotImplementedError

    @abc.abstractmethod
    def standard_deviation(self, rv):
        raise NotImplementedError

    @abc.abstractmethod
    def mean(self, rv):
        raise NotImplementedError

    @abc.abstractmethod
    def qoi(self, rv):
        raise NotImplementedError

    @abc.abstractmethod
    def rescale_cholesky(self, rv, factor, /):
        raise NotImplementedError
