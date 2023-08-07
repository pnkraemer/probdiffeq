import abc


class RandomVariableBackend(abc.ABC):
    @abc.abstractmethod
    def variable(self, mean, cholesky):
        raise NotImplementedError

    @abc.abstractmethod
    def qoi_like(self):  # todo: move to ssm_util
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
