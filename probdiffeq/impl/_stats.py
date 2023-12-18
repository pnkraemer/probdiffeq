from probdiffeq.backend import abc


class StatsBackend(abc.ABC):
    @abc.abstractmethod
    def mahalanobis_norm_relative(self, u, /, rv):
        raise NotImplementedError

    @abc.abstractmethod
    def logpdf(self, u, /, rv):
        raise NotImplementedError

    @abc.abstractmethod
    def standard_deviation(self, rv):
        raise NotImplementedError

    @abc.abstractmethod
    def mean(self, rv):
        raise NotImplementedError

    @abc.abstractmethod
    def sample_shape(self, rv):
        raise NotImplementedError
