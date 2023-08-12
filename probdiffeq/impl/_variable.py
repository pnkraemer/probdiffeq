import abc


class VariableBackend(abc.ABC):
    @abc.abstractmethod
    def variable(self, mean, cholesky):
        raise NotImplementedError

    @abc.abstractmethod
    def to_multivariate_normal(self, u, rv):
        raise NotImplementedError

    @abc.abstractmethod
    def rescale_cholesky(self, rv, factor, /):
        raise NotImplementedError

    @abc.abstractmethod
    def transform_unit_sample(self, unit_sample, /, rv):
        raise NotImplementedError
