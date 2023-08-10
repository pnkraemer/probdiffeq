import abc


class RandomVariableBackend(abc.ABC):
    # todo: is this module getting out of hand?
    #  Split into three modules "SSMState, stats, normal"?

    @abc.abstractmethod
    def variable(self, mean, cholesky):
        raise NotImplementedError

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
    def cholesky(self, rv):
        raise NotImplementedError

    @abc.abstractmethod
    def cov_dense(self, rv):
        raise NotImplementedError

    @abc.abstractmethod
    def qoi(self, rv):
        raise NotImplementedError

    @abc.abstractmethod
    def rescale_cholesky(self, rv, factor, /):
        raise NotImplementedError

    @abc.abstractmethod
    def marginal_nth_derivative(self, rv):
        raise NotImplementedError

    @abc.abstractmethod
    def sample_shape(self, rv):
        raise NotImplementedError

    @abc.abstractmethod
    def transform_unit_sample(self, unit_sample, /, rv):
        raise NotImplementedError

    @abc.abstractmethod
    def qoi_from_sample(self, sample, /):
        raise NotImplementedError

    @abc.abstractmethod
    def to_multivariate_normal(self, u, rv):
        raise NotImplementedError
