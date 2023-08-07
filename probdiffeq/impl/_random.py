import abc


class RandomVariableBackend(abc.ABC):
    # todo: is this module getting out of hand?
    #  Split into three modules "SSMState, stats, normal"?

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
