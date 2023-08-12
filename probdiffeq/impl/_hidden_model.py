import abc


class HiddenModelBackend(abc.ABC):
    @abc.abstractmethod
    def qoi(self, rv):
        raise NotImplementedError

    @abc.abstractmethod
    def marginal_nth_derivative(self, rv):
        raise NotImplementedError

    @abc.abstractmethod
    def qoi_from_sample(self, sample, /):
        raise NotImplementedError
