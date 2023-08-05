import abc


class ErrorBackEnd(abc.ABC):
    @abc.abstractmethod
    def estimate(self, observed, /):
        raise NotImplementedError
