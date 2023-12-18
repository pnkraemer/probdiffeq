from probdiffeq.backend import abc


class TransformBackend(abc.ABC):
    @abc.abstractmethod
    def marginalise(self, rv, transformation, /):
        raise NotImplementedError

    @abc.abstractmethod
    def revert(self, rv, transformation, /):
        raise NotImplementedError
