import abc


class TransformBackEnd(abc.ABC):
    @abc.abstractmethod
    def marginalise(self, rv, transformation, /):
        raise NotImplementedError

    @abc.abstractmethod
    def revert(self, rv, transformation, /):
        raise NotImplementedError
