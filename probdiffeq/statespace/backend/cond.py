import abc


class ConditionalBackEnd(abc.ABC):
    @abc.abstractmethod
    def marginalise_transformation(self, rv, transformation, /):
        raise NotImplementedError

    @abc.abstractmethod
    def revert_transformation(self, rv, transformation, /):
        raise NotImplementedError
