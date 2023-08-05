import abc


class ConditionalImpl(abc.ABC):
    @abc.abstractmethod
    def marginalise(self, rv, conditional, /):
        raise NotImplementedError

    @abc.abstractmethod
    def revert(self, rv, conditional, /):
        raise NotImplementedError

    @abc.abstractmethod
    def apply(self, x, conditional, /):
        raise NotImplementedError

    @abc.abstractmethod
    def merge(self, cond1, cond2, /):
        raise NotImplementedError


class ConditionalBackEnd(abc.ABC):
    @property
    @abc.abstractmethod
    def conditional(self) -> ConditionalImpl:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def transform(self) -> ConditionalImpl:
        raise NotImplementedError
