import abc


class RandomVariableBackEnd(abc.ABC):
    @abc.abstractmethod
    def qoi_like(self):
        raise NotImplementedError
