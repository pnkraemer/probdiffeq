from probdiffeq.backend import abc


class PrototypeBackend(abc.ABC):
    @abc.abstractmethod
    def qoi(self):
        raise NotImplementedError

    @abc.abstractmethod
    def observed(self):
        raise NotImplementedError

    @abc.abstractmethod
    def error_estimate(self):
        raise NotImplementedError

    @abc.abstractmethod
    def output_scale(self):
        raise NotImplementedError
