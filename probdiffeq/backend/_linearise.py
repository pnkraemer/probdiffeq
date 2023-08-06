import abc


class LineariseODEBackEnd(abc.ABC):
    @abc.abstractmethod
    def constraint_0th(self, ode_order):
        raise NotImplementedError

    @abc.abstractmethod
    def constraint_1st(self, ode_order):
        raise NotImplementedError
