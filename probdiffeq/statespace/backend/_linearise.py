import abc


class LineariseODEBackEnd(abc.ABC):
    @abc.abstractmethod
    def constraint_0th(self, ode_order):
        raise NotImplementedError
