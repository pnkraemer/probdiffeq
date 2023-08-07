import abc


class LinearisationBackend(abc.ABC):
    @abc.abstractmethod
    def constraint_0th(self, ode_order):
        raise NotImplementedError

    @abc.abstractmethod
    def constraint_1st(self, ode_order):
        raise NotImplementedError

    @abc.abstractmethod
    def constraint_statistical_1st(self, cubature_fun):  # ode_order > 1 not supported
        raise NotImplementedError

    @abc.abstractmethod
    def constraint_statistical_0th(self, cubature_fun):  # ode_order > 1 not supported
        raise NotImplementedError
