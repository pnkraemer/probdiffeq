from probdiffeq.backend import abc
from probdiffeq.backend import numpy as np
from probdiffeq.impl import _normal


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


class DensePrototype(PrototypeBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def qoi(self):
        return np.empty(self.ode_shape)

    def observed(self):
        mean = np.empty(self.ode_shape)
        cholesky = np.empty(self.ode_shape + self.ode_shape)
        return _normal.Normal(mean, cholesky)

    def error_estimate(self):
        return np.empty(self.ode_shape)

    def output_scale(self):
        return np.empty(())


class IsotropicPrototype(PrototypeBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def qoi(self):
        return np.empty(self.ode_shape)

    def observed(self):
        mean = np.empty((1, *self.ode_shape))
        cholesky = np.empty(())
        return _normal.Normal(mean, cholesky)

    def error_estimate(self):
        return np.empty(())

    def output_scale(self):
        return np.empty(())


class BlockDiagPrototype(PrototypeBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def qoi(self):
        return np.empty(self.ode_shape)

    def observed(self):
        mean = np.empty((*self.ode_shape, 1))
        cholesky = np.empty((*self.ode_shape, 1, 1))
        return _normal.Normal(mean, cholesky)

    def error_estimate(self):
        return np.empty(self.ode_shape)

    def output_scale(self):
        return np.empty(self.ode_shape)
