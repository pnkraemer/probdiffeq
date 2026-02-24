from probdiffeq.backend import abc, np


class PrototypeBackend(abc.ABC):
    @abc.abstractmethod
    def std(self):
        raise NotImplementedError

    @abc.abstractmethod
    def output_scale_calibrated(self):
        raise NotImplementedError


class DensePrototype(PrototypeBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def std(self):
        return np.ones(self.ode_shape)

    def output_scale_calibrated(self):
        return np.ones(())


class IsotropicPrototype(PrototypeBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def std(self):
        return np.ones(())

    def output_scale_calibrated(self):
        return np.ones(())


class BlockDiagPrototype(PrototypeBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def std(self):
        return np.ones(self.ode_shape)

    def output_scale_calibrated(self):
        return np.ones(self.ode_shape)
