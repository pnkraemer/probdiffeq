from probdiffeq.backend import abc


class VariableBackend(abc.ABC):
    pass


class ScalarVariable(VariableBackend):
    pass


class DenseVariable(VariableBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape


class IsotropicVariable(VariableBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape


class BlockDiagVariable(VariableBackend):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape
