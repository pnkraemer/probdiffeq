from probdiffeq.backend import _cond
from probdiffeq.backend.dense import random


class TransformImpl(_cond.TransformImpl):
    pass


class ConditionalImpl(_cond.ConditionalImpl):
    def apply(self, x, conditional, /):
        A, noise = conditional
        return random.Normal(A @ x + noise.mean, noise.cholesky)

    def marginalise(self, rv, conditional, /):
        raise NotImplementedError

    def merge(self, cond1, cond2, /):
        raise NotImplementedError

    def revert(self, rv, conditional, /):
        raise NotImplementedError


class ConditionalBackEnd(_cond.ConditionalBackEnd):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    @property
    def transform(self) -> TransformImpl:
        return TransformImpl()

    @property
    def conditional(self) -> ConditionalImpl:
        return ConditionalImpl()
