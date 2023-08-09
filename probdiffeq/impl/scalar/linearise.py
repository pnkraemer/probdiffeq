"""Linearisation."""
from probdiffeq.impl import _linearise


class LinearisationBackend(_linearise.LinearisationBackend):
    def ode_taylor_0th(self, ode_order):
        def linearise_fun_wrapped(fun, mean):
            fx = ts0(fun, mean[:ode_order])
            return lambda s: s[ode_order], -fx

        return linearise_fun_wrapped

    def ode_taylor_1st(self, ode_order):
        raise NotImplementedError

    def ode_statistical_1st(self, cubature_fun):
        raise NotImplementedError

    def ode_statistical_0th(self, cubature_fun):
        raise NotImplementedError


def ts0(fn, m):
    return fn(m)
