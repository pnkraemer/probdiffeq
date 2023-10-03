"""Linearisation."""
from probdiffeq.impl import _linearise
from probdiffeq.util import linop_util


class LinearisationBackend(_linearise.LinearisationBackend):
    def ode_taylor_0th(self, ode_order):
        def linearise_fun_wrapped(fun, mean):
            m0 = mean[:, :ode_order]
            fx = ts0(fun, m0.T)

            def a1(s):
                return s[:, [ode_order], ...]

            return linop_util.parametrised_linop(lambda v, _p: a1(v)), -fx[:, None]

        return linearise_fun_wrapped

    def ode_taylor_1st(self, ode_order):
        raise NotImplementedError

    def ode_statistical_0th(self, cubature_fun):
        raise NotImplementedError

    def ode_statistical_1st(self, cubature_fun):
        raise NotImplementedError


def ts0(fn, m):
    return fn(m)
