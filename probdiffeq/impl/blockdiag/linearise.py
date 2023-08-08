"""Linearisation."""
from probdiffeq.impl import _linearise, matfree


class LinearisationBackend(_linearise.LinearisationBackend):
    def constraint_0th(self, ode_order):
        def linearise_fun_wrapped(fun, mean):
            m0 = mean[:, :ode_order]
            fx = ts0(fun, m0.T)

            def a1(s):
                return s[:, [ode_order], ...]

            return matfree.parametrised_linop(lambda v, _p: a1(v)), -fx[:, None]

        return linearise_fun_wrapped

    def constraint_1st(self, ode_order):
        raise NotImplementedError

    def constraint_statistical_0th(self, cubature_fun):
        raise NotImplementedError

    def constraint_statistical_1st(self, cubature_fun):
        raise NotImplementedError


def ts0(fn, m):
    return fn(m)
