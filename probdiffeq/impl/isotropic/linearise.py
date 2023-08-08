from probdiffeq.impl import _linearise, matfree


class LinearisationBackend(_linearise.LinearisationBackend):
    def constraint_1st(self, ode_order):
        raise NotImplementedError

    def constraint_0th(self, ode_order):
        def linearise_fun_wrapped(fun, mean):
            fx = ts0(fun, mean[:ode_order, ...])
            linop = matfree.parametrised_linop(lambda s, _p: s[ode_order, ...])
            return linop, -fx

        return linearise_fun_wrapped

    def constraint_statistical_0th(self, cubature_fun):
        raise NotImplementedError

    def constraint_statistical_1st(self, cubature_fun):
        raise NotImplementedError


def ts0(fn, m):
    return fn(m)
