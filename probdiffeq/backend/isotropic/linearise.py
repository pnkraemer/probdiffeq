from probdiffeq.backend import _linearise


class LineariseODEBackEnd(_linearise.LineariseODEBackEnd):
    def constraint_1st(self, ode_order):
        raise NotImplementedError

    def constraint_0th(self, ode_order):
        def linearise_fun_wrapped(fun, mean):
            fx = ts0(fun, mean[:ode_order, ...])
            return lambda s: s[ode_order, ...], -fx

        return linearise_fun_wrapped

    def constraint_statistical_0th(self, cubature_fun):
        raise NotImplementedError

    def constraint_statistical_1st(self, cubature_fun):
        raise NotImplementedError


def ts0(fn, m):
    return fn(m)
