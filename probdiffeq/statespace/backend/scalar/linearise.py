from probdiffeq.statespace.backend import linearise


class LineariseODE(linearise.LineariseODE):
    def constraint_0th(self, ode_order):
        def linearise_fun_wrapped(fun, mean):
            fx = linearise.ts0(fun, mean[:ode_order])
            return lambda s: s[ode_order], -fx

        return linearise_fun_wrapped
