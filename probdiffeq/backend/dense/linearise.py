from probdiffeq.backend import _linearise


class LineariseODEBackEnd(_linearise.LineariseODEBackEnd):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def constraint_0th(self, ode_order):
        def linearise_fun_wrapped(fun, mean):
            select = functools.partial(_select_derivative, ode_shape=self.ode_shape)
            a0 = functools.partial(select, i=slice(0, ode_order))
            a1 = functools.partial(select, i=ode_order)

            if jnp.shape(a0(mean)) != (expected_shape := (ode_order,) + self.ode_shape):
                raise ValueError(f"{jnp.shape(a0(mean))} != {expected_shape}")

            fx = linearise.ts0(fun, a0(mean))
            return _autobatch_linop(a1), -fx

        return linearise_fun_wrapped
