import functools

import jax
import jax.numpy as jnp

from probdiffeq.backend import _linearise


class LineariseODEBackEnd(_linearise.LineariseODEBackEnd):
    def __init__(self, ode_shape):
        self.ode_shape = ode_shape

    def constraint_0th(self, ode_order):
        def linearise_fun_wrapped(fun, mean):
            a0 = functools.partial(self._select_dy, idx_or_slice=slice(0, ode_order))
            a1 = functools.partial(self._select_dy, idx_or_slice=ode_order)

            if jnp.shape(a0(mean)) != (expected_shape := (ode_order,) + self.ode_shape):
                raise ValueError(f"{jnp.shape(a0(mean))} != {expected_shape}")

            fx = ts0(fun, a0(mean))
            return _autobatch_linop(a1), -fx

        return linearise_fun_wrapped

    def constraint_1st(self, ode_order):
        def new(fun, mean, /):
            a0 = functools.partial(self._select_dy, idx_or_slice=slice(0, ode_order))
            a1 = functools.partial(self._select_dy, idx_or_slice=ode_order)

            if jnp.shape(a0(mean)) != (expected_shape := (ode_order,) + self.ode_shape):
                raise ValueError(f"{jnp.shape(a0(mean))} != {expected_shape}")

            jvp, fx = ts1(fun, a0(mean))

            def A(x):
                x1 = a1(x)
                x0 = a0(x)
                return x1 - jvp(x0)

            return _autobatch_linop(A), -fx

        return new

    def _select_dy(self, x, idx_or_slice):
        (d,) = self.ode_shape
        x_reshaped = jnp.reshape(x, (-1, d), order="F")
        return x_reshaped[idx_or_slice, ...]


def _autobatch_linop(fun):
    def fun_(x):
        if jnp.ndim(x) > 1:
            return jax.vmap(fun_, in_axes=1, out_axes=1)(x)
        return fun(x)

    return fun_


def ts0(fn, m):
    return fn(m)


def ts1(fn, m):
    b, jvp = jax.linearize(fn, m)
    return jvp, b - jvp(m)
