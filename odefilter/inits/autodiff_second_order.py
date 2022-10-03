"""Exact initialisation."""
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental.jet import jet


@partial(jax.jit, static_argnames=("f", "num_derivatives"))
def taylormode(*, f, u0, du0, num_derivatives):

    if num_derivatives == 0:
        return u0.reshape((1, -1))

    if num_derivatives == 1:
        return jnp.stack((u0, du0), axis=0)

    f0 = f(u0, du0)
    u_primals, u_series = u0, (du0, f0)

    @partial(jax.jit, static_argnames=("fun",))
    def next_ode_derivative(fun, primals, series):
        primals_new, series_new = jet(
            fun=fun,
            primals=(primals, series[0]),
            series=(series[:-1], series[1:]),
        )
        return series + (series_new[-1],)

    # Not a scan, because I dont know how to do it.
    # I have a strong feeling that it is not possible.
    # But since the number of derivatives <= 12, it should be okay.
    for _ in range(num_derivatives - 2):
        u_series = next_ode_derivative(fun=f, primals=u_primals, series=u_series)

    return jnp.stack((u_primals,) + u_series)


@partial(jax.jit, static_argnames=("f", "num_derivatives"))
def forwardmode(*, f, u0, du0, num_derivatives):

    if num_derivatives == 0:
        return u0.reshape((1, -1))

    if num_derivatives == 1:
        return jnp.stack((u0, du0), axis=0)

    def next_ode_derivative(fun, fun0):
        def dg(x, dx):
            term1 = jax.jacfwd(fun, argnums=0)(x, dx) @ dx
            term2 = jax.jacfwd(fun, argnums=1)(x, dx) @ fun0(x, dx)
            return term1 + term2

        return dg

    g, g0 = f, f
    du0_all = [u0, du0, g(u0, du0)]
    for _ in range(num_derivatives - 2):
        g = next_ode_derivative(fun=g, fun0=g0)
        du0_all.append(g(u0, du0))

    return jnp.stack(du0_all)


@partial(jax.jit, static_argnames=("f", "num_derivatives"))
def reversemode(*, f, u0, du0, num_derivatives):

    if num_derivatives == 0:
        return u0.reshape((1, -1))

    if num_derivatives == 1:
        return jnp.stack((u0, du0), axis=0)

    def next_ode_derivative(fun, fun0):
        def dg(x, dx):
            term1 = jax.jacrev(fun, argnums=0)(x, dx) @ dx
            term2 = jax.jacrev(fun, argnums=1)(x, dx) @ fun0(x, dx)
            return term1 + term2

        return dg

    g, g0 = f, f
    du0_all = [u0, du0, g(u0, du0)]
    for _ in range(num_derivatives - 2):
        g = next_ode_derivative(fun=g, fun0=g0)
        du0_all.append(g(u0, du0))

    return jnp.stack(du0_all)


@partial(jax.jit, static_argnames=("f", "num_derivatives"))
def forwardmode_jvp(*, f, u0, du0, num_derivatives):

    if num_derivatives == 0:
        return u0.reshape((1, -1))

    if num_derivatives == 1:
        return jnp.stack((u0, du0), axis=0)

    def next_ode_derivative(fun, fun0):
        def dg(x, dx):
            _, y = jax.jvp(fun, (x, dx), (dx, fun0(x, dx)))
            return y

        return dg

    g, g0 = f, f
    du0_all = [u0, du0, g(u0, du0)]
    for _ in range(num_derivatives - 2):
        g = next_ode_derivative(fun=g, fun0=g0)
        du0_all.append(g(u0, du0))

    return jnp.stack(du0_all)
