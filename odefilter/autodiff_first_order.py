"""Exact initialisation."""
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental.jet import jet


def taylormode(*, f, u0, num_derivatives):
    """Compute the initial derivatives with Taylor-mode AD."""
    if num_derivatives == 0:
        return u0.reshape((1, -1))

    f0 = f(u0)
    u_primals, u_series = u0, (f0,)

    # Not a scan, because I dont know how to do it.
    # But since the number of derivatives <= 12, it should be okay.
    for _ in range(num_derivatives - 1):
        u_series = _taylormode_next_ode_derivative(
            fun=f, primals=u_primals, series=u_series
        )
    return jnp.stack((u_primals,) + u_series)


def _taylormode_next_ode_derivative(fun, primals, series):
    p, s = jet(fun, primals=(primals,), series=(series,))
    return p, *s


def forwardmode_jvp(*, f, u0, num_derivatives):
    """Compute the initial derivatives with forward-mode AD (JVPs)."""

    if num_derivatives == 0:
        return u0.reshape((1, -1))

    g, g0 = f, f
    du0 = [u0, g(u0)]
    for _ in range(num_derivatives - 1):
        g = _forwardmode_jvp_next_ode_derivative(fun=g, fun0=g0)
        du0.append(g(u0))

    return jnp.stack(du0)


def _forwardmode_jvp_next_ode_derivative(fun, fun0):
    def dg(x):
        _, y = jax.jvp(fun, (x,), (fun0(x),))
        return y

    return dg


def forwardmode(*, f, u0, num_derivatives):
    """Compute the initial derivatives with forward-mode AD."""

    if num_derivatives == 0:
        return u0.reshape((1, -1))

    def next_ode_derivative(fun, fun0):
        def dg(x):
            return jax.jacfwd(fun)(x) @ fun0(x)

        return dg

    g, g0 = f, f
    du0 = [u0, g(u0)]
    for _ in range(num_derivatives - 1):
        g = next_ode_derivative(fun=g, fun0=g0)
        du0.append(g(u0))

    return jnp.stack(du0)


def reversemode(*, f, u0, num_derivatives):
    """Compute the initial derivatives with reverse-mode AD."""

    if num_derivatives == 0:
        return u0.reshape((1, -1))

    def next_ode_derivative(fun, fun0):
        def dg(x):
            return jax.jacrev(fun)(x) @ fun0(x)

        return dg

    g, g0 = f, f
    du0 = [u0, g(u0)]
    for _ in range(num_derivatives - 1):
        g = next_ode_derivative(fun=g, fun0=g0)
        du0.append(g(u0))

    return jnp.stack(du0)
