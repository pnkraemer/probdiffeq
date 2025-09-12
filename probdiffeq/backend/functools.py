"""Function transformation tools."""

import functools

import jax
import jax.experimental.jet


def vmap(func, /, in_axes=0, out_axes=0):
    return jax.vmap(func, in_axes=in_axes, out_axes=out_axes)


def partial(func, *args, **kwargs):
    return functools.partial(func, *args, **kwargs)


def jit(func, /, static_argnums=None, static_argnames=None):
    return jax.jit(func, static_argnums=static_argnums, static_argnames=static_argnames)


def jet(func, /, primals, series, *, is_tcoeff=False):
    return jax.experimental.jet.jet(
        func, primals=primals, series=series, factorial_scaled=not is_tcoeff
    )


def linearize(func, *args):
    return jax.linearize(func, *args)


def jvp(func, /, primals, tangents):
    return jax.jvp(func, primals, tangents)


def jacrev(func):
    return jax.jacrev(func)


def grad(func):
    return jax.grad(func)
