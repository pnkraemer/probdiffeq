"""Function transformation tools."""
import functools

import jax


def vmap(func, /, in_axes=0, out_axes=0):
    return jax.vmap(func, in_axes=in_axes, out_axes=out_axes)


def partial(func, *args, **kwargs):
    return functools.partial(func, *args, **kwargs)


def jit(func, /, static_argnums=None, static_argnames=None):
    return jax.jit(func, static_argnums=static_argnums, static_argnames=static_argnames)
