"""Function transformation tools."""
import functools

import jax


def vmap(func, /, in_axes=0, out_axes=0):
    return jax.vmap(func, in_axes=in_axes, out_axes=out_axes)


def partial(func, *args, **kwargs):
    return functools.partial(func, *args, **kwargs)
