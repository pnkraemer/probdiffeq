"""Control-flow wrappers."""

import contextlib

import jax

_jax_scan = jax.lax.scan
_jax_while_loop = jax.lax.while_loop


@contextlib.contextmanager
def overwrite_scan_func(func, /):
    """Overwrite the scan() function.

    Parameters
    ----------
    func:
        A function with the same signature as `jax.lax.scan`.
    """
    global _jax_scan
    tmp = _jax_scan

    try:
        _jax_scan = func
        yield _jax_scan
    finally:
        _jax_scan = tmp


def scan(step_func, /, init, xs, *, reverse=False, length=None):
    return _jax_scan(step_func, init=init, xs=xs, reverse=reverse, length=length)


@contextlib.contextmanager
def overwrite_while_loop_func(func, /):
    """Overwrite the while_loop() function.

    Parameters
    ----------
    func:
        A function with the same signature as `jax.lax.while_loop`.
    """
    global _jax_while_loop
    tmp = _jax_while_loop

    try:
        _jax_while_loop = func
        yield _jax_while_loop
    finally:
        _jax_while_loop = tmp


def while_loop(cond_func, body_func, /, init):
    return _jax_while_loop(cond_fun=cond_func, body_fun=body_func, init_val=init)


def cond(use_true_func, true_func, false_func, *operands):
    return jax.lax.cond(use_true_func, true_func, false_func, *operands)
