"""Control-flow wrappers."""

import jax

_jax_scan = jax.lax.scan
_jax_while_loop = jax.lax.while_loop


def overwrite_scan_func(func, /) -> None:
    """Overwrite the scan() function.

    Parameters
    ----------
    func:
        A function with the same signature as `jax.lax.scan`.
    """
    global _jax_scan
    _jax_scan = func


def scan(step_func, /, init, xs, *, reverse=False, length=None):
    return _jax_scan(step_func, init=init, xs=xs, reverse=reverse, length=length)


def overwrite_while_loop_func(func, /) -> None:
    """Overwrite the while_loop() function.

    Parameters
    ----------
    func:
        A function with the same signature as `jax.lax.while_loop`.
    """
    global _jax_while_loop
    _jax_while_loop = func


def while_loop(cond_func, body_func, /, init):
    return _jax_while_loop(cond_fun=cond_func, body_fun=body_func, init_val=init)


def cond(use_true_func, true_func, false_func, *operands):
    return jax.lax.cond(use_true_func, true_func, false_func, *operands)
