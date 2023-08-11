"""Control-flow wrappers."""

import jax


def while_loop(cond_func, body_func, /, init):
    return jax.lax.while_loop(cond_fun=cond_func, body_fun=body_func, init_val=init)
