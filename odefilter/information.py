"""Information functions."""
import functools

import jax
import jax.numpy as jnp
import jax.tree_util


@functools.lru_cache(maxsize=None)
def isotropic_ek0(*, ode_order=1):
    """EK0-linearise an ODE assuming a linearisation-point with\
     isotropic Kronecker structure."""

    @functools.lru_cache(maxsize=None)
    def create_ek0_info_op_linearised(f):
        """Create a "linearize()" implementation according to what\
         the EK0 does to the ODE residual."""

        @jax.jit
        def jvp(x, *, t, p):
            return x[ode_order]

        def info_op(x, *, t, p):
            bias = x[ode_order, ...] - f(*x[:ode_order, ...], t=t, p=p)
            return bias, jax.tree_util.Partial(jvp, t=t, p=p)

        return jax.tree_util.Partial(info_op)

    return create_ek0_info_op_linearised


@functools.lru_cache(maxsize=None)
def ek1(*, ode_dimension, ode_order=1):
    """EK1 information."""

    @functools.lru_cache(maxsize=None)
    def create_ek1_info_op_linearised(f):
        @jax.jit
        def residual(x, *, t, p):
            x_reshaped = jnp.reshape(x, (-1, ode_dimension), order="F")
            return x_reshaped[ode_order, ...] - f(
                *x_reshaped[:ode_order, ...], t=t, p=p
            )

        def info_op(x, *, t, p):
            return jax.linearize(jax.tree_util.Partial(residual, t=t, p=p), x)

        return jax.tree_util.Partial(info_op)

    return create_ek1_info_op_linearised
