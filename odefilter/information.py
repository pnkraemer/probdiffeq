"""Information functions."""
import functools

import jax
import jax.numpy as jnp


@functools.lru_cache(maxsize=None)
def isotropic_ek0(*, ode_order=1):
    """EK0-linearise an ODE assuming a linearisation-point with\
     isotropic Kronecker structure."""

    @functools.lru_cache(maxsize=None)
    def create_ek0_info_op_linearised(f):
        """Create a "linearize()" implementation according to what\
         the EK0 does to the ODE residual."""

        def jvp(t, x, *p):
            return x[ode_order]

        def info_op(t, x, *p):
            bias = x[ode_order, ...] - f(t, *x[:ode_order, ...], *p)
            return bias, lambda s: jvp(t, s, *p)

        return info_op

    return create_ek0_info_op_linearised


@functools.lru_cache(maxsize=None)
def ek1(*, ode_dimension, ode_order=1):
    """EK1 information."""

    @functools.lru_cache(maxsize=None)
    def create_ek1_info_op_linearised(f):
        def residual(t, x, *p):
            x_reshaped = jnp.reshape(x, (-1, ode_dimension), order="F")
            return x_reshaped[ode_order, ...] - f(t, *x_reshaped[:ode_order, ...], *p)

        def info_op(t, x, *p):
            return jax.linearize(lambda s: residual(t, s, *p), x)

        return info_op

    return create_ek1_info_op_linearised
