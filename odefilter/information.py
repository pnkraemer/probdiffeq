r"""Information function(al)s.

An information function(al) for ordinary differential equations
measures the _local defect_, or _residual_ of the differential equation:

* $\dot u = f(u):$ $F(u, \dot u, ...) = \dot u - f(u)$
* $\ddot u = f(u, \dot u):$  $F(u, \dot u, ...) = \ddot u - f(u, \dot u)$
* $M \ddot u = f(u):$ $F(u, \dot u, ...) = M\ddot u - f(u)$
* ...

ODE filters rely on such information function:
essentially, the idea of probabilistic numerical solvers is to condition
some prior distribution on the event $F(u, \dot u, ...) = 0$ (using a finite grid).

This module implements such information operators and approximations thereof.
In particular, most functions in this module compute _linearized_ information functions,
using one of the many approximation methods (EK0, EK1, UK1, ...).
The common signature here is

  bias, linearfn = <information_fn>(ode_function, $x_0$)

where the _bias_ and the _linearfn_ mirror what `jax.linearize` returns:
A fixed bias point and a push-forward- operator.
In fact, most of the `EK`-like functions are implemented with `jax.linearize`.

In the end, this approximates $F(u) = \mathrm{bias} + \mathrm{linearfn}(u)$
and we can use it in ODE solvers.
"""


import jax
import jax.numpy as jnp


def isotropic_ek0(*, ode_order=1):
    """EK0-linearise an ODE assuming a linearisation-point with\
     isotropic Kronecker structure."""

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


def ek1(*, ode_dimension, ode_order=1):
    """EK1 information."""

    def create_ek1_info_op_linearised(f):
        def residual(t, x, *p):
            x_reshaped = jnp.reshape(x, (-1, ode_dimension), order="F")
            return x_reshaped[ode_order, ...] - f(t, *x_reshaped[:ode_order, ...], *p)

        def info_op(t, x, *p):
            return jax.linearize(lambda s: residual(t, s, *p), x)

        return info_op

    return create_ek1_info_op_linearised
