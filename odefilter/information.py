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


import equinox as eqx
import jax
import jax.numpy as jnp


def isotropic_ek0_first_order():
    """EK0-Linearize an ODE assuming a linearisation-point with\
     isotropic Kronecker structure."""
    op = _IsotropicEK0FIrstOrder()

    def fn(f, /):
        return _curry_info_op(op, f)

    return fn


def ek1_first_order(**kwargs):
    """EK1 information."""
    op = _EK1FirstOrder(**kwargs)

    def fn(f, /):
        return _curry_info_op(op, f)

    return fn


def _curry_info_op(
    info_op,
    vector_field,
):
    def info_op_curried(t, *ys_and_ps):
        def vf(*xs_and_ps):
            return vector_field(t, *xs_and_ps)

        return info_op(vf, *ys_and_ps)

    return info_op_curried


class _IsotropicEK0FIrstOrder(eqx.Module):
    def __call__(self, f, x, *p):
        def approx_residual(u):
            return u[1] - f(x[0], *p)

        bias = approx_residual(x)

        def jvp(y):
            return y[1]

        return bias, jvp


class _EK1FirstOrder(eqx.Module):

    # static, because it affects the behaviour of the residual fn
    ode_dimension: int = eqx.static_field()

    def __call__(self, f, x, *p):
        def residual(u):
            u_reshaped = jnp.reshape(u, (-1, self.ode_dimension), order="F")
            return u_reshaped[1] - f(u_reshaped[0], *p)

        bias, jvp = jax.linearize(residual, x)
        return bias, jvp
