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

from typing import Callable, Tuple

import equinox as eqx
from jaxtyping import Array, Float


class IsotropicEK0(eqx.Module):
    """EK0-Linearize an ODE assuming a linearisation-point with\
     isotropic Kronecker structure.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>>
    >>> def f(x):
    ...     return x*(1-x)
    >>>
    >>> x0 = 0.5 * jnp.ones((3, 1))
    >>> linearise = IsotropicEK0(ode_order=1)
    >>> b, fn = linearise(f, x0)
    >>> assert jnp.allclose(b, x0[1] - f(x0[0]))
    >>>
    >>> print(x0)
    [[0.5]
     [0.5]
     [0.5]]
    >>> print(x0[1])
    [0.5]
    >>> print(fn(x0))
    [0.5]
    """

    ode_order: int

    def __call__(
        self,
        f: Callable[[Float[Array, "n d"]], Float[Array, " d"]],
        x: Float[Array, "n d"],
    ) -> Tuple[Float[Array, " d"], Callable[[Float[Array, "n d"]], Float[Array, " d"]]]:
        """Linearise the ODE.

        Parameters
        ----------
        f :
            Vector field of a first-order ODE.
        x :
            Linearisation point.

        Returns
        -------
        :
            Output bias.
        :
            Linear function (pushforward of the tangent spaces).

        """
        if self.ode_order == 1:
            return self._linearise_1st(f=f, x=x)
        if self.ode_order == 2:
            return self._linearise_2st(f=f, x=x)
        raise ValueError

    @staticmethod
    def _linearise_1st(*, f, x):
        def approx_residual(u):
            return u[1] - f(x[0])

        bias = approx_residual(x)

        def jvp(y):
            return y[1]

        return bias, jvp

    @staticmethod
    def _linearise_2nd(*, f, x):
        def approx_residual(u):
            return u[2] - f(x[0], x[1])

        bias = approx_residual(x)

        def jvp(y):
            return y[2]

        return bias, jvp
