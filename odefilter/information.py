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
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class IsotropicEK0FirstOrder(eqx.Module):
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
    >>> linearise = IsotropicEK0FirstOrder()
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

    def __call__(
        self,
        f: Callable[[Float[Array, "n d"]], Float[Array, " d"]],
        x: Float[Array, "n d"],
    ) -> Tuple[Float[Array, " d"], Callable[[Float[Array, "n d"]], Float[Array, " d"]]]:
        """Linearise the ODE.

        Parameters
        ----------
        f :
            Vector field of a first-order ODE. Signature ``f(x)``.
        x :
            Linearisation point.

        Returns
        -------
        :
            Output bias.
        :
            Linear function (pushforward of the tangent spaces).
        """

        def approx_residual(u):
            return u[1] - f(x[0])

        bias = approx_residual(x)

        def jvp(y):
            return y[1]

        return bias, jvp


class EK1FirstOrder(eqx.Module):
    """EK1 information."""

    ode_dimension: int

    def __call__(self, f, x):
        """Linearise the ODE."""

        def residual(u):
            u_reshaped = jnp.reshape(u, (-1, self.ode_dimension), order="F")
            return u_reshaped[1] - f(u_reshaped[0])

        bias, jvp = jax.linearize(residual, x)
        return bias, jvp
