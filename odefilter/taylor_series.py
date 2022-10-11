r"""Compute the Taylor series of the solution of a differential equation.

Commonly, this is done with recursive automatic differentiation
and used for the consistent initialisation of ODE filters.

More specifically:
ODE filters require an initialisation of the state
and its first $\nu$ derivatives, and numerical stability
requires an accurate initialisation.
The gold standard is Taylor-mode differentiation, but others can be
useful, too.

The general interface of these differentiation functions is as follows:

\begin{align*}
(u_0, \dot u_0, \ddot u_0, ...) &= G(f, (u_0,), n), \\
(u_0, \dot u_0, \ddot u_0, ...) &= G(f, (u_0, \dot u_0), n), \\
(u_0, \dot u_0, \ddot u_0, ...) &= G(f, (u_0, \dot u_0, ...), n).
\end{align*}

The inputs are vector fields, initial conditions, and an integer;
the outputs are unnormalised Taylor coefficients (or equivalently,
higher-order derivatives of the initial condition).
$f = f(u, du, ...)$ is an autonomous vector field of a
potentially high-order differential equation; $(u_0, \dot u_0, ...)$
is a tuple of initial conditions that matches the signature of the vector field;
and $n$ is the number of recursions, which commonly describes how many
derivatives to "add" to the existing initial conditions.
"""

from functools import partial
from typing import Callable, List, Tuple

import equinox as eqx
import jax
from jax.experimental.jet import jet
from jaxtyping import Array, Float


class TaylorMode(eqx.Module):
    """Taylor-mode initialisation."""

    @partial(jax.jit, static_argnames=("vector_field", "num"))
    def __call__(
        self,
        *,
        vector_field: Callable[..., Float[Array, " d"]],
        initial_values: Tuple[Float[Array, " d"], ...],
        num: int
    ) -> List[Float[Array, " d"]]:
        """Recursively differentiate the initial value of an \
         ODE with **Taylor-mode** automatic differentiation.

        Parameters
        ----------
        vector_field :
            An autonomous vector field to be differentiated.
        initial_values :
            A tuple (or iterable) of initial values.
            This is the initial set of Taylor coefficients.
            The vector field evaluates as ``vector_field(*initial_values)``.
        num :
            How many recursions the iteration shall use.
            One recursion adds one derivative to
            the existing set of Taylor coefficients.


        Returns
        -------
        :
            A list of (unnormalised) Taylor coefficients.

        Examples
        --------
        >>> import jax.tree_util
        >>>
        >>> def tree_round(x, *a, **kw):
        ...     return jax.tree_util.tree_map(lambda s: jnp.round(s, *a, **kw), x)
        >>>
        >>> import jax.numpy as jnp
        >>> f = lambda x: (x+1)**2*(1-jnp.cos(x))
        >>> u0 = (jnp.ones(1)*0.5,)
        >>> print(tree_round(f(*u0), 1))
        [0.3]

        >>> taylor_mode = TaylorMode()
        >>>
        >>> tcoeffs = taylor_mode(vector_field=f, initial_values=u0, num=1)
        >>> print(tree_round(tcoeffs, 1))
        [DeviceArray([0.5], dtype=float32), DeviceArray([0.3], dtype=float32)]

        >>> tcoeffs = taylor_mode(vector_field=f, initial_values=u0, num=2)
        >>> print(tree_round(tcoeffs, 1))
        [DeviceArray([0.5], dtype=float32), DeviceArray([0.3], dtype=float32), \
DeviceArray([0.4], dtype=float32)]

        >>>
        >>> f = lambda x, dx: dx**2*(1-jnp.sin(x))
        >>> u0 = (jnp.ones(1)*0.5, jnp.ones(1)*0.2)
        >>> print(tree_round(f(*u0), 2))
        [0.02]

        >>> tcoeffs = taylor_mode(vector_field=f, initial_values=u0, num=1)
        >>> print(tree_round(tcoeffs, 2))
        [DeviceArray([0.5], dtype=float32), DeviceArray([0.19999999], dtype=float32), \
DeviceArray([0.02], dtype=float32)]

        >>> tcoeffs = taylor_mode(vector_field=f, initial_values=u0, num=4)
        >>> print(tree_round(tcoeffs,1))
        [DeviceArray([0.5], dtype=float32), DeviceArray([0.2], dtype=float32), \
DeviceArray([0.], dtype=float32), DeviceArray([-0.], dtype=float32), \
DeviceArray([-0.], dtype=float32), DeviceArray([-0.], dtype=float32)]

        """
        # Number of positional arguments in f
        num_arguments = len(initial_values)

        # Initial Taylor series (u_0, u_1, ..., u_k)
        primals = vector_field(*initial_values)
        taylor_coeffs = [*initial_values, primals]
        for _ in range(num - 1):
            series = _subsets(taylor_coeffs[1:], num_arguments)
            primals, series_new = jet(
                vector_field, primals=initial_values, series=series
            )
            taylor_coeffs = [*initial_values, primals, *series_new]

        return taylor_coeffs


def _subsets(set, n):
    """Compute specific subsets until exhausted.

    See example below.

    Examples
    --------
    >>> a = (1, 2, 3, 4, 5)
    >>> print(_subsets(a, n=1))
    [(1, 2, 3, 4, 5)]
    >>> print(_subsets(a, n=2))
    [(1, 2, 3, 4), (2, 3, 4, 5)]
    >>> print(_subsets(a, n=3))
    [(1, 2, 3), (2, 3, 4), (3, 4, 5)]
    """

    def mask(i):
        return None if i == 0 else i

    return [set[mask(k) : mask(k + 1 - n)] for k in range(n)]


class ForwardMode(eqx.Module):
    """Forward-mode initialisation."""

    @partial(jax.jit, static_argnames=("vector_field", "num"))
    def __call__(
        self,
        *,
        vector_field: Callable[..., Float[Array, " d"]],
        initial_values: Tuple[Float[Array, " d"], ...],
        num: int
    ) -> List[Float[Array, " d"]]:
        """Recursively differentiate the initial value of an \
             ODE with **forward-mode** automatic differentiation.

        Parameters
        ----------
        vector_field :
            An autonomous vector field to be differentiated.
        initial_values :
            A tuple (or iterable) of initial values.
            This is the initial set of Taylor coefficients.
            The vector field evaluates as ``vector_field(*initial_values)``.
        num :
            How many recursions the iteration shall use.
            One recursion adds one derivative to
            the existing set of Taylor coefficients.

        Returns
        -------
        :
            A list of (unnormalised) Taylor coefficients.

        Examples
        --------
        >>> import jax.tree_util
        >>>
        >>> def tree_round(x, *a, **kw):
        ...     return jax.tree_util.tree_map(lambda s: jnp.round(s, *a, **kw), x)
        >>>
        >>> import jax.numpy as jnp
        >>> f = lambda x: (x+1)**2*(1-jnp.cos(x))
        >>> u0 = (jnp.ones(1)*0.5,)
        >>> print(tree_round(f(*u0), 1))
        [0.3]

        >>> forward_mode = ForwardMode()
        >>> tcoeffs = forward_mode(vector_field=f, initial_values=u0, num=1)
        >>> print(tree_round(tcoeffs, 1))
        [DeviceArray([0.5], dtype=float32), DeviceArray([0.3], dtype=float32)]

        >>> tcoeffs = forward_mode(vector_field=f, initial_values=u0, num=2)
        >>> print(tree_round(tcoeffs, 1))
        [DeviceArray([0.5], dtype=float32), DeviceArray([0.3], dtype=float32), \
DeviceArray([0.4], dtype=float32)]

        >>>
        >>> f = lambda x, dx: dx**2*(1-jnp.sin(x))
        >>> u0 = (jnp.ones(1)*0.5, jnp.ones(1)*0.2)
        >>> print(tree_round(f(*u0), 2))
        [0.02]

        >>> tcoeffs = forward_mode(vector_field=f, initial_values=u0, num=1)
        >>> print(tree_round(tcoeffs, 2))
        [DeviceArray([0.5], dtype=float32), DeviceArray([0.19999999], dtype=float32), \
DeviceArray([0.02], dtype=float32)]

        >>> tcoeffs = forward_mode(vector_field=f, initial_values=u0, num=4)
        >>> print(tree_round(tcoeffs,1))
        [DeviceArray([0.5], dtype=float32), DeviceArray([0.2], dtype=float32), \
DeviceArray([0.], dtype=float32), DeviceArray([-0.], dtype=float32), \
DeviceArray([-0.], dtype=float32), DeviceArray([-0.], dtype=float32)]

        """
        g_n, g_0 = vector_field, vector_field
        taylor_coeffs = [*initial_values, vector_field(*initial_values)]
        for _ in range(num - 1):
            g_n = self._fwd_recursion_iterate(fun_n=g_n, fun_0=g_0)
            taylor_coeffs = [*taylor_coeffs, g_n(*initial_values)]
        return taylor_coeffs

    @staticmethod
    def _fwd_recursion_iterate(*, fun_n, fun_0):
        r"""Increment $F_{n+1}(x) = \langle (JF_n)(x), f_0(x) \rangle$."""

        def df(*args):
            r"""Differentiate with a general version of the chain rule.

            More specifically, this function implements a generalised
            call $F(x) = \langle (Jf)(x), f_0(x)\rangle$.
            """
            # Assign primals and tangents for the JVP
            vals = (*args, fun_0(*args))
            primals_in, tangents_in = vals[:-1], vals[1:]

            primals_out, tangents_out = jax.jvp(fun_n, primals_in, tangents_in)
            return tangents_out

        return df
