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
from typing import Callable, Tuple

import jax
from jax.experimental.jet import jet


@partial(jax.jit, static_argnames=["num"])
def taylor_mode_fn(*, vector_field: Callable, initial_values: Tuple, num: int):
    """Taylor-mode AD."""
    # Number of positional arguments in f
    num_arguments = len(initial_values)

    # Initial Taylor series (u_0, u_1, ..., u_k)
    primals = vector_field(*initial_values)
    taylor_coeffs = [*initial_values, primals]
    for _ in range(num - 1):
        series = _subsets(taylor_coeffs[1:], num_arguments)
        primals, series_new = jet(vector_field, primals=initial_values, series=series)
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


@partial(jax.jit, static_argnames=["num"])
def forward_mode_fn(*, vector_field: Callable, initial_values: Tuple, num: int):
    """Forward-mode AD."""
    g_n, g_0 = vector_field, vector_field
    taylor_coeffs = [*initial_values, vector_field(*initial_values)]
    for _ in range(num - 1):
        g_n = _fwd_recursion_iterate(fun_n=g_n, fun_0=g_0)
        taylor_coeffs = [*taylor_coeffs, g_n(*initial_values)]
    return taylor_coeffs


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
