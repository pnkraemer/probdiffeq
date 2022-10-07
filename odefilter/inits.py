"""Initialisation of ODE filters via (automatic) differentiation."""

import jax
import jax.numpy as jnp
from jax.experimental.jet import jet


def taylor_mode(*, vector_field, initial_values, num_recursions):
    """

    Parameters
    ----------
    f: Function with signature f(*init)
    init: Tuple of initial values
    num: number of derivatives <to add?>

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

    >>> tcoeffs = tm(f=f, init=u0, num=1)
    >>> print(tree_round(tcoeffs, 1))
    (DeviceArray([0.5], dtype=float32), DeviceArray([0.3], dtype=float32))

    >>> tcoeffs = tm(f=f, init=u0, num=2)
    >>> print(tree_round(tcoeffs, 1))
    (DeviceArray([0.5], dtype=float32), DeviceArray([0.3], dtype=float32), DeviceArray([0.4], dtype=float32))


    >>>
    >>> f = lambda x, dx: dx**2*(1-jnp.sin(x))
    >>> u0 = (jnp.ones(1)*0.5, jnp.ones(1)*0.2)
    >>> print(tree_round(f(*u0), 2))
    [0.02]

    >>> tcoeffs = tm(f=f, init=u0, num=1)
    >>> print(tree_round(tcoeffs, 2))
    (DeviceArray([0.5], dtype=float32), DeviceArray([0.19999999], dtype=float32), DeviceArray([0.02], dtype=float32))

    >>> tcoeffs = tm(f=f, init=u0, num=4)
    >>> print(tree_round(tcoeffs,1))
    (DeviceArray([0.5], dtype=float32), DeviceArray([0.2], dtype=float32), DeviceArray([0.], dtype=float32), DeviceArray([-0.], dtype=float32), DeviceArray([-0.], dtype=float32), DeviceArray([-0.], dtype=float32))

    """
    assert num_recursions >= 1

    # Number of positional arguments in f
    num_arguments = len(initial_values)

    # Initial Taylor series (u_0, u_1, ..., u_k)
    primals = vector_field(*initial_values)
    taylor_coeffs = (*initial_values, primals)
    for _ in range(num_recursions - 1):
        series = _subsets(taylor_coeffs[1:], num_arguments)
        primals, series_new = jet(vector_field, primals=initial_values, series=series)
        taylor_coeffs = (*initial_values, primals, *series_new)

    return taylor_coeffs


def _subsets(set, n):
    """Computes specific subsets until exhausted.

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
    mask = lambda i: None if i == 0 else i
    return [set[mask(k) : mask(k + 1 - n)] for k in range(n)]


def forwardmode_jvp_fn(*, f, u0, num_derivatives):
    """Compute the initial derivatives with forward-mode AD (JVPs)."""
    if num_derivatives == 0:
        return u0.reshape((1, -1))

    g, g0 = f, f
    du0 = [u0, g(u0)]
    for _ in range(num_derivatives - 1):
        g = _forwardmode_jvp_next_ode_derivative(fun=g, fun0=g0)
        du0.append(g(u0))

    return jnp.stack(du0)


def _forwardmode_jvp_next_ode_derivative(fun, fun0):
    def dg(x):
        _, y = jax.jvp(fun, (x,), (fun0(x),))
        return y

    return dg


def forwardmode_fn(*, f, u0, num_derivatives):
    """Compute the initial derivatives with forward-mode AD."""
    if num_derivatives == 0:
        return u0.reshape((1, -1))

    def next_ode_derivative(fun, fun0):
        def dg(x):
            return jax.jacfwd(fun)(x) @ fun0(x)

        return dg

    g, g0 = f, f
    du0 = [u0, g(u0)]
    for _ in range(num_derivatives - 1):
        g = next_ode_derivative(fun=g, fun0=g0)
        du0.append(g(u0))

    return jnp.stack(du0)


def reversemode_fn(*, f, u0, num_derivatives):
    """Compute the initial derivatives with reverse-mode AD."""
    if num_derivatives == 0:
        return u0.reshape((1, -1))

    def next_ode_derivative(fun, fun0):
        def dg(x):
            return jax.jacrev(fun)(x) @ fun0(x)

        return dg

    g, g0 = f, f
    du0 = [u0, g(u0)]
    for _ in range(num_derivatives - 1):
        g = next_ode_derivative(fun=g, fun0=g0)
        du0.append(g(u0))

    return jnp.stack(du0)
