r"""Taylor-expand the solution of an initial value problem (IVP)."""

import functools
from typing import Callable, Tuple

import jax
import jax.experimental.jet
import jax.experimental.ode
import jax.numpy as jnp

from probdiffeq.impl import impl
from probdiffeq.solvers.strategies import discrete

#
# from probdiffeq.solvers.statespace import extra


@functools.partial(jax.jit, static_argnames=["vector_field", "num"])
def taylor_mode_fn(
    *, vector_field: Callable, initial_values: Tuple, num: int, t, parameters
):
    """Taylor-expand the solution of an IVP with Taylor-mode differentiation."""
    # Number of positional arguments in f
    num_arguments = len(initial_values)

    vf = jax.tree_util.Partial(vector_field, t=t, p=parameters)

    # Initial Taylor series (u_0, u_1, ..., u_k)
    primals = vf(*initial_values)
    taylor_coeffs = [*initial_values, primals]

    def body_fn(tcoeffs, _):
        # Pad the Taylor coefficients in zeros, call jet, and return the solution.
        # This works, because the $i$th output coefficient of jet()
        # is independent of the $i+j$th input coefficient
        # (see also the explanation in taylor_mode_doubling_fn)
        series = _subsets(tcoeffs[1:], num_arguments)  # for high-order ODEs
        p, s_new = jax.experimental.jet.jet(vf, primals=initial_values, series=series)

        # The final values in s_new are nonsensical
        # (well, they are not; but we don't care about them)
        # so we remove them
        tcoeffs = [*initial_values, p, *s_new[:-1]]
        return tcoeffs, None

    # Pad the initial Taylor series with zeros
    num_outputs = num_arguments + num
    zeros = jnp.zeros_like(primals)
    taylor_coeffs = _pad_to_length(taylor_coeffs, length=num_outputs, value=zeros)

    # Early exit for num=1.
    #  Why? because zero-length scan and disable_jit() don't work together.
    if num == 1:
        return taylor_coeffs

    # Compute all coefficients with scan().
    taylor_coeffs, _ = jax.lax.scan(
        body_fn, init=taylor_coeffs, xs=None, length=num - 1
    )
    return taylor_coeffs


def _pad_to_length(x, /, *, length, value):
    return x + [value] * (length - len(x))


def _subsets(x, /, n):
    """Compute staggered subsets.

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

    return [x[mask(k) : mask(k + 1 - n)] for k in range(n)]


@functools.partial(jax.jit, static_argnames=["vector_field", "num"])
def forward_mode_fn(
    *, vector_field: Callable, initial_values: Tuple, num: int, t, parameters
):
    """Taylor-expand the solution of an IVP with forward-mode differentiation.

    !!! warning "Compilation time"
        JIT-compiling this function unrolls a loop of length `num`.



    """
    vf = jax.tree_util.Partial(vector_field, t=t, p=parameters)

    g_n, g_0 = vf, vf
    taylor_coeffs = [*initial_values, vf(*initial_values)]
    for _ in range(num - 1):
        g_n = _fwd_recursion_iterate(fun_n=g_n, fun_0=g_0)
        taylor_coeffs = [*taylor_coeffs, g_n(*initial_values)]
    return taylor_coeffs


def _fwd_recursion_iterate(*, fun_n, fun_0):
    r"""Increment $F_{n+1}(x) = \langle (JF_n)(x), f_0(x) \rangle$."""

    def df(*args):
        # Assign primals and tangents for the JVP
        vals = (*args, fun_0(*args))
        primals_in, tangents_in = vals[:-1], vals[1:]

        _, tangents_out = jax.jvp(fun_n, primals_in, tangents_in)
        return tangents_out

    return jax.tree_util.Partial(df)


@functools.partial(jax.jit, static_argnames=["vector_field", "num"])
def affine_recursion(
    *, vector_field: Callable, initial_values: Tuple, num: int, t, parameters
):
    """Evaluate the Taylor series of an affine differential equation.

    !!! warning "Compilation time"
        JIT-compiling this function unrolls a loop of length `num`.

    """
    if num == 0:
        return initial_values

    vf = jax.tree_util.Partial(vector_field, t=t, p=parameters)
    fx, jvp_fn = jax.linearize(vf, *initial_values)

    tmp = fx
    fx_evaluations = [tmp := jvp_fn(tmp) for _ in range(num - 1)]
    return [*initial_values, fx, *fx_evaluations]


def make_runge_kutta_starter_fn(*, dt=1e-6, atol=1e-12, rtol=1e-10):
    """Create an estimator that uses a Runge-Kutta starter."""
    return functools.partial(_runge_kutta_starter_fn, dt0=dt, atol=atol, rtol=rtol)


# atol and rtol must be static bc. of jax.odeint...
@functools.partial(jax.jit, static_argnames=["vector_field", "num", "atol", "rtol"])
def _runge_kutta_starter_fn(
    *, vector_field, initial_values, num: int, t, parameters, dt0, atol, rtol
):
    # todo [inaccuracy]: the initial-value uncertainty is discarded
    # todo [feature]: allow implementations other than IsoIBM?
    # todo [feature]: higher-order ODEs

    # Assertions and early exits

    if len(initial_values) > 1:
        raise ValueError("Higher-order ODEs are not supported at the moment.")

    if num == 0:
        return initial_values

    if num == 1:
        return (*initial_values, vector_field(*initial_values, t=t, p=parameters))

    # Generate data

    def func(y, t, *p):
        return vector_field(y, t=t, p=p)

    # todo: allow flexible "solve" method?
    k = num + 1  # important: k > num
    ts = jnp.linspace(t, t + dt0 * (k - 1), num=k, endpoint=True)
    ys = jax.experimental.ode.odeint(
        func, initial_values[0], ts, *parameters, atol=atol, rtol=rtol
    )

    # Discretise the prior
    rv = impl.ssm_util.standard_normal(num + 1, 1.0)
    cond_initial = impl.ssm_util.identity_conditional(num + 1)
    discretise = impl.ssm_util.ibm_transitions(num)
    cond, precon = jax.vmap(discretise)(jnp.diff(ts))

    # Generate an observation-model for the QOI
    # (1e-7 observation noise for nuggets and for reusing existing code)
    model_fun = jax.vmap(impl.ssm_util.conditional_to_derivative, in_axes=(None, 0))
    models = model_fun(0, 1e-7 * jnp.ones_like(ts))

    # Run the preconditioned fixedpoint smoother
    (corrected, conditional), _ = discrete.fixedpointsmoother_precon(
        ys,
        init=(rv, cond_initial),
        conditional=(cond, precon),
        observation_model=models,
    )
    initial = impl.conditional.marginalise(corrected, conditional)
    return tuple(impl.random.mean(initial))


@functools.partial(jax.jit, static_argnames=["vector_field", "num"])
def taylor_mode_doubling_fn(
    *, vector_field: Callable, initial_values: Tuple, num: int, t, parameters
):
    """Combine Taylor-mode differentiation and Newton's doubling.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        Support for Newton's doubling is highly experimental.
        There is no guarantee that it works correctly.
        It might be deleted tomorrow
        and without any deprecation policy.

    !!! warning "Compilation time"
        JIT-compiling this function unrolls a loop of length `num`.

    """
    vf = jax.tree_util.Partial(vector_field, t=t, p=parameters)
    (u0,) = initial_values
    zeros = jnp.zeros_like(u0)

    def jet_embedded(*c, degree):
        """Call a modified jax.experimental.jet().

        The modifications include:
        * We merge "primals" and "series" into a single set of coefficients
        * We expect and return _normalised_ Taylor coefficients.

        The reason for the latter is that the doubling-recursion
        simplifies drastically for normalised coefficients
        (compared to unnormalised coefficients).
        """
        coeffs_emb = [*c] + [zeros] * degree
        p, *s = _unnormalise(*coeffs_emb)
        p_new, s_new = jax.experimental.jet.jet(vf, (p,), (s,))
        return _normalise(p_new, *s_new)

    taylor_coefficients = [u0]
    while (deg := len(taylor_coefficients)) < num + 1:
        jet_embedded_deg = jax.tree_util.Partial(jet_embedded, degree=deg)
        fx, jvp_fn = jax.linearize(jet_embedded_deg, *taylor_coefficients)

        # Compute the next set of coefficients.
        # todo: can we jax.fori_loop() this loop?
        #  the running variable (cs_padded) should have constant size
        cs = [(fx[deg - 1] / deg)]
        for k in range(deg, min(2 * deg, num)):
            # The Jacobian of the embedded jet is block-banded,
            # i.e., of the form (for j=3)
            # (A0, 0, 0; A1, A0, 0; A2, A1, A0; *, *, *; *, *, *; *, *, *)
            # Thus, by attaching zeros to the current set of coefficients
            # until the input and output shapes match, we compute
            # the convolution-like sum of matrix-vector products with
            # a single call to the JVP function.
            # Bettencourt et al. (2019;
            # "Taylor-mode autodiff for higher-order derivatives in JAX")
            # explain details.
            cs_padded = cs + [zeros] * (2 * deg - k - 1)
            linear_combination = jvp_fn(*cs_padded)[k - deg]
            cs += [(fx[k] + linear_combination) / (k + 1)]

        # Store all new coefficients
        taylor_coefficients.extend(cs)

    return _unnormalise(*taylor_coefficients)


def _normalise(primals, *series):
    """Un-normalised Taylor series to normalised Taylor series."""
    series_new = [s / _fct(i + 1) for i, s in enumerate(series)]
    return primals, *series_new


def _unnormalise(primals, *series):
    """Normalised Taylor series to un-normalised Taylor series."""
    series_new = [s * _fct(i + 1) for i, s in enumerate(series)]
    return primals, *series_new


def _fct(n, /):  # factorial
    return jax.lax.exp(jax.lax.lgamma(n + 1.0))