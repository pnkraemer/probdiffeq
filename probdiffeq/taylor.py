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

import functools
from typing import Callable, Tuple

import jax
import jax.experimental.jet
import jax.experimental.ode
import jax.numpy as jnp

from probdiffeq.implementations import recipes


@functools.partial(jax.jit, static_argnames=["vector_field", "num"])
def taylor_mode_fn(
    *, vector_field: Callable, initial_values: Tuple, num: int, t, parameters
):
    """Taylor-mode AD."""
    # raise RuntimeError("if the vector  field depends on t, we are a bit screwed?")
    # Number of positional arguments in f
    num_arguments = len(initial_values)

    vf = jax.tree_util.Partial(vector_field, t=t, p=parameters)

    # Initial Taylor series (u_0, u_1, ..., u_k)
    primals = vf(*initial_values)
    taylor_coeffs = [*initial_values, primals]
    for _ in range(num - 1):
        series = _subsets(taylor_coeffs[1:], num_arguments)  # for high-order ODEs
        primals, series_new = jax.experimental.jet.jet(
            vf, primals=initial_values, series=series
        )
        taylor_coeffs = [*initial_values, primals, *series_new]
    return taylor_coeffs


def _subsets(x, /, n):
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

    return [x[mask(k) : mask(k + 1 - n)] for k in range(n)]


# @functools.partial(jax.jit, static_argnames=["vector_field", "num"])
def taylor_mode_doubling_fn(
    *, vector_field: Callable, initial_values: Tuple, num: int, t, parameters
):
    """Taylor-mode AD."""
    vf = jax.tree_util.Partial(vector_field, t=t, p=parameters)
    tcoeffs = list(initial_values)
    zeros = jnp.zeros_like(tcoeffs[0])

    def jet_pad(x, jj):
        return jet_normalised(vf, x + [zeros] * jj)

    # s = 0, j = 1
    j = len(tcoeffs)
    ys = jet_pad(tcoeffs, jj=j)
    Js = jax.jacfwd(jet_pad, argnums=0)(tcoeffs, jj=j)

    y0 = ys[0]
    y1 = (ys[1] + Js[0][0] @ y0) / 2
    tcoeffs = [*tcoeffs, y0, y1]

    # s = 1, j = 3
    j = len(tcoeffs)
    ys = jet_pad(tcoeffs, jj=j)
    Js = jax.jacfwd(jet_pad, argnums=0)(tcoeffs, jj=j)
    # print(Js)

    y2 = ys[2] / 3
    y3 = (ys[3] + Js[0][0] @ y2) / 4
    y4 = (ys[4] + Js[1][0] @ y2 + Js[0][0] @ y3) / 5
    y5 = (ys[5] + Js[2][0] @ y2 + Js[1][0] @ y3 + Js[0][0] @ y4) / 6
    tcoeffs = [*tcoeffs, y2, y3, y4, y5]
    return _unnormalise(tcoeffs)
    # print(jnp.stack(tcoeffs))
    # assert False

    print(yhat)
    print(jac)

    #
    # def f_jet_raw(tcoeffs, f, num_zeros):
    #     primals, *series = tcoeffs
    #     if num_zeros > 0:
    #         zeros = [jax.tree_util.tree_map(jnp.zeros_like, primals)] * num_zeros
    #         series = [*series, *zeros]
    #     primals_new, series_new = jax.experimental.jet.jet(f, (primals,), (series,))
    #     return primals_new, *series_new

    fx = vf(*initial_values)
    # taylor_coefficients = [*initial_values, fx]
    taylor_coefficients = initial_values


#


def jet_normalised(fn, tcoeffs):
    tcoeffs = list(tcoeffs)
    tcoeffs = _unnormalise(tcoeffs)
    p, *s = tcoeffs
    p_new, s_new = jax.experimental.jet.jet(fn, (p,), (s,))
    tcoeffs = [p_new, *s_new]
    return _normalise(tcoeffs)


def _normalise(tcoeffs):
    primals, *series = tcoeffs
    k = len(series)
    for i in range(k):
        series[i] = series[i] / _fct(1 + i)
    return primals, *series


def _unnormalise(tcoeffs):
    primals, *series = tcoeffs
    k = len(series)
    for i in range(k):
        series[i] = series[i] * _fct(1 + i)
    return primals, *series


def _fct(n, /):
    return jax.lax.exp(jax.lax.lgamma(n + 1.0))


@functools.partial(jax.jit, static_argnames=["vector_field", "num"])
def forward_mode_fn(
    *, vector_field: Callable, initial_values: Tuple, num: int, t, parameters
):
    """Forward-mode AD."""
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
        r"""Differentiate with a general version of the chain rule.

        More specifically, this function implements a generalised
        call $F(x) = \langle (Jf)(x), f_0(x)\rangle$.
        """
        # Assign primals and tangents for the JVP
        vals = (*args, fun_0(*args))
        primals_in, tangents_in = vals[:-1], vals[1:]

        _, tangents_out = jax.jvp(fun_n, primals_in, tangents_in)
        return tangents_out

    return jax.tree_util.Partial(df)


def affine_recursion(
    *, vector_field: Callable, initial_values: Tuple, num: int, t, parameters
):
    """Compute the exact Taylor series of an affine differential equation."""
    vf = jax.tree_util.Partial(vector_field, t=t, p=parameters)

    fx, jvp_fn = jax.linearize(vf, *initial_values)
    ys = [*initial_values, fx]
    if num == 0:
        return initial_values

    if num == 1:
        return ys

    for _ in range(num - 2):
        fx = jvp_fn(fx)
        ys = [*ys, fx]
    return ys


def make_runge_kutta_starter_fn(*, dt=1e-6, atol=1e-12, rtol=1e-10):
    """Create a routine that estimates a Taylor series with a Runge-Kutta starter."""
    return functools.partial(_runge_kutta_starter_fn, dt0=dt, atol=atol, rtol=rtol)


# atol and rtol are static bc. of odeint...
@functools.partial(jax.jit, static_argnames=["vector_field", "num", "atol", "rtol"])
def _runge_kutta_starter_fn(
    *, vector_field, initial_values, num: int, t, parameters, dt0, atol, rtol
):
    # todo [INACCURATE]: the initial-value uncertainty is discarded
    # todo [FEATURE]: allow implementations other than IsoIBM?
    # todo [FEATURE]: higher-order ODEs

    # Assertions and early exits

    if len(initial_values) > 1:
        raise ValueError("Higher-order ODEs are not supported at the moment.")

    if num == 0:
        return initial_values

    if num == 1:
        return initial_values + (vector_field(*initial_values, t=t, p=parameters),)

    # Generate data

    def func(y, t, *p):
        return vector_field(y, t=t, p=p)

    k = num + 1  # important: k > num
    ts = jnp.linspace(t, t + dt0 * (k - 1), num=k, endpoint=True)
    ys = jax.experimental.ode.odeint(
        func, initial_values[0], ts, *parameters, atol=atol, rtol=rtol
    )

    # Run fixed-point smoother

    _impl = recipes.IsoTS0.from_params(num_derivatives=num)
    extrapolation = _impl.extrapolation

    # Initialise
    init_ssv = extrapolation.init_ssv(ode_shape=initial_values[0].shape)

    # Estimate
    u0_full = _runge_kutta_starter_improve(init_ssv, extrapolation, ys=ys, dt=dt0)

    # Turn the mean into a tuple of arrays and return
    taylor_coefficients = tuple(u0_full.hidden_state.mean)
    return taylor_coefficients


def _runge_kutta_starter_improve(init_ssv, extrapolation, ys, dt):
    # Initialise backward-transitions
    init_bw = extrapolation.init_conditional(ssv_proto=init_ssv)
    init_val = init_ssv, init_bw

    # Scan
    fn = functools.partial(_rk_filter_step, extrapolation=extrapolation, dt=dt)
    carry_fin, _ = jax.lax.scan(fn, init=init_val, xs=ys, reverse=False)
    (corrected_fin, bw_fin) = carry_fin

    # Backward-marginalise to get the initial value
    return bw_fin.marginalise(corrected_fin)


def _rk_filter_step(carry, y, extrapolation, dt):
    # Read
    (rv, bw_old) = carry

    # Extrapolate (with fixed-point-style merging)
    lin_pt, extra_cache = extrapolation.begin_extrapolation(rv, dt=dt)
    extra, bw_model = extrapolation.revert_markov_kernel(
        linearisation_pt=lin_pt, p0=rv, cache=extra_cache, output_scale_sqrtm=1.0
    )
    bw_new = bw_old.merge_with_incoming_conditional(bw_model)

    # Correct
    _, (corr, _) = extra.condition_on_qoi_observation(y, observation_std=0.0)

    # Return correction and backward-model
    return (corr, bw_new), None
