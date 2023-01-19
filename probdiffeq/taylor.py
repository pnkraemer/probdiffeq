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


# @functools.partial(jax.jit, static_argnames=["vector_field", "num"])
def taylor_mode_doubling_fn(
    *, vector_field: Callable, initial_values: Tuple, num: int, t, parameters
):
    """Taylor-mode AD."""
    vf = jax.tree_util.Partial(vector_field, t=t, p=parameters)

    # Compute the recursion in normalised Taylor coefficients.
    # It simplifies extremely.
    def jet_padded(p, *s):  # "primals" and "series"
        zeros = jnp.zeros_like(p)
        tcoeffs_padded = [p, *s] + [zeros] * (len(s) + 1)
        p, *s = _unnormalise(*tcoeffs_padded)
        p_new, s_new = jax.experimental.jet.jet(vf, (p,), (s,))
        return _normalise(p_new, *s_new)

    taylor_coefficients = list(initial_values)
    while (deg := len(taylor_coefficients)) < num + 1:

        # Propagate the Taylor coefficients through f
        # to compute the Taylor series of f(taylor_series).
        # Assemble the Jacobians of the 'deg'th Taylor coefficient
        # of the output with respect to the input coefficients.
        # f_coeff is a (2*deg, ode_dim) array.
        # df_deg_coeff is a (??, ??) array.
        # todo: return only f_coeff[deg-1:] because all others are irrelevant.
        fx, dfx = _linearize(jet_padded, *taylor_coefficients)
        fx_current, dfx_current = fx, dfx[deg - 1][:deg]

        # Compute the next set of coefficients.
        cs = [(fx_current[deg - 1] / deg)]
        for k in range(deg, min(2 * deg, num)):

            # array([J[-1],]),
            # array([J[-2], J[-1]]), (...),
            # array([J[0], ..., J[-1]])
            df_deg_coeff = dfx_current[(2 * deg - 1 - k) :]

            # Compute the next coefficient
            # according to Table 13.7 in Griewank/Walther
            linear_combination = _jvp_recursion(cs, dfx=df_deg_coeff)
            cs += [(fx_current[k] + linear_combination) / (k + 1)]

        # Store all new coefficients
        taylor_coefficients.extend(cs)

    # Return only the coefficients that are of interest.
    return _unnormalise(*taylor_coefficients)


def _linearize(fn, *x):
    fx = fn(*x)
    jac = jax.jacfwd(fn, argnums=tuple(range(len(x))))(*x)
    return fx, jnp.asarray(jac)


def _jvp_recursion(c, *, dfx):
    """Compute the Jacobian-vector product recursion.

    dfx[-1] @ c[0]                                      (-> c[1])
    dfx[-2] @ c[0] + dfx[-1] @ c[1]                     (-> c[2])
    dfx[-3] @ c[0] + dfx[-2] @ c[1] + dfx[-1] @ c[2]    (-> c[3])

    and so on.
    According to
    "Taylor-Mode Automatic Differentiation for Higher-Order Derivatives in JAX"
    by Bettencourt et al. (2019), this could be done in a
    single Jacobian-vector product.
    """
    summands = jnp.einsum("ijk,ik->ij", dfx, jnp.asarray(c))
    return jnp.sum(summands, axis=0)


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
