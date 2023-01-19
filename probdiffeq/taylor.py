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
    tcoeffs = initial_values

    # Compute the recursion in normalised Taylor coefficients.
    # It simplifies extremely.
    def jet_padded(p, *s):
        tcoeffs_padded = _pad_with_zeros(p, *s)
        return jet_normalised(vf, *tcoeffs_padded)

    while True:

        ys, Js, jvp_fn = _naive_linearize(jet_padded, *tcoeffs)
        # ys, jvp_fn = jax.linearize(jet_padded, *tcoeffs)

        # Double the number of Taylor coefficients (compute "k" more)
        j = len(tcoeffs)
        for k in range(j - 1, 2 * j):
            # get x_k and augment tcoeffs as follows:
            tcoeffs = [
                *tcoeffs,
                _next_coeff(tcoeffs, ys=ys, Js=Js, jvp_fn=jvp_fn, j=j, k=k),
                # _next_coeff_jvp(tcoeffs, ys=ys, jvp_fn=jvp_fn, j=j, k=k),
            ]

            if k + 1 == num:
                return _unnormalise(*tcoeffs)


def _naive_linearize(fn, *x):
    fx, jvp_fn = jax.linearize(fn, *x)
    jacfwd = jax.jacfwd(fn, argnums=tuple(range(len(x))))(*x)
    jac_test = jnp.asarray(jacfwd)

    n, d = len(x), len(x[0])
    # fn: (n,d) -> (2n, d)
    # jac(fn): (n, d) -> [(n, d) -> (2n, n, d, d)]
    # jac(fn)[i][j] is the jacobian of the ith output w.r.t. the jth input

    # we build this jacobian from its JVPs.
    # jvp(fn) maps n directions with shape (d,) to
    # 2n directional derivatives with shape (d,).
    # we can replicate with jnp.einsum().
    direction = jnp.arange(1.0, 1.0 + n * d).reshape(n, d)
    deriv1 = jnp.einsum("ijkl,jl->ik", jac_test, direction)
    deriv2 = jvp_fn(*direction)
    assert jnp.allclose(deriv1, jnp.stack(deriv2))

    # to compute jac from jvp(fn), we repeat this n*d times
    # covering "all" directions. To do so, build an identity matrix
    # with shape (nd,nd). Reshape into n chunks of size (d,n,d).
    # and double-vmap jvp_fn along the "last" axis.
    # This gives 2n directional derivatives with shapes (d,n,d)
    # Stack them into an array with shape (2n,d,n,d)
    # and swap the first and second axis.
    id_nd = jnp.eye(n * d)
    std_basis = id_nd.reshape((n, d, n, d))
    jvp_fn_vmap = jax.vmap(jvp_fn, in_axes=-1, out_axes=-1)
    jvp_fn_vmap = jax.vmap(jvp_fn_vmap, in_axes=-1, out_axes=-1)
    jvp_evals = jnp.stack(jvp_fn_vmap(*std_basis))
    jac = jnp.swapaxes(jvp_evals, axis1=2, axis2=1)
    assert jnp.allclose(jac_test, jac)

    return fx, jac, jvp_fn


def _pad_with_zeros(*x):
    """Return padded array.

    E.g.
    (1, 2, 3) -> (1, 2, 3, 0, 0, 0)
    (1,) -> (1, 0)
    (1, 1, 1, 1) -> (1, 1, 1, 1, 0, 0, 0, 0),
    etc.
    """
    zeros = jnp.zeros_like(x[0])
    return [*x] + [zeros] * len(x)


def _next_coeff(tcoeffs, *, ys, jvp_fn, j, k, Js):
    if k < j:
        return ys[k] / (k + 1)

    # reproduce jvp_fn(*tangent) as jnp.einsum("ijkl,jl->ik", Js, tangent)
    d = len(tcoeffs[0])
    tangent = jnp.arange(1.0, 1.0 + j * d).reshape(j, d)
    deriv1 = jnp.einsum("ijkl,jl->ik", Js, tangent)
    deriv2 = jvp_fn(*tangent)
    assert jnp.allclose(deriv1, jnp.stack(deriv2))

    # Compute the sum
    Js_relevant = Js[j - 1][(2 * j - 1 - k) :]
    tcoeffs_relevant = tcoeffs[j:]
    summands = jnp.einsum(
        "ijk,ik->ij", Js_relevant[: k + 1 - j], tcoeffs_relevant[: k + 1 - j]
    )
    # summ = 0.0
    # for i in range(k + 1 - j):  # todo: remove loop
    #     summ += Js_relevant[i] @ tcoeffs_relevant[i]  # todo: use JVPs
    summ = jnp.sum(summands, axis=0)
    return (ys[k] + summ) / (k + 1)


def jet_normalised(fn, primals, *series):
    """jet(), but without primals & series and in normalised Taylor coefficients."""
    # todo: while this function is nice for not-so-good-at-jvp people like me (N),
    #   this function is unnecessary.
    p, *s = _unnormalise(primals, *series)
    p_new, s_new = jax.experimental.jet.jet(fn, (p,), (s,))
    return _normalise(p_new, *s_new)


def _normalise(primals, *series):
    """Unnormalised Taylor series to normalised Taylor series."""
    series_new = [s / _fct(i + 1) for i, s in enumerate(series)]
    return primals, *series_new


def _unnormalise(primals, *series):
    """Normalised Taylor series to unnormalised Taylor series."""
    series_new = [s * _fct(i + 1) for i, s in enumerate(series)]
    return primals, *series_new


def _fct(n, /):
    return jax.lax.exp(jax.lax.lgamma(n + 1.0))
