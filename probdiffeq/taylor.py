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

    def f_jet_raw(tcoeffs, f, num_zeros):
        primals, *series = tcoeffs
        if num_zeros > 0:
            zeros = [jax.tree_util.tree_map(jnp.zeros_like, primals)] * num_zeros
            series = [*series, *zeros]
        primals_new, series_new = jax.experimental.jet.jet(f, (primals,), (series,))
        return primals_new, *series_new

    fx = vf(*initial_values)
    # taylor_coefficients = [*initial_values, fx]
    taylor_coefficients = initial_values

    ############################################################################
    # prepare
    ############################################################################

    (x0,) = initial_values
    zero_term = jnp.zeros_like(x0)

    jet = jax.experimental.jet.jet
    linearize = jax.linearize
    _fct = _factorial

    coeffs = jnp.zeros((num + 2,) + x0.shape)
    coeffs = coeffs.at[0].set(x0)
    # print("Coeffs", coeffs)
    ############################################################################
    # s = 0, j = 1: we compute 2 more coefficients
    ############################################################################
    s, j = 0, 1

    def f_jet(a):
        return jet(vf, (a,), ((zero_term,),))

    (yhat_p, [yhat_s]), jvp_jet = linearize(f_jet, *coeffs[:1])

    for k in range(0, 2):
        if k < j:
            yk = yhat_p
            print("YK", yk)
        else:
            _temp = [
                _fct((j - 1) - (k - i)) / _fct(i) * coeffs[i] for i in range(j, k + 1)
            ]
            jac = jvp_jet(*_temp)
            yk = yhat_s[k - 1] + _fct(k) / _fct(j - 1) * jac[0]
            print("YK", yk)
        coeffs = coeffs.at[k + 1].set(yk)
    print()
    ############################################################################
    # s = 1, j = 3: we compute 4 more coefficients to have 7 in total
    ############################################################################
    s, j = 1, 3

    def f_jet(a, b, c):
        return jet(
            vf,
            (a,),
            (
                (
                    b,
                    c,
                    zero_term,
                    zero_term,
                    zero_term,
                ),
            ),
        )

    (yhat_p, yhat_s), jvp_jet = linearize(f_jet, *coeffs[:j])

    for k in range(2, 5):
        if k < j:
            yk = yhat_s[k - 1]
            print("YK", yk)
        else:
            # if k == j:
            #     print(coeffs[k])
            _temp1 = [
                _fct((j - 1) - (k - i)) / _fct(i) * coeffs[i] for i in range(j, k + 1)
            ] + [zero_term] * (2 - (k - j))
            # _temp1 =   [zero_term]*(2-(k-j)) +  [_fct((j - 1) - (k - i)) / _fct(i) * coeffs[i] for i in range(j, k + 1)]
            jac1 = jvp_jet(*_temp1)
            # jac2 = jvp_jet(*_temp2)
            # print("JAC", jac)
            yk = yhat_s[k - 1] + _fct(k) / _fct(j - 1) * jac1[0]  # [0]

            print("YK", yk)
        coeffs = coeffs.at[k + 1].set(yk)

    #
    # for J in jac[1]:
    #     print(J)
    # print(yhat_s[k-1] + _fct(k) / _fct(j - 1) * J)

    print(coeffs)

    assert False

    #
    #
    # def f_jet0(a):
    #         return jax.experimental.jet.jet(vf, (a,), ((zero_term,),))
    #
    #     (y0, [y1h]), f_jvp = jax.linearize(f_jet0, x0)
    #     x1 = y0
    #     y1 = y1h + f_jvp(x1)[0]
    #     x2 = y1
    #     taylor_coefficients = [*taylor_coefficients, x1, x2]
    #     print(jnp.stack(taylor_coefficients))
    #     print()
    #     def f_jet01(a, b):
    #         return jax.experimental.jet.jet(vf, (a,), ([b, x2] + [zero_term]*3,))
    #
    #     (y0, [y1, y2, y3h, y4h, y5h]), f_jvp = jax.linearize(f_jet01, x0, x1)
    #     y3 = y3h + f_jvp(zero_term, y2)[1][0]
    #     y4 = y4h + f_jvp(zero_term, y3)[1][0]
    #     y5 = y5h + f_jvp(y4, zero_term)[1][0]
    #     x3 = y2
    #     x4 = y3
    #     x5 = y4
    #     x6 = y5
    #
    #     taylor_coefficients = [*taylor_coefficients, x3, x4, x5, x6]
    #     print(jnp.stack(taylor_coefficients))
    #
    #
    #
    #
    #
    #
    #     assert False
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #

    for s in range(2):

        # s = 0, j = 1 (from 1 to 3 initial values)
        j = len(taylor_coefficients)
        jet_fn = jax.tree_util.Partial(f_jet_raw, f=vf, num_zeros=j)

        yhats = jet_fn(taylor_coefficients)
        assert len(yhats) == 2 * j, len(yhats) - 2 * j

        As_all = jax.jacfwd(jet_fn)(taylor_coefficients)
        As = [As_all[0][i] for i in range(j)]
        assert len(As) == j, len(As) - j

        g0, g1 = yhats
        (G0,) = As

        x1 = g0
        x2 = g1 + G0 @ x1

        taylor_coefficients = [*initial_values, x1, x2]
        print(jnp.stack(taylor_coefficients))

        # s = 1, j = 3 (from 3 to 5 initial values)
        j = len(taylor_coefficients)
        jet_fn = jax.tree_util.Partial(f_jet_raw, f=vf, num_zeros=j)

        yhats = jet_fn(taylor_coefficients)
        assert len(yhats) == 2 * j, len(yhats) - 2 * j

        As_all = jax.jacfwd(jet_fn)(taylor_coefficients)
        As = [As_all[0][i] for i in range(j)]
        assert len(As) == j, len(As) - j

        _, _, g2, g3, g4, g5 = yhats
        G0, G1, G2 = As

        x3 = g2
        x4 = g3 + G0 @ x3
        x5 = g4 + _factorial(4) / _factorial(2) * (
            _factorial(2 - 1) / _factorial(3) * G1 @ x3
            + _factorial(2 - 0) / _factorial(4) * G0 @ x4
        )
        x6 = g5 + _factorial(5) / _factorial(2) * (
            1 / _factorial(3) * G2 @ x3
            + _factorial(2 - 1) / _factorial(4) * G1 @ x4
            + _factorial(2 - 0) / _factorial(5) * G0 @ x5
        )

        taylor_coefficients = [*taylor_coefficients, x3, x4, x5, x6]
        print(jnp.stack(taylor_coefficients))

        assert False

        # s = 2, j = 7 (from 3 to 5 initial values)
        j = len(taylor_coefficients)
        jet_fn = jax.tree_util.Partial(f_jet_raw, f=vf, num_zeros=j)

        yhats = jet_fn(taylor_coefficients)
        assert len(yhats) == 2 * j, len(yhats) - 2 * j

        As_all = jax.jacfwd(jet_fn)(taylor_coefficients)
        As = [As_all[0][i] for i in range(j)]
        assert len(As) == j, len(As) - j

        g0, g1, g2, g3, g4, g5, g6, g7, g8, g9 = yhats
        G0, G1, G2, G3, G4 = As

        taylor_coefficients = [
            *taylor_coefficients,
            g4,
            g5 + G0 @ g4,
            # g6 + G0 @ g4 + G1 @ (g5 + G0 @ g4),
            # g7 + G0 @ g4 + G1 @ (g5 + G0 @ g4) + G2 @ (g6 + G0 @ g4 + G1 @ (g5 + G0 @ g4)),
            #
            # g2,
            # g3 + G0 @ g2,
            # # g4 + G0 @ g2 + G1 @ (g3 + G0 @ g2),
            # # g5 + G0 @ g2 + G1 @ (g3 + G0 @ g2) + G2 @ (g4 + G0 @ g2 + G1 @ (g3 + G0 @ g2))
        ]
        print(jnp.stack(taylor_coefficients))

        assert False

        for k in range(j):
            a = yhats[k]
            b = As[-k]
            c = taylor_coefficients[-1]
            yk = a + b @ c  # * _factorial(j+k) / _factorial(j)
            taylor_coefficients = [*taylor_coefficients, yk]
            print(j + k, 2 * j, jnp.stack(taylor_coefficients))
            if k + j >= num:
                print("kfahsjkfajklshds")
                return taylor_coefficients
        # assert False
        print()
    #
    #     assert False
    #     print(As)
    #     # print(yhats[0], As[0], yhats[1])
    #     x0 = yhats[0]
    #     x1 = As[0][0] @ yhats[1]
    #     print([*taylor_coefficients, x0, x1])
    #     print(jax.tree_util.tree_map(jnp.shape, (yhats, As)))
    #     assert False
    #     yhats, jvps = jax.linearize(jet_fn, taylor_coefficients)
    #     """'yhat' is (y_0, ..., y_(2j-1). 'jvps' is (A_0, ..., A_(2j-1)) expresses as linear operators."""
    #
    #     for k in range(j, 2*j):
    #         print("k", k)
    #         print(taylor_coefficients, j, k+1)
    #         # print(len(taylor_coefficients), j, k)
    #         _temp = [taylor_coefficients[i-1] for i in range(j, k+1) ]
    #         jvp = jvps(_temp)
    #         # print(jvp)
    #         taylor_coefficients = [*taylor_coefficients, yhats[k]]
    #         print(taylor_coefficients)
    #         # print(taylor_coefficients)
    #         # print(k, num)
    #         #
    #         # if k == num:
    #         #     print()
    #         #     break
    # return taylor_coefficients
    # # Number of positional arguments in f
    #
    #
    # # Initial Taylor series (u_0, u_1, ..., u_k)
    # primals = vf(*initial_values)
    # taylor_coeffs = [*initial_values, primals]
    # for _ in range(num - 1):
    #     series = taylor_coeffs[1:]  # for high-order ODEs
    #     primals, series_new = jax.experimental.jet.jet(
    #         vf, primals=initial_values, series=(series,)
    #     )
    #     taylor_coeffs = [*initial_values, primals, *series_new]
    # return taylor_coeffs


#
# assert False
# print(j, taylor_coefficients)
#
# print()
# print()
# print("jet1")
# print(f_jet(taylor_coefficients))
# print()
# print()
# print("jet2")
# yhat, jvp_jet = jax.linearize(f_jet, taylor_coefficients)
# print(yhat)
# print()
# print()
# for k in range(j-1, 2*j):
#     print(j, k)
#     _temp = [_factorial(j-1-k+i) / _factorial(i) * taylor_coefficients[i] for i in range(j, k+1)]
#     print("jet3 (jvp)")
#     yk = yhat[k] + _factorial(k) / _factorial(j-1) * jvp_jet(_temp)
#
#     taylor_coefficients = [*taylor_coefficients, yk]


def _factorial(n, /):
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
