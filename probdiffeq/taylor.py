r"""Taylor-expand the solution of an initial value problem (IVP)."""

from probdiffeq.backend import control_flow, functools, ode, tree_util
from probdiffeq.backend import numpy as np
from probdiffeq.backend.typing import Array, ArrayLike, Callable, Sequence
from probdiffeq.util import filter_util


def runge_kutta_starter(dt, *, num: int, prior, ssm, atol=1e-12, rtol=1e-10):
    """Create an estimator that uses a Runge-Kutta starter."""

    def starter(vf, initial_values: tuple, /, t):
        # TODO: higher-order ODEs
        # TODO: allow flexible "solve" method?

        # Assertions and early exits

        if len(initial_values) > 1:
            msg = "Higher-order ODEs are not supported at the moment."
            raise ValueError(msg)

        if num == 0:
            return [*initial_values]

        if num == 1:
            return [*initial_values, vf(*initial_values, t=t)]

        # Generate data

        k = num + 1  # important: k > num
        ts = np.linspace(t, t + dt * (k - 1), num=k, endpoint=True)
        ys = ode.odeint_and_save_at(
            vf, initial_values, save_at=ts, atol=atol, rtol=rtol
        )

        # Initial condition
        scale = ssm.prototypes.output_scale()
        rv_t0 = ssm.normal.standard(num + 1, scale)
        estimator = filter_util.fixedpointsmoother_precon(ssm=ssm)
        conditional_t0 = ssm.conditional.identity(num + 1)
        init = (rv_t0, conditional_t0)

        # Discretised prior
        scale = ssm.prototypes.output_scale()
        prior_vmap = functools.vmap(prior, in_axes=(0, None))
        ibm_transitions = prior_vmap(np.diff(ts), scale)

        # Generate an observation-model for the QOI
        # (1e-7 observation noise for nuggets and for reusing existing code)
        model_fun = functools.vmap(ssm.conditional.to_derivative, in_axes=(None, 0, 0))
        std = tree_util.tree_map(lambda s: 1e-7 * np.ones((len(s),)), ys)
        models = model_fun(0, ys, std)

        zeros = np.zeros_like(models.noise.mean)

        # Run the preconditioned fixedpoint smoother
        (corrected, conditional), _ = filter_util.estimate_fwd(
            zeros,
            init=init,
            prior_transitions=ibm_transitions,
            observation_model=models,
            estimator=estimator,
        )
        initial = ssm.conditional.marginalise(corrected, conditional)
        mean = ssm.stats.mean(initial)
        return ssm.unravel(mean)

    return starter


def odejet_padded_scan(vf: Callable, inits: Sequence[Array], /, num: int):
    """Taylor-expand the solution of an IVP with Taylor-mode differentiation.

    Other than `odejet_unroll()`, this function implements the loop via a scan,
    which comes at the price of padding the loop variable with zeros as appropriate.
    It is expected to compile more quickly than `odejet_unroll()`, but may
    execute more slowly.

    The differences should be small.
    Consult the benchmarks if performance is critical.
    """
    if not isinstance(inits[0], ArrayLike):
        _, unravel = tree_util.ravel_pytree(inits[0])
        inits_flat = [tree_util.ravel_pytree(m)[0] for m in inits]

        def vf_wrapped(*ys, **kwargs):
            ys = tree_util.tree_map(unravel, ys)
            return tree_util.ravel_pytree(vf(*ys, **kwargs))[0]

        tcoeffs = odejet_padded_scan(vf_wrapped, inits_flat, num=num)
        return tree_util.tree_map(unravel, tcoeffs)

    # Number of positional arguments in f
    num_arguments = len(inits)

    # Initial Taylor series (u_0, u_1, ..., u_k)
    primals = vf(*inits)
    taylor_coeffs = [*inits, primals]

    increment = odejet_coefficient_increment(vf, num_arguments=num_arguments)

    def body(tcoeffs, _):
        tcoeffs = increment(tcoeffs)

        # The final values in s_new are nonsensical
        # (well, they are not; but we don't care about them)
        # so we remove them
        return tcoeffs[:-1], None

    # Pad the initial Taylor series with zeros
    num_outputs = num_arguments + num
    zeros = np.zeros_like(primals)
    taylor_coeffs = _pad_to_length(taylor_coeffs, length=num_outputs, value=zeros)

    # Early exit for num=1.
    #  Why? because zero-length scan and disable_jit() don't work together.
    if num == 1:
        return taylor_coeffs

    # Compute all coefficients with scan().
    taylor_coeffs, _ = control_flow.scan(
        body, init=taylor_coeffs, xs=None, length=num - 1
    )
    return taylor_coeffs


def _pad_to_length(x, /, *, length, value):
    return x + [value] * (length - len(x))


def odejet_unroll(vf: Callable, inits: Sequence[Array], /, num: int):
    """Taylor-expand the solution of an IVP with Taylor-mode differentiation.

    Other than `odejet_padded_scan()`, this function does not depend on zero-padding
    the coefficients at the price of unrolling a loop of length `num-1`.
    It is expected to compile more slowly than `odejet_padded_scan()`,
    but execute more quickly.

    The differences should be small.
    Consult the benchmarks if performance is critical.

    !!! warning "Compilation time"
        JIT-compiling this function unrolls a loop.

    """
    if not isinstance(inits[0], ArrayLike):
        _, unravel = tree_util.ravel_pytree(inits[0])
        inits_flat = [tree_util.ravel_pytree(m)[0] for m in inits]

        def vf_wrapped(*ys, **kwargs):
            ys = tree_util.tree_map(unravel, ys)
            return tree_util.ravel_pytree(vf(*ys, **kwargs))[0]

        tcoeffs = odejet_unroll(vf_wrapped, inits_flat, num=num)
        return tree_util.tree_map(unravel, tcoeffs)

    # Number of positional arguments in f
    num_arguments = len(inits)

    # Initial Taylor series (u_0, u_1, ..., u_k)
    primals = vf(*inits)
    taylor_coeffs = [*inits, primals]

    increment = odejet_coefficient_increment(vf, num_arguments=num_arguments)
    for _ in range(num - 1):
        taylor_coeffs = increment(taylor_coeffs)
    return taylor_coeffs


def jet_unpack_series(taylor_series, num, /):
    """Compute Jet-compatible arguments from a Taylor series.

    Arguments
    ---------
    taylor_series
        A sequence of arrays to evaluate the Taylor series at.
    num
        The number of inputs to the root
        (2 for a first-order ODE, 3 for second-order, etc.)

    Examples
    --------
    >>> a = (1, 2, 3, 4, 5)
    >>> print(jet_unpack_series(a, n=1))
    [1], [(1, 2, 3, 4, 5)]
    >>> print(jet_unpack_series(a, n=2))
    [1, 2], [(1, 2, 3, 4), (2, 3, 4, 5)]
    >>> print(jet_unpack_series(a, n=3))
    [1, 2, 3], [(1, 2, 3), (2, 3, 4), (3, 4, 5)]

    """
    primals, series = taylor_series[:num], taylor_series[1:]

    def mask(i):
        return None if i == 0 else i

    series_ = [series[mask(k) : mask(k + 1 - num)] for k in range(num)]
    return primals, series_


def odejet_via_jvp(vf: Callable, inits: Sequence[Array], /, num: int):
    """Taylor-expand the solution of an IVP with recursive forward-mode differentiation.

    !!! warning "Compilation time"
        JIT-compiling this function unrolls a loop.

    """
    if not isinstance(inits[0], ArrayLike):
        _, unravel = tree_util.ravel_pytree(inits[0])
        inits_flat = [tree_util.ravel_pytree(m)[0] for m in inits]

        def vf_wrapped(*ys, **kwargs):
            ys = tree_util.tree_map(unravel, ys)
            return tree_util.ravel_pytree(vf(*ys, **kwargs))[0]

        tcoeffs = odejet_via_jvp(vf_wrapped, inits_flat, num=num)
        return tree_util.tree_map(unravel, tcoeffs)

    g_n, g_0 = vf, vf
    taylor_coeffs = [*inits, vf(*inits)]
    for _ in range(num - 1):
        g_n = _fwd_recursion_iterate(fun_n=g_n, fun_0=g_0)
        taylor_coeffs = [*taylor_coeffs, g_n(*inits)]
    return taylor_coeffs


def _fwd_recursion_iterate(*, fun_n, fun_0):
    r"""Increment $F_{n+1}(x) = \langle (JF_n)(x), f_0(x) \rangle$."""

    def df(*args):
        # Assign primals and tangents for the JVP
        vals = (*args, fun_0(*args))
        primals_in, tangents_in = vals[:-1], vals[1:]

        _, tangents_out = functools.jvp(fun_n, primals_in, tangents_in)
        return tangents_out

    return tree_util.Partial(df)


def odejet_doubling_unroll(vf: Callable, inits: Sequence[Array], /, num_doublings: int):
    """Combine Taylor-mode differentiation and Newton's doubling.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        Support for Newton's doubling is highly experimental.
        There is no guarantee that it works correctly.
        It might be deleted tomorrow
        and without any deprecation policy.

    !!! warning "Compilation time"
        JIT-compiling this function unrolls a loop.

    """
    if not isinstance(inits[0], ArrayLike):
        # If the Pytree elements are matrices or scalars,
        # promote and unpromote accordingly
        _, unravel = tree_util.ravel_pytree(inits[0])
        inits_flat = [tree_util.ravel_pytree(m)[0] for m in inits]

        def vf_wrapped(*ys, **kwargs):
            ys = tree_util.tree_map(unravel, ys)
            return tree_util.ravel_pytree(vf(*ys, **kwargs))[0]

        tcoeffs = odejet_doubling_unroll(
            vf_wrapped, inits_flat, num_doublings=num_doublings
        )
        return tree_util.tree_map(unravel, tcoeffs)

    double = odejet_coefficient_double(vf)
    (u0,) = inits  # This asserts ODEs are first-order only. High order is a todo
    taylor_coefficients = [u0]
    for _ in range(num_doublings):
        taylor_coefficients = double(taylor_coefficients)
    return _unnormalise(*taylor_coefficients)


def odejet_coefficient_increment(vf, *, num_arguments):
    """Construct a method that increments Taylor series' of an ODE."""

    def increment(taylor_coeffs):
        primals, series = jet_unpack_series(taylor_coeffs, num_arguments)
        p, s_new = functools.jet(vf, primals=primals, series=series)
        return [*primals, p, *s_new]

    return increment


def odejet_coefficient_double(vf):
    """Construct a method that doubles Taylor series' lengths of an ODE."""

    def double(taylor_coefficients):
        zeros = np.zeros_like(taylor_coefficients[0])
        deg = len(taylor_coefficients)

        jet_embedded_deg = tree_util.Partial(jet_embedded, degree=deg)
        fx, jvp = functools.linearize(jet_embedded_deg, *taylor_coefficients)

        def body_fun(cs_padded, i_and_fx_i):
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
            # i = k - deg
            i, fx_i = i_and_fx_i
            linear_combination = jvp(*(cs_padded[:-1]))[i]
            new = (fx_i + linear_combination) / (i + deg + 1)
            cs_padded = cs_padded.at[i + 1].set(new)
            return cs_padded, None

        cs = [(fx[deg - 1] / deg)]
        cs_padded = cs + [zeros] * (deg)
        cs_padded = np.stack(cs_padded)

        xs = [np.arange(0, len(fx[deg : 2 * deg])), fx[deg : 2 * deg]]
        cs_padded, _ = control_flow.scan(body_fun, xs=xs, init=cs_padded)

        taylor_coefficients.extend(cs_padded)
        return taylor_coefficients

    def jet_embedded(*c, degree):
        """Call a modified jet().

        The modifications include:
        * We merge "primals" and "series" into a single set of coefficients
        * We expect and return _normalised_ Taylor coefficients.

        The reason for the latter is that the doubling-recursion
        simplifies drastically for normalised coefficients
        (compared to unnormalised coefficients).
        """
        zeros = np.zeros_like(c[0])

        coeffs_emb = [*c] + [zeros] * degree
        p, *s = coeffs_emb
        p_new, s_new = functools.jet(vf, (p,), (s,), is_tcoeff=True)
        return np.stack([p_new, *s_new])

    return double


def _normalise(primals, *series):
    """Un-normalised Taylor series to normalised Taylor series."""
    series_new = [s / np.factorial(i + 1) for i, s in enumerate(series)]
    return [primals, *series_new]


def _unnormalise(primals, *series):
    """Normalised Taylor series to un-normalised Taylor series."""
    series_new = [s * np.factorial(i + 1) for i, s in enumerate(series)]
    return [primals, *series_new]


def odejet_affine(vf: Callable, inits: Sequence[Array], /, num: int):
    """Evaluate the Taylor series of an affine differential equation.

    !!! warning "Compilation time"
        JIT-compiling this function unrolls a loop of length `num`.

    """
    if num == 0:
        return inits

    if not isinstance(inits[0], ArrayLike):
        _, unravel = tree_util.ravel_pytree(inits[0])
        inits_flat = [tree_util.ravel_pytree(m)[0] for m in inits]

        def vf_wrapped(*ys, **kwargs):
            ys = tree_util.tree_map(unravel, ys)
            return tree_util.ravel_pytree(vf(*ys, **kwargs))[0]

        tcoeffs = odejet_affine(vf_wrapped, inits_flat, num=num)
        return tree_util.tree_map(unravel, tcoeffs)

    fx, jvp_fn = functools.linearize(vf, *inits)

    tmp = fx
    fx_evaluations = [tmp := jvp_fn(tmp) for _ in range(num - 1)]
    return [*inits, fx, *fx_evaluations]
