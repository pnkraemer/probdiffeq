r"""Taylor-expand the solution of an initial value problem (IVP)."""


from probdiffeq.backend import control_flow, functools, itertools, tree_util
from probdiffeq.backend import numpy as np
from probdiffeq.backend.typing import Array, Callable


def taylor_mode_scan(vf: Callable, inits: tuple[Array, ...], /, num: int):
    """Taylor-expand the solution of an IVP with Taylor-mode differentiation.

    Other than `taylor_mode_unroll()`, this function implements the loop via a scan,
    which comes at the price of padding the loop variable with zeros as appropriate.
    It is expected to compile more quickly than `taylor_mode_unroll()`, but may
    execute more slowly.

    The differences should be small.
    Consult the benchmarks if performance is critical.
    """
    # Number of positional arguments in f
    num_arguments = len(inits)

    # Initial Taylor series (u_0, u_1, ..., u_k)
    primals = vf(*inits)
    taylor_coeffs = [*inits, primals]

    def body(tcoeffs, _):
        # Pad the Taylor coefficients in zeros, call jet, and return the solution.
        # This works, because the $i$th output coefficient of jet()
        # is independent of the $i+j$th input coefficient
        # (see also the explanation in taylor_mode_doubling)
        series = _subsets(tcoeffs[1:], num_arguments)  # for high-order ODEs
        p, s_new = functools.jet(vf, primals=inits, series=series)

        # The final values in s_new are nonsensical
        # (well, they are not; but we don't care about them)
        # so we remove them
        tcoeffs = [*inits, p, *s_new[:-1]]
        return tcoeffs, None

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


def taylor_mode_unroll(vf: Callable, inits: tuple[Array, ...], /, num: int):
    """Taylor-expand the solution of an IVP with Taylor-mode differentiation.

    Other than `taylor_mode_scan()`, this function does not depend on zero-padding
    the coefficients at the price of unrolling a loop of length `num-1`.
    It is expected to compile more slowly than `taylor_mode_scan()`,
    but execute more quickly.

    The differences should be small.
    Consult the benchmarks if performance is critical.

    !!! warning "Compilation time"
        JIT-compiling this function unrolls a loop.

    """
    # Number of positional arguments in f
    num_arguments = len(inits)

    # Initial Taylor series (u_0, u_1, ..., u_k)
    primals = vf(*inits)
    taylor_coeffs = [*inits, primals]

    for _ in range(num - 1):
        series = _subsets(taylor_coeffs[1:], num_arguments)  # for high-order ODEs
        p, s_new = functools.jet(vf, primals=inits, series=series)
        taylor_coeffs = [*inits, p, *s_new]
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


def forward_mode_recursive(vf: Callable, inits: tuple[Array, ...], /, num: int):
    """Taylor-expand the solution of an IVP with recursive forward-mode differentiation.

    !!! warning "Compilation time"
        JIT-compiling this function unrolls a loop.

    """
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


def taylor_mode_doubling(vf: Callable, inits: tuple[Array, ...], /, num_doublings: int):
    """Combine Taylor-mode differentiation and Newton's doubling.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        Support for Newton's doubling is highly experimental.
        There is no guarantee that it works correctly.
        It might be deleted tomorrow
        and without any deprecation policy.

    !!! warning "Compilation time"
        JIT-compiling this function unrolls a loop.

    """
    (u0,) = inits
    zeros = np.zeros_like(u0)

    def jet_embedded(*c, degree):
        """Call a modified jet().

        The modifications include:
        * We merge "primals" and "series" into a single set of coefficients
        * We expect and return _normalised_ Taylor coefficients.

        The reason for the latter is that the doubling-recursion
        simplifies drastically for normalised coefficients
        (compared to unnormalised coefficients).
        """
        coeffs_emb = [*c] + [zeros] * degree
        p, *s = _unnormalise(*coeffs_emb)
        p_new, s_new = functools.jet(vf, (p,), (s,))
        return _normalise(p_new, *s_new)

    taylor_coefficients = [u0]
    degrees = list(itertools.accumulate(map(lambda s: 2**s, range(num_doublings))))
    for deg in degrees:
        jet_embedded_deg = tree_util.Partial(jet_embedded, degree=deg)
        fx, jvp = functools.linearize(jet_embedded_deg, *taylor_coefficients)

        # Compute the next set of coefficients.
        # TODO: can we fori_loop() this loop?
        #  the running variable (cs_padded) should have constant size
        cs = [(fx[deg - 1] / deg)]
        cs_padded = cs + [zeros] * (deg - 1)
        for i, fx_i in enumerate(fx[deg : 2 * deg]):
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
            linear_combination = jvp(*cs_padded)[i]
            cs_ = cs_padded[: (i + 1)]
            cs_ += [(fx_i + linear_combination) / (i + deg + 1)]
            cs_padded = cs_ + [zeros] * (deg - i - 2)

        # Store all new coefficients
        taylor_coefficients.extend(cs_padded)

    return _unnormalise(*taylor_coefficients)


def _normalise(primals, *series):
    """Un-normalised Taylor series to normalised Taylor series."""
    series_new = [s / np.factorial(i + 1) for i, s in enumerate(series)]
    return primals, *series_new


def _unnormalise(primals, *series):
    """Normalised Taylor series to un-normalised Taylor series."""
    series_new = [s * np.factorial(i + 1) for i, s in enumerate(series)]
    return primals, *series_new
