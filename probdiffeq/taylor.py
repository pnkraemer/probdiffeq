r"""Taylor-series estimation in initial value problems.

This module does not contain any probabilistic numerics logic.
Instead, its sole purpose is to make the probabilistic solvers
in probdiffeq.probdiffeq easier to access.

See the tutorials for example use cases.
"""

from probdiffeq import ssm_impl
from probdiffeq.backend import flow, func, inspect, np, tree
from probdiffeq.backend.typing import Array, ArrayLike, Callable, Sequence


def odejet_padded_scan(vf: Callable, inits: Sequence[ArrayLike], /, num: int):
    """Taylor-expand the solution of an IVP with Taylor-mode differentiation.

    Other than `odejet_unroll()`, this function implements the loop via a scan,
    which comes at the price of padding the loop variable with zeros as appropriate.
    It is expected to compile more quickly than `odejet_unroll()`, but may
    execute more slowly.

    The differences should be small.
    Consult the benchmarks if performance is critical.
    """
    inits = tree.tree_map(np.asarray, inits)
    if not isinstance(inits[0], Array):
        _, unravel = tree.ravel_pytree(inits[0])
        inits_flat = [tree.ravel_pytree(m)[0] for m in inits]

        def vf_wrapped(*ys, **kwargs):
            ys = tree.tree_map(unravel, ys)
            return tree.ravel_pytree(vf(*ys, **kwargs))[0]

        tcoeffs = odejet_padded_scan(vf_wrapped, inits_flat, num=num)
        return tree.tree_map(unravel, tcoeffs)

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
    taylor_coeffs, _ = flow.scan(body, init=taylor_coeffs, xs=None, length=num - 1)
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
    inits = tree.tree_map(np.asarray, inits)
    if not isinstance(inits[0], Array):
        _, unravel = tree.ravel_pytree(inits[0])
        inits_flat = [tree.ravel_pytree(m)[0] for m in inits]

        def vf_wrapped(*ys, **kwargs):
            ys = tree.tree_map(unravel, ys)
            return tree.ravel_pytree(vf(*ys, **kwargs))[0]

        tcoeffs = odejet_unroll(vf_wrapped, inits_flat, num=num)
        return tree.tree_map(unravel, tcoeffs)

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
    inits = tree.tree_map(np.asarray, inits)
    if not isinstance(inits[0], Array):
        _, unravel = tree.ravel_pytree(inits[0])
        inits_flat = [tree.ravel_pytree(m)[0] for m in inits]

        def vf_wrapped(*ys, **kwargs):
            ys = tree.tree_map(unravel, ys)
            return tree.ravel_pytree(vf(*ys, **kwargs))[0]

        tcoeffs = odejet_via_jvp(vf_wrapped, inits_flat, num=num)
        return tree.tree_map(unravel, tcoeffs)

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

        _, tangents_out = func.jvp(fun_n, primals_in, tangents_in)
        return tangents_out

    return tree.Partial(df)


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
    inits = tree.tree_map(np.asarray, inits)
    if not isinstance(inits[0], Array):
        # If the Pytree elements are matrices or scalars,
        # promote and unpromote accordingly
        _, unravel = tree.ravel_pytree(inits[0])
        inits_flat = [tree.ravel_pytree(m)[0] for m in inits]

        def vf_wrapped(*ys, **kwargs):
            ys = tree.tree_map(unravel, ys)
            return tree.ravel_pytree(vf(*ys, **kwargs))[0]

        tcoeffs = odejet_doubling_unroll(
            vf_wrapped, inits_flat, num_doublings=num_doublings
        )
        return tree.tree_map(unravel, tcoeffs)

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
        p, s_new = func.jet(vf, primals=primals, series=series)
        return [*primals, p, *s_new]

    return increment


def odejet_coefficient_double(vf):
    """Construct a method that doubles Taylor series' lengths of an ODE."""

    def double(taylor_coefficients):
        zeros = np.zeros_like(taylor_coefficients[0])
        deg = len(taylor_coefficients)

        jet_embedded_deg = tree.Partial(jet_embedded, degree=deg)
        fx, jvp = func.linearize(jet_embedded_deg, *taylor_coefficients)

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
        cs_padded, _ = flow.scan(body_fun, xs=xs, init=cs_padded)

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
        p_new, s_new = func.jet(vf, (p,), (s,), is_tcoeff=True)
        return np.stack([p_new, *s_new])

    return double


def _unnormalise(primals, *series):
    """Normalised Taylor series to un-normalised Taylor series."""
    series_new = [s * np.factorial(i + 1) for i, s in enumerate(series)]
    return [primals, *series_new]


def odejet_affine(vf: Callable, inits: Sequence[Array], /, num: int):
    """Evaluate the Taylor series of an affine differential equation.

    !!! warning "Compilation time"
        JIT-compiling this function unrolls a loop of length `num`.

    """
    inits = tree.tree_map(np.asarray, inits)

    if num == 0:
        return inits

    if not isinstance(inits[0], Array):
        _, unravel = tree.ravel_pytree(inits[0])
        inits_flat = [tree.ravel_pytree(m)[0] for m in inits]

        def vf_wrapped(*ys, **kwargs):
            ys = tree.tree_map(unravel, ys)
            return tree.ravel_pytree(vf(*ys, **kwargs))[0]

        tcoeffs = odejet_affine(vf_wrapped, inits_flat, num=num)
        return tree.tree_map(unravel, tcoeffs)

    fx, jvp_fn = func.linearize(vf, *inits)

    tmp = fx
    fx_evaluations = [tmp := jvp_fn(tmp) for _ in range(num - 1)]
    return [*inits, fx, *fx_evaluations]


def daejet_nonlinear_lstsq(
    differential: Callable,
    algebraic: Callable,
    inits: Sequence[Array],
    /,
    num: int,
    nlstsq: Callable,
):
    """Evaluate the Taylor series of a differential-algebraic equation system."""
    # For really high orders, recursively call this function

    root_order_differential = _verify_dae_signature_and_parse_order(differential)
    root_order_algebraic = _verify_dae_signature_and_parse_order(algebraic)

    # Determine degrees of freedom ("dof") and initialse all others diffusely
    # Concretely: The provided 'inits' are not DOFs, all added ones are.
    zeros = tree.tree_map(np.zeros_like, inits[0])
    ones = tree.tree_map(np.ones_like, inits[0])
    inits_std = [*[zeros for _ in inits], *[ones for _ in range(num)]]

    # Pad the initial values to the desired size
    zeros = tree.tree_map(np.zeros_like, inits[0])
    inits_mean = [*inits, *[zeros for _ in range(num)]]

    ssm = ssm_impl.FactSsmImpl.from_tcoeffs_dense(inits_mean)
    rv = ssm.normal.from_mean_and_std(inits_mean, inits_std)

    x0, unravel = tree.ravel_pytree(inits_mean)

    def root_jet(tcoeffs_flat):
        tcoeffs_all = unravel(tcoeffs_flat)

        # Differential part.
        # Assumes that the DAE is first order.
        ps, ss = jet_unpack_series(tcoeffs_all, root_order_differential)
        primals1, series1 = func.jet(differential, ps, ss, is_tcoeff=False)

        # Algebraic part
        # Assumes that the DAE is first order.
        ps, ss = jet_unpack_series(tcoeffs_all, root_order_algebraic)
        primals2, series2 = func.jet(algebraic, ps, ss, is_tcoeff=False)

        # Put together (order doesn't matter)
        fx = [primals1, *series1, primals2, *series2]
        return tree.ravel_pytree(fx)[0]

    x1, info = nlstsq(root_jet, x0, rv.mean, rv.cholesky)
    return unravel(x1), info


def _verify_dae_signature_and_parse_order(vf) -> int:
    """Parse the vector-field structure from its signature."""
    sig = inspect.signature(vf)
    params = list(sig.parameters.values())

    msg = f"""The dynamics' signature is not compatible with the constraint.

    More precisely, the dynamics are expected to look like

      - f(u, /),
      - f(u, du, /),
      - f(u, du, ddu, /),

    and so on, where the number of positional arguments
    specifies the order of the problem.
    Replace `u`, `du`, and so on with any variable name of your choosing
    but mind the keyword-only argument 't' in the signatures above.

    That said, the arguments

    {[(p.name, p.kind) for p in params]}

    have been detected in the dynamics function.

    Try wrapping the vector field through a pure Python function
    with the correct arguments before passing it to the ODE constraint.

      - No *args or **kwargs
      - No functools.partial

    """

    POSITIONAL = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    KEYWORD = (inspect.Parameter.KEYWORD_ONLY,)

    def is_positional(p):
        return p.kind in POSITIONAL

    def is_keyword(p):
        return p.kind in KEYWORD

    state_args = [p for p in params if is_positional(p)]
    contains_no_positional = len(state_args) == 0
    contains_keyword = len([p for p in params if is_keyword(p)]) > 0

    if contains_no_positional or contains_keyword:
        raise TypeError(msg)

    return len(state_args)
