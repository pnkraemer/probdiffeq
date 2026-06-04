r"""Evaluate jet-recursions in differential equations."""

from probdiffeq import probdiffeq
from probdiffeq._probdiffeq import constraints, problem_types, utilities
from probdiffeq.backend import flow, func, np, tree
from probdiffeq.backend.typing import Array, Protocol, Sequence, TypeVar

__all__ = [
    "JetExpansionAlg",
    "jetexpand_ode_doubling_unroll",
    "jetexpand_ode_padded_scan",
    "jetexpand_ode_unroll",
    "jetexpand_ode_via_jvp",
    "jetexpand_residual",
]


T = TypeVar("T")
F = TypeVar(
    "F", bound=problem_types.ODEFunction | problem_types.Residual, contravariant=True
)


class JetExpansionAlg(Protocol[F]):
    """A protocol for methods that evaluate Taylor series' of IVPs from initial conditions."""

    def __call__(
        self, vf: F, inits: Sequence[T], /, *, t: float
    ) -> tuple[list[T], dict]:
        """Evaluate the Taylor series of an IVP.

        Parameters
        ----------
        vf
            The vector field of the IVP. It is expected to have a signature like
            `vf(u, /, t=...)` for first-order ODEs, `vf(u, du, /, t=...)` for second-order ODEs,
            and so on. That is, the number of positional arguments specifies the order of the problem.
            Replace `u`, `du`, and so on with any variable name of your choosing
            but mind the keyword-only argument 't' in the signatures above.
        inits
            The initial conditions of the IVP. The number of elements in this sequence should match the number of positional arguments in `vf`.
        t
            Additional keyword arguments to pass to `vf`. This is useful for passing the time `t` in the signatures above.

        Returns
        -------
        Sequence[T]
            A sequence of Taylor coefficients, starting with the initial conditions and followed by the derivatives.
        """


def jetexpand_ode_padded_scan(
    *, num: int
) -> JetExpansionAlg[problem_types.ODEFunction]:
    """Taylor-expand the solution of an IVP with Taylor-mode differentiation.

    Other than `jetexpand_ode_unroll()`, this function implements the loop via a scan,
    which comes at the price of padding the loop variable with zeros as appropriate.
    It is expected to compile more quickly than `jetexpand_ode_unroll()`, but may
    execute more slowly.

    The differences should be small.
    Consult the benchmarks if performance is critical.
    """

    @_error_if_vf_not_odefunction_type
    @_allow_pytree_inits
    def expand(
        vf: problem_types.ODEFunction, inits: Sequence[T], /, *, t: float
    ) -> tuple[list[T], dict]:
        if num == 0:
            return list(inits), {}

        # Number of positional arguments in f
        num_arguments = len(inits)

        # Initial Taylor series (u_0, u_1, ..., u_k)
        primals = vf.vector_field(jet_coords=inits, t=t)
        taylor_coeffs = [*inits, primals]

        # Early exit for num=1.
        #  Why? because zero-length scan and disable_jit() don't work together.
        if num == 1:
            return taylor_coeffs, {}

        increment = jetexpand_ode_coefficient_increment(num_arguments=num_arguments)

        def body(tcoeffs, _):
            tcoeffs = increment(vf, tcoeffs, t=t)

            # The final values in s_new are nonsensical
            # (well, they are not; but we don't care about them)
            # so we remove them
            return tcoeffs[:-1], None

        # Pad the initial Taylor series with zeros
        num_outputs = num_arguments + num
        zeros = np.zeros_like(primals)
        taylor_coeffs = _pad_to_length(taylor_coeffs, length=num_outputs, value=zeros)

        # Compute all coefficients with scan().
        taylor_coeffs, _ = flow.scan(body, init=taylor_coeffs, xs=None, length=num - 1)
        return taylor_coeffs, {}

    return expand


def _pad_to_length(x, /, *, length, value):
    return x + [value] * (length - len(x))


def jetexpand_ode_unroll(*, num: int) -> JetExpansionAlg[problem_types.ODEFunction]:
    """Taylor-expand the solution of an IVP with Taylor-mode differentiation.

    Other than `jetexpand_ode_padded_scan()`, this function does not depend on zero-padding
    the coefficients at the price of unrolling a loop of length `num-1`.
    It is expected to compile more slowly than `jetexpand_ode_padded_scan()`,
    but execute more quickly.

    The differences should be small.
    Consult the benchmarks if performance is critical.

    !!! warning "Compilation time"
        JIT-compiling this function unrolls a loop.

    """

    @_error_if_vf_not_odefunction_type
    @_allow_pytree_inits
    def expand(
        vf: problem_types.ODEFunction, inits: Sequence[T], /, *, t: float
    ) -> tuple[list[T], dict]:
        inits = tree.tree_map(np.asarray, inits)

        if num == 0:
            return list(inits), {}

        # Number of positional arguments in f
        num_arguments = len(inits)

        increment = jetexpand_ode_coefficient_increment(num_arguments=num_arguments)

        # Initial Taylor series (u_0, u_1, ..., u_k)
        primals = vf.vector_field(jet_coords=inits, t=t)
        taylor_coeffs = [*inits, primals]

        for _ in range(num - 1):
            taylor_coeffs = increment(vf, taylor_coeffs, t=t)
        return taylor_coeffs, {}

    return expand


def jetexpand_ode_coefficient_increment(*, num_arguments):
    """Construct a method that increments Taylor series' of an ODE."""

    def increment(
        vf: problem_types.ODEFunction, taylor_coeffs: Sequence[T], *, t: float
    ) -> list[T]:
        def vf_with_kwargs(*u: *tuple[T, ...]) -> T:
            return vf.vector_field(jet_coords=u, t=t)

        primals, series = utilities.jet_coords_to_primals_and_series(
            taylor_coeffs, num_arguments
        )
        p, s_new = func.jet(vf_with_kwargs, primals=primals, series=series)
        return [*primals, p, *s_new]

    return increment


def jetexpand_ode_via_jvp(*, num: int) -> JetExpansionAlg[problem_types.ODEFunction]:
    """Taylor-expand the solution of an IVP with recursive forward-mode differentiation.

    !!! warning "Compilation time"
        JIT-compiling this function unrolls a loop.

    """

    @_error_if_vf_not_odefunction_type
    @_allow_pytree_inits
    def expand(
        vf: problem_types.ODEFunction, inits: Sequence[T], /, *, t: float
    ) -> tuple[list[T], dict]:
        if num == 0:
            return list(inits), {}

        vf_wrapped = func.partial(vf, t=t)
        g_n, g_0 = vf_wrapped, vf_wrapped

        taylor_coeffs = [*inits, vf(*inits, t=t)]
        for _ in range(num - 1):
            g_n = _fwd_recursion_iterate(fun_n=g_n, fun_0=g_0)
            taylor_coeffs = [*taylor_coeffs, g_n(*inits)]
        return taylor_coeffs, {}

    return expand


def _fwd_recursion_iterate(*, fun_n, fun_0):
    r"""Increment $F_{n+1}(x) = \langle (JF_n)(x), f_0(x) \rangle$."""

    def df(*jet_coords: *tuple[T]) -> list[T]:
        # Assign primals and tangents for the JVP
        vals = (*jet_coords, fun_0(*jet_coords))
        primals_in, tangents_in = vals[:-1], vals[1:]

        _, tangents_out = func.jvp(fun_n, primals_in, tangents_in)
        return tangents_out

    return tree.Partial(df)


def jetexpand_ode_doubling_unroll(
    *, num_doublings: int
) -> JetExpansionAlg[problem_types.ODEFunction]:
    """Combine Taylor-mode differentiation and Newton's doubling.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        Support for Newton's doubling is highly experimental.
        There is no guarantee that it works correctly.
        It might be deleted tomorrow
        and without any deprecation policy.

    !!! warning "Compilation time"
        JIT-compiling this function unrolls a loop.

    """
    # TODO: error on the wrong type

    @_allow_pytree_inits
    def expand(
        vf: problem_types.ODEFunction, inits: Sequence[T], /, *, t: float
    ) -> tuple[list[T], dict]:
        inits = tree.tree_map(np.asarray, inits)

        double = jetexpand_ode_coefficient_double()
        (u0,) = inits  # This asserts ODEs are first-order only. High order is a todo
        taylor_coefficients = [u0]
        for _ in range(num_doublings):
            taylor_coefficients, _ = double(vf, taylor_coefficients, t=t)
        return _unnormalise(*taylor_coefficients), {}

    return expand


def jetexpand_ode_coefficient_double() -> JetExpansionAlg[problem_types.ODEFunction]:
    """Construct a method that doubles Taylor series' lengths of an ODE."""

    def double(
        vf: problem_types.ODEFunction, inits: Sequence[T], *, t: float
    ) -> tuple[list[T], dict]:
        taylor_coefficients = inits
        zeros = np.zeros_like(taylor_coefficients[0])
        deg = len(taylor_coefficients)

        jet_embedded_deg = tree.Partial(jet_embedded, vf, degree=deg, t=t)
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

        taylor_coefficients = list(taylor_coefficients)
        taylor_coefficients.extend(cs_padded)
        return taylor_coefficients, {}

    def jet_embedded(vf, *c, degree, t: float):
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
        p_new, s_new = func.jet(
            lambda *u: vf.vector_field(jet_coords=u, t=t), (p,), (s,), is_tcoeff=True
        )
        return np.stack([p_new, *s_new])

    return double


def _unnormalise(primals, *series):
    """Normalised Taylor series to un-normalised Taylor series."""
    series_new = [s * np.factorial(i + 1) for i, s in enumerate(series)]
    return [primals, *series_new]


def _error_if_vf_not_odefunction_type(expand):
    """Construct a decorator to check that the vector field is of type JetFunction."""

    def expand_wrapped(vf, inits, /, *, t: float):
        if not isinstance(vf, problem_types.ODEFunction):
            msg = f"Expected type {problem_types.ODEFunction}, but got {vf} of type {type(vf)}. "
            msg += "Make sure to wrap your vector field with `probdiffeq.ode()`."
            raise TypeError(msg)
        return expand(vf, inits, t=t)

    return expand_wrapped


def _allow_pytree_inits(expand):
    """Construct a decorator to allow pytrees as initial conditions in jet expansion algorithms."""

    def expand_wrapped(vf, inits, /, *, t: float):
        """Allow pytrees as initial conditions in jet expansion algorithms.

        If the initial conditions are not arrays,
        we assume they are pytrees of arrays and promote and unpromote accordingly
        """
        inits = tree.tree_map(np.asarray, inits)
        if not isinstance(inits[0], Array):
            _, unravel = tree.ravel_pytree(inits[0])
            inits_flat = [tree.ravel_pytree(m)[0] for m in inits]

            if vf.num_derivatives_in_args == 1:

                @probdiffeq.ode
                def vf_wrapped(y: T, /, *, t: float) -> T:
                    y = tree.tree_map(unravel, y)
                    fy = vf.vector_field(jet_coords=(y,), t=t)
                    return tree.ravel_pytree(fy)[0]
            elif vf.num_derivatives_in_args == 2:

                @probdiffeq.ode_second_order
                def vf_wrapped(y: T, dy: T, /, *, t: float) -> T:
                    y = tree.tree_map(unravel, y)
                    dy = tree.tree_map(unravel, dy)
                    fy = vf.vector_field(jet_coords=(y, dy), t=t)
                    return tree.ravel_pytree(fy)[0]
            else:
                raise ValueError
            tcoeffs = expand(vf_wrapped, inits_flat, t=t)
            return tree.tree_map(unravel, tcoeffs)

        return expand(vf, inits, t=t)

    return expand_wrapped


def jetexpand_residual(
    num: int,
    nlstsq: constraints.WeightedLeastSquaresNonlinearlyConstrained | None = None,
) -> JetExpansionAlg[problem_types.Residual]:
    """Evaluate the Taylor series of a differential-algebraic equation system."""
    if nlstsq is None:
        nlstsq = constraints.wlstsq_nc_gauss_newton()

    # TODO: don't try too hard to refactor this one here, I dont think it'll be around for long
    # TODO: enable pytree inputs/outputs
    # TODO: raise error if DAE has the wrong type
    def expand(
        residual: problem_types.Residual, inits: Sequence[T], /, *, t: float
    ) -> tuple[list[T], dict]:
        """Evaluate the Taylor series of a differential-algebraic equation system.

        The Taylor coefficients are computed by solving a nonlinear least squares problem
        at each order. This is a generalisation of the method described in
        "Taylor expansions of solutions to differential equations: a constructive approach"
        by Berz and Makino (1998) to DAEs.

        The method is expected to be more computationally expensive than jetexpand_ode_unroll(),
        but it can handle DAEs and extremely high orders.

        !!! warning "Warning: highly EXPERIMENTAL feature!"
            Support for DAEs is highly experimental.
            There is no guarantee that it works correctly.
            It might be deleted tomorrow
            and without any deprecation policy.
        """
        if num == 0:
            return list(inits), {}

        # Determine degrees of freedom ("dof") and initialse all others diffusely
        # Concretely: The provided 'inits' are not DOFs, all added ones are.
        ssm = probdiffeq.state_space_model()
        rv, _ = probdiffeq.prior_wiener_integrated(
            inits, diffuse_derivatives=num, ssm=ssm
        )

        x0, unravel = tree.ravel_pytree(rv.mean)

        def residual_jet(tcoeffs_flat):
            tcoeffs_all = unravel(tcoeffs_flat)
            coords = tcoeffs_all[: residual.num_derivatives_in_args]
            output = residual.residual_function(jet_coords=coords, t=t)

            # Flatten the output so that the Jacobians are matrices, not Pytrees.
            return tree.ravel_pytree(output)[0]

        x1, info = nlstsq(residual_jet, x0, rv.mean_flat, rv.cholesky_flat)
        return list(unravel(x1)), info

    return expand
