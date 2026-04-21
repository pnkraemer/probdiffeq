"""Probabilistic solvers for differential equations.

See the tutorials for example use cases.
"""

from probdiffeq import diffeqjet, ssm_impl
from probdiffeq.backend import flow, func, inspect, linalg, np, random, structs, tree
from probdiffeq.backend.typing import (
    Any,
    Array,
    Callable,
    Generic,
    Literal,
    Protocol,
    Sequence,
    TypeVar,
)

C = TypeVar("C", bound=Sequence)
"""A type-variable to describe sequences.

Used to type Taylor coefficients, for example.
"""

N = TypeVar("N", bound=ssm_impl.AbstractTreeNormal)
"""A type-variable to describe normal distributions.

Used to type marginals, for example.
"""


class JacobianHandler:
    """An interface for working with Jacobian matrices."""

    def init_jacobian_handler(self):
        """Initialize the handler state.

        For example, if the handler uses stochastic sampling,
        this initialisation would create a random key.
        """
        raise NotImplementedError

    def materialize_dense(self, fun, x, state, /):
        """Materialize a dense Jacobian.

        This is typically used for first-order linearization in dense
        state-space models.
        """
        raise NotImplementedError

    def calculate_trace(self, fun, x, state, /):
        """Calculate the trace of a Jacobian.

        This is typically used for first-order linearization in isotropic
        state-space models.
        """
        raise NotImplementedError

    def calculate_diagonal(self, fun, x, state, /):
        """Calculate the diagonal of a Jacobian.

        This is typically used for first-order linearization in block-diagonal
        state-space models.
        """
        raise NotImplementedError


class jacobian_materialize(JacobianHandler):
    """Construct a handler that always materialized Jacobian matrices.

    Use this Jacobian if the dimension of the problem is relatively small.
    """

    def __init__(self, *, jacfun=func.jacfwd) -> None:
        self.jacfun = jacfun

    def init_jacobian_handler(self):
        return ()

    def materialize_dense(self, fun, x, state, /):
        del state
        fx = fun(x)
        dfx = func.jacfwd(fun)(x)
        return fx, dfx, ()

    def calculate_trace(self, fun, x, state, /):
        del state
        fx = fun(x)
        dfx = func.jacfwd(fun)(x)
        dfx_trace = linalg.trace(dfx)
        return fx, dfx_trace, ()

    def calculate_diagonal(self, fun, x, state, /):
        del state
        fx = fun(x)
        dfx = func.jacfwd(fun)(x)
        dfx_diagonal = linalg.diagonal(dfx)
        return fx, dfx_diagonal, ()


class jacobian_hutchinson_fwd(JacobianHandler):
    """Construct a handler that uses stochastic trace estimation for traces/diagonals.

    Use a Hutchinson handler if the dimension of the problem is large.

    This implementation uses **forward-mode** automatic differentiation.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This function is highly experimental and not safe to use.
        There is no guarantee that it works correctly (or at all).
        It might be deleted tomorrow and without any deprecation policy.

    """

    def __init__(self, *, seed=1, num_probes=10) -> None:
        self.seed = seed
        self.num_probes = num_probes

    def init_jacobian_handler(self):
        return random.prng_key(seed=self.seed)

    def materialize_dense(self, fun, x, state, /):
        # TODO: approximate Jacobian with outer products instead of forming?
        # What is the "correct" thing to do?
        fx = fun(x)
        dfx = func.jacfwd(fun)(x)
        return fx, dfx, state

    def calculate_trace(self, fun, x, key, /):
        key, subkey = random.split(key, num=2)
        sample_shape = (self.num_probes, *x.shape)
        v = random.rademacher(subkey, shape=sample_shape, dtype=x.dtype)

        fx, Jvp = func.linearize(fun, x)
        J_trace = func.vmap(lambda s: linalg.vector_dot(s, Jvp(s)))(v)
        J_trace = J_trace.mean(axis=0)
        return fx, J_trace, key

    def calculate_diagonal(self, fun, x, key, /):
        key, subkey = random.split(key, num=2)
        sample_shape = (self.num_probes, *x.shape)
        v = random.rademacher(subkey, shape=sample_shape, dtype=x.dtype)

        fx, Jvp = func.linearize(fun, x)
        vJv = func.vmap(lambda s: s * Jvp(s))(v)
        J_diagonal = vJv.mean(axis=0)
        return fx, J_diagonal, key


class jacobian_hutchinson_rev(JacobianHandler):
    """Construct a handler that uses stochastic trace estimation for traces/diagonals.

    Use a Hutchinson handler if the dimension of the problem is large.

    This implementation uses **reverse-mode** automatic differentiation.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This function is highly experimental and not safe to use.
        There is no guarantee that it works correctly (or at all).
        It might be deleted tomorrow and without any deprecation policy.

    """

    def __init__(self, *, seed=1, num_probes=10) -> None:
        self.seed = seed
        self.num_probes = num_probes

    def init_jacobian_handler(self):
        return random.prng_key(seed=self.seed)

    def materialize_dense(self, fun, x, state, /):
        # TODO: approximate Jacobian with outer products instead of forming?
        # What is the "correct" thing to do?
        fx = fun(x)
        dfx = func.jacrev(fun)(x)
        return fx, dfx, state

    def calculate_trace(self, fun, x, key, /):
        key, subkey = random.split(key, num=2)
        sample_shape = (self.num_probes, *x.shape)
        v = random.rademacher(subkey, shape=sample_shape, dtype=x.dtype)

        fx, vjp = func.vjp(fun, x)
        J_trace = func.vmap(lambda s: linalg.vector_dot(s, vjp(s)[0]))(v)
        J_trace = J_trace.mean(axis=0)
        return fx, J_trace, key

    def calculate_diagonal(self, fun, x, key, /):
        key, subkey = random.split(key, num=2)
        sample_shape = (self.num_probes, *x.shape)
        v = random.rademacher(subkey, shape=sample_shape, dtype=x.dtype)

        fx, vjp = func.vjp(fun, x)
        vJv = func.vmap(lambda s: s * vjp(s)[0])(v)
        J_diagonal = vJv.mean(axis=0)
        return fx, J_diagonal, key


def loss_lml_terminal_values(*, ssm: ssm_impl.FactSsmImpl, tcoeff_index=0):
    """Construct a log-marginal-likelihood loss for the terminal value."""

    def loss(u, /, *, marginals, std):
        u = tree.tree_map(np.asarray, u)

        # TODO: this is the wrong shape! We should expect std.shape == u.shape...
        std_expected = marginals.std_tree()[tcoeff_index]
        std = tree.tree_map(np.asarray, std)
        shapes = tree.tree_map(lambda a, b: a.shape == b.shape, std, std_expected)
        shapes_equal = tree.tree_all(shapes)

        if not shapes_equal:
            msg = "The standard deviation container differs from what was expected."
            msg += f" Expected: shape={tree.tree_map(np.shape, std_expected)}."
            msg += f" Received: shape={tree.tree_map(np.shape, std)}."
            msg += f" For reference, data: shape={tree.tree_map(np.shape, u)}."
            raise ValueError(msg)

        model = ssm.prior.to_derivative(tcoeff_index, std)
        marg = ssm.conditional.marginalise(marginals, model)
        return marg.logpdf_tree(u)

    return loss


def loss_lml_timeseries(
    *,
    ssm: ssm_impl.FactSsmImpl,
    average_pdfs: bool = True,
    tcoeff_index=0,
    solve_triu=linalg.lstsq_svd,
):
    """Construct a log-marginal-likelihood loss for a time-series."""

    def loss(u, /, *, posterior, std):
        if not isinstance(posterior, MarkovSequence):
            msg = "The datatype of the posterior is not as expected."
            msg += f" Expected: {MarkovSequence}."
            msg += f" Received: {type(posterior)}."
            msg += " Did you perhaps use a filter instead of a smoother"
            msg += ", or did you perhaps intend to use a different loss?"
            raise TypeError(msg)

        u = tree.tree_map(np.asarray, u)

        std_expected = posterior.marginal.std_tree()[tcoeff_index]
        std = tree.tree_map(np.asarray, std)
        shapes = tree.tree_map(lambda a, b: a.shape == b.shape, std, std_expected)
        shapes_equal = tree.tree_all(shapes)

        if not shapes_equal:
            msg = "The standard deviation container differs from what was expected."
            msg += f" Expected: shape={tree.tree_map(np.shape, std_expected)}."
            msg += f" Received: shape={tree.tree_map(np.shape, std)}."
            msg += f" For reference, data: shape={tree.tree_map(np.shape, u)}."
            raise ValueError(msg)

        def make_model(s):
            return ssm.prior.to_derivative(tcoeff_index, s)

        model = func.vmap(make_model)(std)

        # Use solve_triu=lstsq because for noise-free observations, the initial state
        # of the ODE solution tends to be noise-free,
        # which clashes and returns NaNs if we use exact solvers.
        return posterior.evaluate_lml(
            u, model=model, ssm=ssm, average_pdfs=average_pdfs, solve_triu=solve_triu
        )

    return loss


class Constraint(Protocol):
    """An interface for constraints + linearization in probabilistic solvers.

    Related:
    [`constraint_ode_ts0`](#probdiffeq.probdiffeq.constraint_ode_ts0),
    [`constraint_ode_ts1`](#probdiffeq.probdiffeq.constraint_ode_ts1),
    """

    init_linearization: Callable
    """Initialize the linearization of the constraint."""

    linearize: Callable
    """Linearize the constraint."""

    root_order: int
    """The order of the root-constraint.

    Here, 'order' relates to the highest derivative that the
    constraint depends on; for instance, in first-order ODEs,
    the root_order would be two; and in second-order ODEs,
    the root_order would be three.
    """


# TODO: should we go back to an EK0 and EK1 naming to ensure consistency
#       with papers and other libraries?
#       There is no more statistical linear regression
#       (nor will there ever be) so technicalities regarding *how* we linearize
#       are not relevant anymore.
def constraint_ode_ts0(vf, /, *, ssm):
    """Create an ODE constraint with zeroth-order Taylor linearisation.

    Related:
    [`Constraint`](#probdiffeq.probdiffeq.Constraint).
    """
    ode_order = _verify_vector_field_signature_and_parse_order(vf)
    return ssm.linearize.ode_taylor_0th(vf, ode_order=ode_order)


def constraint_ode_ts1(
    vf, /, *, ssm: ssm_impl.FactSsmImpl, jacobian: JacobianHandler | None = None
):
    """Create an ODE constraint with first-order Taylor linearisation.

    Related:
    [`Constraint`](#probdiffeq.probdiffeq.Constraint).

    The ODE vector field is assumed to be one of ``f(u, *, t)``, ``f(u, du, *, t)``, etc.
    The order of the ODE is read off the number of positional arguments before t.
    That is, for first-order ODEs, pass ``f(u, *, t)``,
    for second-order ODEs, pass ``f(u, du, *, t)``,
    for third-order ODEs ``f(u, du, ddu, *, t)``, and so on.

    """
    ode_order = _verify_vector_field_signature_and_parse_order(vf)
    if jacobian is None:
        # Use Hutchinson-Jacobian handling for backward compatibility.
        jacobian = jacobian_hutchinson_fwd()
    return ssm.linearize.ode_taylor_1st(vf, ode_order=ode_order, jacobian=jacobian)


def _verify_vector_field_signature_and_parse_order(vf) -> int:
    """Parse the vector-field structure from its signature."""
    sig = inspect.signature(vf)
    params = list(sig.parameters.values())

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

    msg = f"""The dynamics' signature is not compatible with the constraint.

    More precisely, the dynamics are expected to look like

      - f(u, /, *, t),
      - f(u, du, /, *, t),
      - f(u, du, ddu, /, *, t),

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

    contains_no_positional = len(state_args) == 0
    t_is_not_keyword = not any(is_keyword(p) and p.name == "t" for p in params)
    contains_keyword_other_than_t = any(is_keyword(p) and p.name != "t" for p in params)

    if contains_no_positional or t_is_not_keyword or contains_keyword_other_than_t:
        raise TypeError(msg)

    return len(state_args)


def constraint_jet(
    root,
    *,
    ssm: ssm_impl.FactSsmImpl,
    jacobian=None,
    nlstsq=None,
    jet_order: int | Literal["max"] = "max",
):
    """Construct a constraint that implements Jet-linearization.

    (What is Jet-linearisation? Stay tuned!).

    To use posterior linearisation, pass a `nlstsq` implementation.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This function is highly experimental and not safe to use.
        There is no guarantee that it works correctly (or at all).
        It might be deleted tomorrow and without any deprecation policy.


    """
    root_order = _verify_vector_field_signature_and_parse_order(root)

    if jacobian is None:
        jacobian = jacobian_hutchinson_fwd()

    def root_jet(*tcoeffs_all, t):
        unravel = tree.ravel_pytree(tcoeffs_all[0])[1]
        if jet_order == "max":
            tcoeffs = tcoeffs_all
        else:
            jet_order_upper = len(tcoeffs_all) - root_order
            if jet_order < 0 or jet_order > jet_order_upper:
                msg = "The provided jet-order is incompatible with the root order."
                msg += f" Expected: 0 <= jet_order <= {jet_order_upper}."
                msg += f" Received: jet_order == {jet_order}."
                raise ValueError(msg)

            order = root_order + jet_order
            tcoeffs = tcoeffs_all[:order]

        # Flatten the root because jax.jet is a bit high maintenance :)
        def jet_call(*y):
            y_tree = [unravel(s) for s in y]
            fx = root(*y_tree, t=t)
            return tree.ravel_pytree(fx)[0]

        flat = [tree.ravel_pytree(s)[0] for s in tcoeffs]
        ps, ss = diffeqjet.jet_unpack_series(flat, root_order)

        if len(tree.tree_leaves(ss)) == 0:
            fx = jet_call(*ps)
            return [fx]

        primals, series = func.jet(jet_call, ps, ss, is_tcoeff=False)
        return [primals, *series]

    order = (
        ssm.prior.shape_info.num_derivatives + 1
        if jet_order == "max"
        else jet_order + root_order
    )
    return ssm.linearize.root(
        root_jet, root_order=order, jacobian=jacobian, nlstsq=nlstsq
    )


def constraint_jet_imex(
    *,
    implicit: Callable,
    explicit: Callable,
    ssm: ssm_impl.FactSsmImpl,
    jacobian=None,
    nlstsq=None,
    jet_order_implicit="max",
    jet_order_explicit="max",
):
    """Like `constraint_jet`, but for roots summing implicit and explicit terms.

    The advantage of a dedicated IMEX constraint is that gradients can be stopped
    through the explicit part, which enables state-space model factorisation.
    In other words, think of the Jet-IMEX constraint as a generalisation
    of zeroth-order methods to implicit differential equations.


    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This function is highly experimental and not safe to use.
        There is no guarantee that it works correctly (or at all).
        It might be deleted tomorrow and without any deprecation policy.

    """
    root_order_im = _verify_vector_field_signature_and_parse_order(implicit)
    root_order_ex = _verify_vector_field_signature_and_parse_order(explicit)

    if jacobian is None:
        jacobian = jacobian_hutchinson_fwd()

    def root_jet(*tcoeffs_all, t):
        _, unravel = tree.ravel_pytree(tcoeffs_all[0])
        fx_implicit = jet_call(
            implicit,
            tcoeffs_all,
            root_order=root_order_im,
            jet_order=jet_order_implicit,
            unravel=unravel,
            t=t,
        )
        fx_explicit = jet_call(
            explicit,
            tcoeffs_all,
            root_order=root_order_ex,
            jet_order=jet_order_explicit,
            unravel=unravel,
            t=t,
        )

        # The Jacobian of the explicit term is ignored,
        # which turns first-order linearisation of root_jet into
        # first-order linearisation of the implicit term but zeroth-order
        # linearisation in the explicit term!
        fx_explicit = [func.stop_gradient(f) for f in fx_explicit]

        # Return the sum: c(x) = Imp(x) + Exp(x)
        return tree.tree_map(lambda a, b: a + b, fx_implicit, fx_explicit)

    def jet_call(fun, tcoeffs_all, /, *, root_order, jet_order, unravel, t):
        """Evaluate the jet'ed root function."""
        if jet_order == "max":
            tcoeffs = tcoeffs_all
        else:
            jet_order_upper = len(tcoeffs_all) - root_order
            if jet_order < 0 or jet_order > jet_order_upper:
                msg = "The provided jet-order is incompatible with the root order."
                msg += f" Expected: 0 <= jet_order <= {jet_order_upper}."
                msg += f" Received: jet_order == {jet_order}."
                raise ValueError(msg)
            order = jet_order + root_order
            tcoeffs = tcoeffs_all[:order]

        # Flatten the root because jax.jet is a bit high maintenance :)
        def jet_call(*y):
            y_tree = [unravel(s) for s in y]
            fx = fun(*y_tree, t=t)
            return tree.ravel_pytree(fx)[0]

        coeffs_flat = [tree.ravel_pytree(s)[0] for s in tcoeffs]
        ps, ss = diffeqjet.jet_unpack_series(coeffs_flat, root_order)
        if len(tree.tree_leaves(ss)) == 0:
            fx = jet_call(*ps)
            return [fx]

        primals1, series1 = func.jet(jet_call, ps, ss, is_tcoeff=False)
        return [primals1, *series1]

    if jet_order_explicit == "max" or jet_order_implicit == "max":
        order = ssm.prior.shape_info.num_derivatives + 1
    else:
        order_ex = root_order_ex + jet_order_explicit
        order_im = root_order_im + jet_order_implicit
        order = max(order_ex, order_im)
    return ssm.linearize.root(
        root_jet, root_order=order, jacobian=jacobian, nlstsq=nlstsq
    )


def constraint_jet_dae(
    differential,
    algebraic,
    *,
    ssm: ssm_impl.FactSsmImpl,
    jacobian=None,
    nlstsq=None,
    jet_order_differential: int | Literal["max"] = "max",
    jet_order_algebraic: int | Literal["max"] = "max",
):
    """Like `constraint_jet`, but for DAEs.

    The advantage of a dedicated DAE constraint is that algebraic and differential
    roots can enjoy different jet-orders, which increases accuracy.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This function is highly experimental and not safe to use.
        There is no guarantee that it works correctly (or at all).
        It might be deleted tomorrow and without any deprecation policy.

    """
    root_order_diff = _verify_vector_field_signature_and_parse_order(differential)
    root_order_alg = _verify_vector_field_signature_and_parse_order(algebraic)

    if jacobian is None:
        jacobian = jacobian_hutchinson_fwd()

    def root_jet(*tcoeffs_all, t):
        unravel = tree.ravel_pytree(tcoeffs_all[0])[1]

        fx1 = jet_evaluate(
            differential,
            tcoeffs_all,
            jet_order=jet_order_differential,
            root_order=root_order_diff,
            unravel=unravel,
            t=t,
        )
        fx2 = jet_evaluate(
            algebraic,
            tcoeffs_all,
            jet_order=jet_order_algebraic,
            root_order=root_order_alg,
            unravel=unravel,
            t=t,
        )

        return [fx1, fx2]

    def jet_evaluate(fun, tcoeffs_all, /, *, jet_order, root_order, unravel, t):
        # Flatten the root because jax.jet is a bit high maintenance :)
        def jet_call(*y):
            y_tree = [unravel(s) for s in y]
            fx = fun(*y_tree, t=t)
            return tree.ravel_pytree(fx)[0]

        if jet_order == "max":
            tcoeffs = tcoeffs_all
        else:
            jet_order_upper = len(tcoeffs_all) - root_order
            if jet_order < 0 or jet_order > jet_order_upper:
                msg = "The provided jet-order is incompatible with the root order."
                msg += f" Expected: 0 <= jet_order <= {jet_order_upper}."
                msg += f" Received: jet_order == {jet_order}."
                raise ValueError(msg)

            order = jet_order + root_order
            tcoeffs = tcoeffs_all[:order]

        flat = [tree.ravel_pytree(s)[0] for s in tcoeffs]
        ps, ss = diffeqjet.jet_unpack_series(flat, root_order)

        if len(tree.tree_leaves(ss)) == 0:
            fx = jet_call(*ps)
            return [fx]

        primals, series = func.jet(jet_call, ps, ss, is_tcoeff=False)
        return [primals, *series]

    if jet_order_differential == "max" or jet_order_algebraic == "max":
        order = ssm.prior.shape_info.num_derivatives + 1
    else:
        order_diff = root_order_diff + jet_order_differential
        order_alg = root_order_alg + jet_order_algebraic
        order = max(order_diff, order_alg)
    return ssm.linearize.root(
        root_jet, root_order=order, jacobian=jacobian, nlstsq=nlstsq
    )


@tree.register_dataclass
@structs.dataclass
class TaylorCoeffTarget(Generic[N]):
    """A probabilistic description of Taylor coefficients.

    Includes means, standard deviations, and marginals.
    Taylor coefficients are common solution targets
    for probabilistic differential equation solvers.
    """

    marginals: N
    """The full marginal distribution of the Taylor coefficient."""

    @property
    def mean(self):
        """A PyTree describing the standard deviation of the Taylor coefficient."""
        return self.marginals.mean_tree()

    @property
    def std(self):
        """A PyTree describing the mean of the Taylor coefficient."""
        return self.marginals.std_tree()


@tree.register_dataclass
@structs.dataclass
class MarkovSequence(Generic[N]):
    """A datastructure for Markov sequences as batches of joint distributions.

    This is the output type of smoother-based estimators.
    (Filter-based estimators do not return specialised types.)
    """

    marginal: N
    """The marginal distribution."""

    conditional: Any
    """The conditional distribution."""

    reverse: bool = structs.dataclass_field(metadata={"static": True})
    """The direction of factorisations."""

    @classmethod
    def from_grid(cls, init, discretize, *, grid, reverse: bool):
        marginal = init.marginals
        conditional = func.vmap(discretize)(np.diff(grid))
        return cls(marginal, conditional, reverse=reverse)

    def rescale_cholesky(self, factor, /):
        marg = self.marginal.rescale_cholesky(factor)
        cond = self.conditional.rescale_noise(factor)
        return MarkovSequence(marg, cond, reverse=self.reverse)

    def evaluate_marginals(self, *, ssm):
        """Extract the (time-)marginals from a Markov sequence.

        This is only needed in combination with smoothing-based strategies.
        """
        if self.marginal.mean.ndim == self.conditional.noise.mean.ndim:
            markov_seq = self._select_terminal()
            return markov_seq.evaluate_marginals(ssm=ssm)

        def step(x, cond):
            extrapolated = ssm.conditional.marginalise(x, cond)
            return extrapolated, extrapolated

        _, marginals = flow.scan(
            step, init=self.marginal, xs=self.conditional, reverse=self.reverse
        )

        if self.reverse:
            # Append the terminal marginal to the computed ones
            return tree.tree_array_append(marginals, self.marginal)

        return tree.tree_array_prepend(self.marginal, marginals)

    def evaluate_lml(
        self,
        u,
        *,
        model,
        ssm: ssm_impl.FactSsmImpl,
        average_pdfs: bool,
        solve_triu: Callable,
    ):
        assert self.reverse

        if self.marginal.mean.ndim == self.conditional.noise.mean.ndim:
            markov_seq = self._select_terminal()
            return markov_seq.evaluate_lml(
                u,
                model=model,
                ssm=ssm,
                average_pdfs=average_pdfs,
                solve_triu=solve_triu,
            )

        # Process the terminal value
        u0 = tree.tree_map(lambda s: s[-1], u)
        model0 = tree.tree_map(lambda s: s[-1], model)
        pdf0, updated = ssm.conditional.bayes_rule_and_logpdf_tree(
            u0, self.marginal, model0, solve_triu=solve_triu
        )

        # Process the remaining values
        def body(rv_and_logpdf, prior_and_observation_and_data):
            rv, logpdf, num_data = rv_and_logpdf
            prior, observation, data = prior_and_observation_and_data

            predicted = ssm.conditional.marginalise(rv, prior)

            logpdf_n, corrected = ssm.conditional.bayes_rule_and_logpdf_tree(
                data, predicted, observation, solve_triu=solve_triu
            )

            # The mean of the PDFs (as opposed to their sum) usually
            # leads to LML values that are more "human-readable"
            # (ie magnitude O(1) instead O(N)). This is technically not
            # a log-marginal-likelihood, but much nicer to work with when
            # optimising the loss.
            if average_pdfs:
                logpdf1 = (logpdf * num_data + logpdf_n) / (num_data + 1)
            else:
                logpdf = logpdf + logpdf_n

            return (corrected, logpdf1, num_data + 1), ()

        u1 = tree.tree_map(lambda s: s[:-1], u)
        model1 = tree.tree_map(lambda s: s[:-1], model)
        init = (updated, pdf0, 1)
        xs = (self.conditional, model1, u1)
        (_, pdf, _), _ = flow.scan(body, init=init, xs=xs, reverse=self.reverse)
        return pdf

    def _select_terminal(self):
        """Discard all intermediate filtering solutions from a Markov sequence.

        This function is useful to convert a smoothing-solution into a Markov sequence
        that is compatible with sampling or marginalisation.
        """
        init = tree.tree_map(lambda x: x[-1, ...], self.marginal)
        return MarkovSequence(init, self.conditional, reverse=self.reverse)

    def sample(self, key, *, ssm: ssm_impl.FactSsmImpl, shape: tuple = ()):
        """Sample from a Markov sequence."""
        # If the MarkovSequence carries unnecessary filtering marginals, remove them
        if self.marginal.mean.ndim == self.conditional.noise.mean.ndim:
            markov_seq = self._select_terminal()
            return markov_seq.sample(key, ssm=ssm, shape=shape)

        # If many samples are required, vmap over recursive calls to sample()
        if len(shape) > 0:
            n, *shape_remaining = shape
            keys = random.split(key, num=n)
            sample_ = func.partial(self.sample, ssm=ssm, shape=shape_remaining)
            return func.vmap(sample_)(keys)

        # Compute a single sample from the Markov sequence

        def body(smp0_and_key, cond):
            smp0, key = smp0_and_key
            predicted = ssm.conditional.apply_tree(smp0, cond)
            key, subkey = random.split(key, num=2)
            smp1 = predicted.sample_tree(subkey)
            return (smp1, key), smp1

        key, subkey = random.split(key, num=2)
        smp = self.marginal.sample_tree(subkey)
        _, smps = flow.scan(
            body, init=(smp, key), xs=self.conditional, reverse=self.reverse
        )

        # Currently, sampling is only implemented for reverse
        if self.reverse:
            return tree.tree_array_append(smps, smp)
        return tree.tree_array_prepend(smp, smps)


T = TypeVar("T", bound=MarkovSequence | ssm_impl.AbstractTreeNormal)
"""A type-variable to describe posterior distributions."""


@tree.register_dataclass
@structs.dataclass
class ProbabilisticSolution(Generic[N, T]):
    """A datastructure for probabilistic solutions of differential equations."""

    t: Array
    """The current time-step."""

    u: TaylorCoeffTarget[N]
    """The current ODE solution estimate."""

    solution_full: T
    """The current posterior estimate."""

    # Todo: merge 'output_scale' and 'auxiliary' and "fun_evals"?
    output_scale: Any
    """The current output scale."""

    num_steps: Array
    """The number of steps taken until the current point."""

    auxiliary: Any
    """Auxiliary states.

    For instance, random keys for Hutchinson-based
    diagonal linearisation in the correction,
    or running means of the MLE calibration.
    """

    fun_evals: Any
    """Function evaluations.

    Used to cache observation models between solver steps
    and error estimates.
    """


S = TypeVar(
    "S", bound=ProbabilisticSolution | MarkovSequence | ssm_impl.AbstractTreeNormal
)
"""A type-variable to describe interpolation results."""


@tree.register_dataclass
@structs.dataclass
class InterpResult(Generic[S]):
    """A datastructure to store interpolation results.

    To ensure correct adaptive time-stepping, it is important
    to distinguish step-from variables from interpolate-from variables.

    For some solvers, e.g. fixed-point-smoother-based ones,
    both stepping and interpolating variables are adjusted during interpolation.
    """

    step_from: S
    """The new 'step_from' field.

    At time `max(t, s1.t)`.
    Use this as the right-most reference state
    in future interpolations, or continue time-stepping from here.
    """

    interp_from: S
    """The new `interp_from` field.

    At time `t`. Use this as the left-most reference state
    in future interpolations.

    The difference between `interpolated` and `interp_from`
    is important around checkpoints:

    - `interpolated` belongs to the just-concluded time interval,
    - `interp_from` belongs to the to-be-started time interval.

    Concretely, this means that for fixed-point smoothers,
    `interp_from` has a unit backward model whereas `interpolated`
    remembers how to step back to the previous target location.
    """


class MarkovStrategy(Generic[T]):
    """An interface for estimation strategies in Markovian state-space models.

    Related:
    [`strategy_filter`](#probdiffeq.probdiffeq.strategy_filter),
    [`strategy_smoother_fixedpoint`](#probdiffeq.probdiffeq.strategy_smoother_fixedpoint),
    [`strategy_smoother_fixedinterval`](#probdiffeq.probdiffeq.strategy_smoother_fixedinterval).
    """

    def __init__(
        self,
        ssm: ssm_impl.FactSsmImpl,
        is_suitable_for_save_at: int,
        is_suitable_for_save_every_step: int,
        is_suitable_for_offgrid_marginals: int,
    ) -> None:
        self.ssm = ssm
        self.is_suitable_for_save_at = is_suitable_for_save_at
        self.is_suitable_for_save_every_step = is_suitable_for_save_every_step
        self.is_suitable_for_offgrid_marginals = is_suitable_for_offgrid_marginals

    def init_posterior(self, *, u: TaylorCoeffTarget) -> T:
        """Initialize a posterior distribution."""
        raise NotImplementedError

    def predict(self, posterior: T, *, transition) -> tuple[TaylorCoeffTarget, T]:
        """Make a prediction."""
        raise NotImplementedError

    def apply_updates(self, prediction: T, *, updates) -> tuple[TaylorCoeffTarget, T]:
        """Apply updates to a prediction."""
        raise NotImplementedError

    def interpolate(
        self, *, posterior_t0: T, posterior_t1: T, transition_t0_t, transition_t_t1
    ) -> tuple[tuple[TaylorCoeffTarget, T], InterpResult[T]]:
        """Interpolate between two points."""
        raise NotImplementedError

    def interpolate_at_t1(
        self, *, posterior_t1: T
    ) -> tuple[tuple[TaylorCoeffTarget, T], InterpResult[T]]:
        """Interpolate at a checkpoint."""
        raise NotImplementedError

    def finalize(self, *, posterior0: T, posterior: T, output_scale) -> T:
        """Finalize the posterior before returning a solution."""
        raise NotImplementedError


class ProbabilisticSolver:
    """An interface for probabilistic differential equation solvers.

    Related:
    [`solver`](#probdiffeq.probdiffeq.solver),
    [`solver_mle`](#probdiffeq.probdiffeq.solver_mle),
    [`solver_dynamic`](#probdiffeq.probdiffeq.solver_dynamic).

    """

    def __init__(
        self,
        *,
        strategy: MarkovStrategy,
        prior: Callable,
        constraint: Constraint,
        ssm: ssm_impl.FactSsmImpl,
        constraint_init: Constraint | None,
    ) -> None:
        self.ssm = ssm
        self.strategy = strategy
        self.prior = prior
        self.constraint = constraint
        self.constraint_init = constraint_init

    @property
    def is_suitable_for_offgrid_marginals(self):
        """Whether the solver admits offgrid-marginal calculation.

        Excludes fixed-point smoothers, for example.
        """
        return self.strategy.is_suitable_for_offgrid_marginals

    @property
    def is_suitable_for_save_at(self):
        """Whether the solver admits adaptive time-stepping with checkpoints.

        Excludes fixed-interval smoothers, for example.
        """
        return self.strategy.is_suitable_for_save_at

    @property
    def is_suitable_for_save_every_step(self):
        """Whether the solver admits adaptive time-stepping without checkpoints.

        Excludes fixed-point smoothers, for example.
        """
        return self.strategy.is_suitable_for_save_every_step

    def init(self, t, init: TaylorCoeffTarget, *, damp: float) -> ProbabilisticSolution:
        """Initialize the probabilistic solution."""
        raise NotImplementedError

    def step(self, state: ProbabilisticSolution, *, dt: float, damp: float):
        """Perform a step."""
        raise NotImplementedError

    def userfriendly_output(
        self, *, solution0: ProbabilisticSolution, solution: ProbabilisticSolution
    ) -> ProbabilisticSolution:
        """Make the solutions 'user-friendly'.

        This may include calibration, calculation of marginals, and other things.
        """
        raise NotImplementedError

    def offgrid_marginals(self, t, *, solution):
        """Compute off-grid marginals via jax.numpy.searchsorted.

        !!! warning
            The time-point 't' may not be an element in the solution grid.
            Otherwise, anything can happen and the solution will be incorrect.
            At the moment, we do not check this.

        !!! warning
            The time-point 't' must also be strictly in (t0, t1).
            It must not lie outside the interval, and it must not coincide
            with the interval boundaries.
            At the moment, we do not check this.
        """
        assert t.shape == solution.t[0].shape
        # side="left" and side="right" are equivalent
        # because we _assume_ that the point sets are disjoint.
        index = np.searchsorted(solution.t, t)

        # Extract the LHS

        def _extract_previous(pytree):
            return tree.tree_map(lambda s: s[index - 1, ...], pytree)

        posterior_t0 = _extract_previous(solution.solution_full)
        t0 = _extract_previous(solution.t)

        # Extract the RHS

        def _extract(pytree):
            return tree.tree_map(lambda s: s[index, ...], pytree)

        t1 = _extract(solution.t)
        output_scale = _extract(solution.output_scale)

        # Take the marginals because we need the t1-value to be informed
        # about *all* datapoints
        u_t1 = _extract(solution.u)
        _, posterior_t1 = self.strategy.init_posterior(u=u_t1)

        if not self.is_suitable_for_offgrid_marginals:
            raise NotImplementedError

        transition_t0_t = self.prior(t - t0, output_scale)
        transition_t_t1 = self.prior(t1 - t, output_scale)
        (estimate, _posterior), _interp_res = self.strategy.interpolate(
            posterior_t0=posterior_t0,
            posterior_t1=posterior_t1,
            transition_t0_t=transition_t0_t,
            transition_t_t1=transition_t_t1,
        )
        return estimate

    def interpolate(
        self, *, t, interp_from: ProbabilisticSolution, interp_to: ProbabilisticSolution
    ):
        """Interpolate between two solution objects."""
        # Domain is (t0, t1]; thus, take the output scale from interp_to
        output_scale = interp_to.output_scale
        transition_t0_t = self.prior(t - interp_from.t, output_scale)
        transition_t_t1 = self.prior(interp_to.t - t, output_scale)

        # Interpolate
        tmp = self.strategy.interpolate(
            posterior_t0=interp_from.solution_full,
            posterior_t1=interp_to.solution_full,
            transition_t0_t=transition_t0_t,
            transition_t_t1=transition_t_t1,
        )
        (estimate, interpolated), step_and_interpolate_from = tmp

        step_from = ProbabilisticSolution(
            t=interp_to.t,
            # New:
            solution_full=step_and_interpolate_from.step_from,
            # Old:
            u=interp_to.u,
            output_scale=interp_to.output_scale,
            auxiliary=interp_to.auxiliary,
            num_steps=interp_to.num_steps,
            fun_evals=interp_to.fun_evals,
        )

        interpolated = ProbabilisticSolution(
            t=t,
            # New:
            solution_full=interpolated,
            u=estimate,
            # Taken from the rhs point
            output_scale=interp_to.output_scale,
            auxiliary=interp_to.auxiliary,
            num_steps=interp_to.num_steps,
            fun_evals=interp_to.fun_evals,
        )

        interp_from = ProbabilisticSolution(
            t=t,
            # New:
            solution_full=step_and_interpolate_from.interp_from,
            # Old:
            u=interp_from.u,
            output_scale=interp_from.output_scale,
            auxiliary=interp_from.auxiliary,
            num_steps=interp_from.num_steps,
            fun_evals=interp_from.fun_evals,
        )

        interp_res = InterpResult(step_from=step_from, interp_from=interp_from)
        return interpolated, interp_res

    def interpolate_at_t1(
        self, *, t, interp_from: ProbabilisticSolution, interp_to: ProbabilisticSolution
    ):
        """Interpolate the solution near a checkpoint."""
        del t
        tmp = self.strategy.interpolate_at_t1(posterior_t1=interp_to.solution_full)
        (estimate, interpolated), step_and_interpolate_from = tmp

        prev = ProbabilisticSolution(
            t=interp_to.t,
            # New
            solution_full=step_and_interpolate_from.interp_from,
            # Old
            u=interp_from.u,
            output_scale=interp_from.output_scale,
            auxiliary=interp_from.auxiliary,
            num_steps=interp_from.num_steps,
            fun_evals=interp_from.fun_evals,
        )
        sol = ProbabilisticSolution(
            t=interp_to.t,
            # New:
            solution_full=interpolated,
            u=estimate,
            # Old:
            output_scale=interp_to.output_scale,
            auxiliary=interp_to.auxiliary,
            num_steps=interp_to.num_steps,
            fun_evals=interp_to.fun_evals,
        )
        acc = ProbabilisticSolution(
            t=interp_to.t,
            # New:
            solution_full=step_and_interpolate_from.step_from,
            # Old
            u=interp_to.u,
            output_scale=interp_to.output_scale,
            auxiliary=interp_to.auxiliary,
            num_steps=interp_to.num_steps,
            fun_evals=interp_to.fun_evals,
        )
        return sol, InterpResult(step_from=acc, interp_from=prev)


def ssm_taylor(
    tcoeffs: C,
    *,
    # Which of the Taylor coefficients are exact
    is_exact: C | bool = True,
    inexact_eps: float = 1e-6,  # a small value
    # The state-space model factorisation
    ssm_fact: Literal["dense", "isotropic", "blockdiag"] = "dense",  # noqa: F821
    # How many extra derivatives to model in the state-space
    diffuse_derivatives: int = 0,
    diffuse_eps: float = 1.0,  # a large value
):
    """Initialize a state-space model over Taylor coefficients."""
    tcoeffs_std = _tcoeffs_std_from_differential_variables(
        tcoeffs, is_exact=is_exact, inexact_eps=inexact_eps, ssm_fact=ssm_fact
    )

    return ssm_taylor_diffuse(
        tcoeffs,
        tcoeffs_std,
        ssm_fact=ssm_fact,
        diffuse_derivatives=diffuse_derivatives,
        diffuse_eps=diffuse_eps,
    )


def _tcoeffs_std_from_differential_variables(
    tcoeffs, *, ssm_fact, is_exact, inexact_eps
):
    # Decide the standard deviation template based on the factorisations
    if ssm_fact in ["dense", "blockdiag"]:
        leaves = tree.tree_leaves(tcoeffs)
        std_per_leaf = np.zeros_like(leaves[0])
    elif ssm_fact in ["isotropic"]:
        std_per_leaf = np.zeros(())
    else:
        msg = f"ssm_fact={ssm_fact} is unknown."
        raise ValueError(msg)

    # Infer the expected standard-deviation tree structure
    leaves, structure = tree.tree_flatten(tcoeffs)
    std_template = tree.tree_unflatten(structure, [std_per_leaf for _ in leaves])

    # If 'is_exact' is a boolean, keep things simple
    if isinstance(is_exact, bool):
        if is_exact:
            return tree.tree_map(np.zeros_like, std_template)
        return tree.tree_map(lambda s: inexact_eps * np.ones_like(s), std_template)

    is_exact = tree.tree_map(np.asarray, is_exact)

    # Before using is_exact, verify it has the correct structure and shape
    try:

        def shape_equal(A, B):
            return tree.tree_map(lambda a, b: a.shape == b.shape, A, B)

        assert tree.tree_all(shape_equal(is_exact, std_template))
    except (ValueError, AssertionError) as err:
        msg = "Input 'is_exact' has the wrong PyTree structure."
        msg += f" Expected: {tree.tree_map(np.shape, std_template)}."
        msg += f" Received: {tree.tree_map(np.shape, is_exact)}."
        raise ValueError(msg) from err

    # Wherever is_exact is True, initialize with zeros.
    # Elsewhere, initialize with a small positivec value.

    def std_init(s: Array) -> Array:
        if s.dtype != np.dtype(bool):
            msg = "Boolean entries expected in `is_exact`."
            msg += f" Received: dtype={np.dtype(s)}"
            raise TypeError(msg)
        return np.where(s, 0.0, inexact_eps)

    return tree.tree_map(std_init, is_exact)


def ssm_taylor_diffuse(
    tcoeffs_mean: C,
    tcoeffs_std: C,
    *,
    # The state-space model factorisation
    ssm_fact: Literal["dense", "isotropic", "blockdiag"] = "dense",  # noqa: F821
    # How many extra derivatives to model in the state-space
    diffuse_derivatives: int = 0,
    diffuse_eps: float = 1.0,
):
    """Initialize a diffuse state-space model for Taylor coefficients.

    The diffuse process has a nonzero initial standard deviation.
    This is typically used to get more visually-pleasing uncertainties and gain
    numerical robustness for high-order solvers in low precision arithmetic.

    Outside of these cases, use the usual Taylor-state-space-model process.
    """
    # Add derivatives to the Taylor coefficients.
    # Warning: This destroys pytree structure in the tcoeffs and the
    # resulting pytree will always be a list (for now at least)
    if diffuse_derivatives > 0:
        tcoeffs_mean, tcoeffs_std = _add_diffuse_derivatives(
            tcoeffs_mean,
            tcoeffs_std,
            diffuse_derivatives=diffuse_derivatives,
            diffuse_eps=diffuse_eps,
        )

    # Choose a state-space model factorisation
    match ssm_fact:
        case "dense":
            marginal, ssm = ssm_impl.FactSsmImpl.from_tcoeffs_dense(
                tcoeffs_mean, tcoeffs_std
            )
        case "blockdiag":
            marginal, ssm = ssm_impl.FactSsmImpl.from_tcoeffs_blockdiag(
                tcoeffs_mean, tcoeffs_std
            )
        case "isotropic":
            marginal, ssm = ssm_impl.FactSsmImpl.from_tcoeffs_isotropic(
                tcoeffs_mean, tcoeffs_std
            )
        case _:
            msg = f"Factorisation ssm_fact='{ssm_fact}' unknown. "
            msg += "Choose one out of {'dense', 'isotropic', 'blockdiag'}."
            raise ValueError(msg)

    # Return the target
    target = TaylorCoeffTarget(marginal)
    return target, ssm


def _add_diffuse_derivatives(
    tcoeffs_mean, tcoeffs_std, *, diffuse_eps, diffuse_derivatives
):
    zeros = tree.tree_map(np.zeros_like, tcoeffs_mean[0])
    tcoeffs_mean = [*tcoeffs_mean, *[zeros for _ in range(diffuse_derivatives)]]

    unknowns = tree.tree_map(lambda s: diffuse_eps * np.ones_like(s), tcoeffs_std[0])
    tcoeffs_std = [*tcoeffs_std, *[unknowns for _ in range(diffuse_derivatives)]]
    return tcoeffs_mean, tcoeffs_std


def prior_wiener_integrated(
    *, ssm: ssm_impl.FactSsmImpl, output_scale: Array | None = None
):
    """Construct an integrated Wiener process prior."""
    return ssm.prior.transition_wiener_integrated(base_scale=output_scale)


def prior_ornstein_uhlenbeck_integrated(
    linop: Callable, /, *, ssm: ssm_impl.FactSsmImpl, output_scale: Array | None = None
):
    """Construct an integrated Ornstein-Uhlenbeck prior."""

    def vf_linear(*tcoeffs):
        return linop(tcoeffs[-1])

    return ssm.prior.transition_exponential(
        vf_linear=vf_linear, base_scale=output_scale
    )


def prior_exponential(
    vf_linear: Callable,
    /,
    *,
    ssm: ssm_impl.FactSsmImpl,
    output_scale: Array | None = None,
):
    """Construct an exponential integrator prior.

    According to https://arxiv.org/abs/2305.14978, but following the numerical
    methods from https://arxiv.org/abs/2310.13462.
    """
    # TODO: offer a "jacobian" option to enable isotropic and blockdiag implementations?
    prior_order = _verify_ioup_signature_and_parse_order(vf_linear)
    if prior_order != ssm.prior.shape_info.num_derivatives + 1:
        msg = f"""The exponential prior does not match the Taylor coefficients in the SSM.

        Concretely:

        - For two Taylor coefficients, we expect `f(u, du, /)`.
        - For three Taylor coefficients, we expect `f(u, du, ddu, /)`.
        - For two Taylor coefficients, we expect `f(u, du, ddu, dddu, /)`.

        and so on. The passed dynamics correspond to **{prior_order}** Taylor
        coefficients, whereas the state-space model includes **{ssm.prior.shape_info.num_derivatives + 1}**
        Taylor coeffients.
        """
        raise TypeError(msg)

    return ssm.prior.transition_exponential(
        vf_linear=vf_linear, base_scale=output_scale
    )


def _verify_ioup_signature_and_parse_order(vf) -> int:
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


class strategy_smoother_fixedinterval(MarkovStrategy[MarkovSequence]):
    """Construct a fixed-interval smoother.

    Use this strategy for fixed steps.
    For adaptive steps, consider using a fixed-point smoother instead.


    Related:
    [`MarkovStrategy`](#probdiffeq.probdiffeq.MarkovStrategy).
    """

    def __init__(self, ssm) -> None:
        super().__init__(
            ssm=ssm,
            is_suitable_for_save_at=False,
            is_suitable_for_save_every_step=True,
            is_suitable_for_offgrid_marginals=True,
        )

    def init_posterior(self, *, u: TaylorCoeffTarget):
        cond = self.ssm.prior.identity()
        posterior = MarkovSequence(marginal=u.marginals, conditional=cond, reverse=True)
        return u, posterior

    def predict(
        self, posterior: MarkovSequence, *, transition
    ) -> tuple[TaylorCoeffTarget, MarkovSequence]:
        marginals, cond = self.ssm.conditional.revert(
            posterior.marginal, transition, solve_triu=linalg.solve_triu
        )
        posterior = MarkovSequence(
            marginal=marginals, conditional=cond, reverse=posterior.reverse
        )

        estimate = TaylorCoeffTarget(marginals)
        return estimate, posterior

    def apply_updates(self, prediction, *, updates):
        posterior = MarkovSequence(
            updates, prediction.conditional, reverse=prediction.reverse
        )
        marginals = updates
        estimate = TaylorCoeffTarget(marginals)
        return estimate, posterior

    def finalize(
        self, *, posterior0: MarkovSequence, posterior: MarkovSequence, output_scale
    ):
        prototype = self.ssm.prior.prototype_output_scale_calibrated()
        assert output_scale.shape == prototype.shape
        posterior0 = posterior0.rescale_cholesky(output_scale)
        posterior = posterior.rescale_cholesky(output_scale)

        # Marginalise
        marginals = posterior.evaluate_marginals(ssm=self.ssm)

        # Prepend the initial condition to the filtering distributions
        init = tree.tree_array_prepend(posterior0.marginal, posterior.marginal)
        posterior = MarkovSequence(
            marginal=init, conditional=posterior.conditional, reverse=posterior.reverse
        )

        # Extract targets
        estimate = TaylorCoeffTarget(marginals)
        return estimate, posterior

    def interpolate(
        self,
        *,
        posterior_t0: MarkovSequence,
        posterior_t1: MarkovSequence,
        transition_t0_t,
        transition_t_t1,
    ):
        """Interpolate between two Markov sequences.

        Here is how a smoother interpolates:

        - Extrapolate from t0 to t, which gives the filtering distribution
          and the backward transition from t to t0.
        - Extrapolate from t to t1, which gives another filtering distribution
          and the backward transition from t1 to t.
        - Apply the new t1-to-t backward transition to the posterior
          to compute the interpolation.

        This intermediate result is informed about its "right-hand side" datum.
        Note how in probdiffeq, all solver subintervals include their right-hand
        side time-point: in other words, they are (t0, t1].

        Specifically, interpolation is not equal to computing offgrid marginals.
        Interpolation always assumes that the current subinterval is the right-most
        subinterval, which is the case during the forward pass.
        After the simulation, if there are observations > t1,
        which happens when computing offgrid-marginals, do not use `interpolate()`.
        """
        # Extrapolate from t0 to t, and from t to t1.

        _, extrapolated_t = self.predict(
            posterior=posterior_t0, transition=transition_t0_t
        )
        _, extrapolated_t1 = self.predict(
            posterior=extrapolated_t, transition=transition_t_t1
        )

        # Marginalise backwards from t1 to t to obtain the interpolated solution.
        marginal_t1 = posterior_t1.marginal
        conditional_t1_to_t = extrapolated_t1.conditional
        rv_at_t = self.ssm.conditional.marginalise(marginal_t1, conditional_t1_to_t)
        solution_at_t = MarkovSequence(
            rv_at_t, extrapolated_t.conditional, reverse=extrapolated_t.reverse
        )

        # The state at t1 gets a new backward model;
        # (it must remember how to get back to t, not to t0).
        solution_at_t1 = MarkovSequence(
            marginal_t1, conditional_t1_to_t, reverse=extrapolated_t.reverse
        )
        interp_res = InterpResult(step_from=solution_at_t1, interp_from=solution_at_t)

        # Extract targets
        marginals = solution_at_t.marginal
        estimate = TaylorCoeffTarget(marginals)
        return (estimate, solution_at_t), interp_res

    def interpolate_at_t1(self, posterior_t1):
        marginals = posterior_t1.marginal
        estimate = TaylorCoeffTarget(marginals)

        interp_res = InterpResult(step_from=posterior_t1, interp_from=posterior_t1)
        return (estimate, posterior_t1), interp_res


class strategy_filter(MarkovStrategy):
    """Construct a filter.

    Filters work with all timestepping strategies.
    However, the uncertainties are not informed by the full
    timeseries, which makes them visually less pleasing.
    Filter solutions also do not admit computing log-marginal
    likelihoods or joint sampling from the posterior distribution.
    For these use-cases, use smoothers instead.

    Related:
    [`MarkovStrategy`](#probdiffeq.probdiffeq.MarkovStrategy).
    """

    def __init__(self, ssm) -> None:
        super().__init__(
            ssm=ssm,
            is_suitable_for_save_at=True,
            is_suitable_for_save_every_step=True,
            is_suitable_for_offgrid_marginals=True,
        )

    def init_posterior(self, *, u: TaylorCoeffTarget):
        return u, u.marginals

    def predict(self, posterior: T, *, transition) -> tuple[TaylorCoeffTarget, T]:
        marginals = self.ssm.conditional.marginalise(posterior, transition)
        estimate = TaylorCoeffTarget(marginals)
        return estimate, marginals

    def apply_updates(self, prediction, *, updates):
        del prediction
        marginals = updates
        estimate = TaylorCoeffTarget(marginals)
        return estimate, marginals

    def finalize(self, *, posterior0, posterior, output_scale):
        expected = self.ssm.prior.prototype_output_scale_calibrated()
        assert output_scale.shape == expected.shape

        # No rescaling because no calibration at the initial step
        posterior0 = posterior0.rescale_cholesky(output_scale)

        # Calibrate
        posterior = posterior.rescale_cholesky(output_scale)

        # Stack
        posterior = tree.tree_array_prepend(posterior0, posterior)

        marginals = posterior
        estimate = TaylorCoeffTarget(marginals)
        return estimate, posterior

    def interpolate(
        self, *, posterior_t0, posterior_t1, transition_t0_t, transition_t_t1
    ):
        del transition_t_t1
        _, interpolated = self.predict(
            posterior=posterior_t0, transition=transition_t0_t
        )
        marginals = interpolated
        estimate = TaylorCoeffTarget(marginals)
        interp_res = InterpResult(step_from=posterior_t1, interp_from=interpolated)
        return (estimate, interpolated), interp_res

    def interpolate_at_t1(self, *, posterior_t1):
        marginals = posterior_t1
        estimate = TaylorCoeffTarget(marginals)

        interp_res = InterpResult(step_from=posterior_t1, interp_from=posterior_t1)
        return (estimate, posterior_t1), interp_res


class strategy_smoother_fixedpoint(MarkovStrategy[MarkovSequence]):
    r"""Construct a fixedpoint-smoother.

    Fixed-point smoothers are the method of choice for adaptive
    time-stepping in probabilistic differential equation solvers.

    Concretely, we implement the fixedpoint smoother by Krämer (2025a).
    Applied to probabilistic solvers, this leads to the algorithm
    by Krämer (2025b). Please consider citing both papers if you use
    fixed-point smoothers and ODE solvers in your work.


    ??? note "BibTex for Krämer (2025a)"
        ```bibtex
        @article{kramer2025numerically,
            title={Numerically Robust Fixed-Point Smoothing Without State Augmentation},
            author={Kr{\"a}mer, Nicholas},
            year={2025},
            journal={Transactions on Machine Learning Research}
        }
        ```

    ??? note "BibTex for Krämer (2025b)"
        ```bibtex
            @InProceedings{kramer2024adaptive,
            title     = {Adaptive Probabilistic ODE Solvers Without Adaptive
            Memory Requirements},
            author    = {Kr\"{a}mer, Nicholas},
            booktitle = {Proceedings of the First International Conference
            on Probabilistic Numerics},
            pages     = {12--24},
            year      = {2025},
            editor    = {Kanagawa, Motonobu and Cockayne, Jon and Gessner,
            Alexandra and Hennig, Philipp},
            volume    = {271},
            series    = {Proceedings of Machine Learning Research},
            publisher = {PMLR},
            url       = {https://proceedings.mlr.press/v271/kramer25a.html}
        }
        ```
    Related:
    [`MarkovStrategy`](#probdiffeq.probdiffeq.MarkovStrategy).


    """

    def __init__(self, ssm) -> None:
        super().__init__(
            ssm=ssm,
            is_suitable_for_save_at=True,
            is_suitable_for_save_every_step=False,
            is_suitable_for_offgrid_marginals=False,
        )

    def init_posterior(self, *, u):
        cond = self.ssm.prior.identity()
        posterior = MarkovSequence(u.marginals, cond, reverse=True)
        return u, posterior

    def predict(
        self, posterior: MarkovSequence, *, transition
    ) -> tuple[TaylorCoeffTarget, MarkovSequence]:
        rv = posterior.marginal
        bw0 = posterior.conditional
        marginals, cond = self.ssm.conditional.revert(
            rv, transition, solve_triu=linalg.solve_triu
        )
        cond = self.ssm.conditional.merge(bw0, cond)
        predicted = MarkovSequence(marginals, cond, reverse=posterior.reverse)

        estimate = TaylorCoeffTarget(marginals)
        return estimate, predicted

    def apply_updates(self, prediction: MarkovSequence, *, updates):
        posterior = MarkovSequence(
            updates, prediction.conditional, reverse=prediction.reverse
        )
        marginals = updates
        estimate = TaylorCoeffTarget(marginals)
        return estimate, posterior

    def finalize(
        self, *, posterior0: MarkovSequence, posterior: MarkovSequence, output_scale
    ):
        expected = self.ssm.prior.prototype_output_scale_calibrated()
        assert output_scale.shape == expected.shape
        posterior0 = posterior0.rescale_cholesky(output_scale)
        posterior = posterior.rescale_cholesky(output_scale)

        # Marginalise
        marginals = posterior.evaluate_marginals(ssm=self.ssm)

        # Prepend the initial condition to the filtering distributions
        init = tree.tree_array_prepend(posterior0.marginal, posterior.marginal)
        posterior = MarkovSequence(
            marginal=init, conditional=posterior.conditional, reverse=posterior.reverse
        )

        # Extract targets
        estimate = TaylorCoeffTarget(marginals)
        return estimate, posterior

    def interpolate_at_t1(self, *, posterior_t1: MarkovSequence):
        cond_identity = self.ssm.prior.identity()
        resume_from = MarkovSequence(
            posterior_t1.marginal,
            conditional=cond_identity,
            reverse=posterior_t1.reverse,
        )
        interp_res = InterpResult(step_from=resume_from, interp_from=resume_from)

        interpolated = posterior_t1
        marginals = interpolated.marginal
        estimate = TaylorCoeffTarget(marginals)
        return (estimate, interpolated), interp_res

    def interpolate(
        self,
        *,
        posterior_t0: MarkovSequence,
        posterior_t1: MarkovSequence,
        transition_t0_t,
        transition_t_t1,
    ):
        """Interpolate between two Markov sequences.

        Assuming `state_t0` has seen $n$ collocation points,
        and `state_t1` has seen $n+1$ collocation points,
        then interpolation at time $t$ is computed as follows:

        1. Extrapolate from $t_0$ to $t$. This yields:
            - the marginal at $t$ given $n$ observations.
            - the backward transition from $t$ to $t_0$ given $n$ observations.

        2. Extrapolate from $t$ to $t_1$. This yields:
            - the marginal at $t_1$ given $n$ observations
              (in contrast,`state_t1` has seen $n+1$ observations)
            - the backward transition from $t_1$ to $t$ given $n$ observations.

        3. Apply the backward transition from $t_1$ to $t$
        to the marginal inside `state_t1`
        to obtain the marginal at $t$ given $n+1$ observations. Similarly,
        the interpolated solution inherits all auxiliary info from the $t_1$ state.

        ---------------------------------------------------------------------

        All comments from fixed-interval smoother interpolation apply.

        ---------------------------------------------------------------------

        Difference to standard smoother interpolation:

        In the fixed-point smoother, backward transitions are modified
        to ensure that future operations remain correct.
        Denote the location of the fixed-point with $t_f$. Then,
        the backward transition at $t$ is merged with that at $t_0$.
        This preserves knowledge of how to move from $t$ to $t_f$.

        Then, `t` becomes the new fixed-point location. To ensure
        that future operations "find their way back to $t$":

        - Subsequent interpolations do not continue from the raw
        interpolated value. Instead, they continue from a nearly
        identical state where the backward transition is replaced
        by the identity.

        - Subsequent solver steps do not continue from the initial $t_1$
        state. Instead, they continue from a version whose backward
        model is replaced with the `t-to-t1` transition.


        ---------------------------------------------------------------------

        As a result, each interpolation must return three distinct states:

        1. the interpolated solution,
        2. the state to continue interpolating from,
        3. the state to continue solver stepping from.

        These are intentionally different in the fixed-point smoother.
        """
        # Note to myself: Don't attempt to remove any of them.
        # They're all important. You will break the code (again) :).

        # Extrapolate from t0 to t, and from t to t1.
        # This yields all building blocks.
        _, extrapolated_t = self.predict(
            posterior=posterior_t0, transition=transition_t0_t
        )
        conditional_id = self.ssm.prior.identity()
        previous_new = MarkovSequence(
            extrapolated_t.marginal, conditional_id, reverse=extrapolated_t.reverse
        )
        _, extrapolated_t1 = self.predict(
            posterior=previous_new, transition=transition_t_t1
        )

        # Marginalise from t1 to t to obtain the interpolated solution.
        marginal_t1 = posterior_t1.marginal
        conditional_t1_to_t = extrapolated_t1.conditional
        rv_at_t = self.ssm.conditional.marginalise(marginal_t1, conditional_t1_to_t)

        # Return the right combination of marginals and conditionals.
        interpolated = MarkovSequence(
            rv_at_t, extrapolated_t.conditional, reverse=extrapolated_t.reverse
        )
        step_from = MarkovSequence(
            posterior_t1.marginal,
            conditional=conditional_t1_to_t,
            reverse=posterior_t1.reverse,
        )
        interp_res = InterpResult(step_from=step_from, interp_from=previous_new)

        marginals = interpolated.marginal
        estimate = TaylorCoeffTarget(marginals)
        return (estimate, interpolated), interp_res


class solver_mle(ProbabilisticSolver):
    """Create a solver that uses maximum-likelihood calibration for the output scale.

    Related:
    [`ProbabilisticSolver`](#probdiffeq.probdiffeq.ProbabilisticSolver).

    """

    def __init__(
        self,
        *,
        constraint: Constraint,
        prior: Callable,
        ssm: ssm_impl.FactSsmImpl,
        strategy: MarkovStrategy,
        constraint_init: Constraint | None = None,
        correct_asymptotic_underconfidence: bool = True,
    ) -> None:
        super().__init__(
            strategy=strategy,
            ssm=ssm,
            prior=prior,
            constraint=constraint,
            constraint_init=constraint_init,
        )
        self.correct_asymptotic_underconfidence = correct_asymptotic_underconfidence

    def init(self, t, u: TaylorCoeffTarget, *, damp) -> ProbabilisticSolution:
        u_pred, prediction = self.strategy.init_posterior(u=u)
        cstate = self.constraint.init_linearization()

        prototype = self.ssm.prior.prototype_output_scale_calibrated()
        output_scale_prior = np.ones_like(prototype)

        # Update
        lin_fun = func.partial(self.constraint.linearize, damp=damp, t=t)
        fx, _cstate = func.eval_shape(lin_fun, u_pred.marginals, cstate)
        fx = tree.tree_map(np.zeros_like, fx)

        if self.constraint_init is not None:
            cstate_init = self.constraint_init.init_linearization()
            fx_init, _cstate = self.constraint_init.linearize(
                u_pred.marginals, cstate_init, damp=damp, t=t
            )
            zeros = tree.tree_map(np.zeros_like, fx_init.noise.mean_tree())
            tmp = self.ssm.conditional.bayes_rule_and_residual_white_rms_tree(
                zeros, u_pred.marginals, fx_init, solve_triu=linalg.lstsq_svd
            )
            output_scale_running, updates = tmp

            u, posterior = self.strategy.apply_updates(prediction, updates=updates)
            num_data = 1.0

        else:
            u, posterior = u_pred, prediction

            output_scale_running = np.zeros_like(output_scale_prior)
            num_data = 0.0

        auxiliary = (cstate, output_scale_running, num_data)

        return ProbabilisticSolution(
            t=t,
            u=u,
            solution_full=posterior,
            auxiliary=auxiliary,
            output_scale=output_scale_prior,
            num_steps=0,
            fun_evals=fx,
        )

    def step(self, state, *, dt: float, damp: float):
        # Discretize
        output_scale = np.ones_like(self.ssm.prior.prototype_output_scale_calibrated())
        transition = self.prior(dt, output_scale)

        # Predict
        u, prediction = self.strategy.predict(
            posterior=state.solution_full, transition=transition
        )

        # Linearize
        (lin_state, output_scale_running, num_data) = state.auxiliary
        fx, cstate = self.constraint.linearize(
            u.marginals, state=lin_state, damp=damp, t=state.t + dt
        )

        # Do the full correction step
        zeros = tree.tree_map(np.zeros_like, fx.noise.mean_tree())
        new_term, updates = self.ssm.conditional.bayes_rule_and_residual_white_rms_tree(
            zeros, u.marginals, fx, solve_triu=linalg.solve_triu
        )
        u, posterior = self.strategy.apply_updates(
            prediction=prediction, updates=updates
        )

        # Calibrate the output scale: c^2 = w_1 * a^2 + w_2 * b^2
        x1 = np.sqrt(num_data / (num_data + 1)) * output_scale_running
        x2 = np.sqrt(1 / (num_data + 1)) * new_term
        output_scale_running = np.hypot(x1, x2)

        # Return the state
        auxiliary = (cstate, output_scale_running, num_data + 1)
        return ProbabilisticSolution(
            t=state.t + dt,
            u=u,
            solution_full=posterior,
            output_scale=state.output_scale,
            auxiliary=auxiliary,
            num_steps=state.num_steps + 1,
            fun_evals=fx,
        )

    def userfriendly_output(
        self, *, solution0: ProbabilisticSolution, solution: ProbabilisticSolution
    ) -> ProbabilisticSolution:
        assert solution.t.ndim > 0

        # This is the MLE solver, so we take the calibrated scale
        _, output_scale, _ = solution.auxiliary
        ones = np.ones_like(output_scale)
        output_scale = output_scale[-1]

        # Improve the calibration like in other Gaussian process models.
        #   ODE priors are generally not as smooth as the ODE solutions,
        #   which means that their uncertainty is often a bit too large.
        #   See e.g. the "asymptotic underconfidence" derivations
        #   in https://arxiv.org/abs/2001.10965
        if self.correct_asymptotic_underconfidence:
            output_scale = output_scale / np.sqrt(solution.num_steps[-1])

        # Finalize the solution with the calibrated output scale
        init = solution0.solution_full
        posterior = solution.solution_full
        estimate, posterior = self.strategy.finalize(
            posterior0=init, posterior=posterior, output_scale=output_scale
        )

        output_scale = ones * output_scale[None, ...]
        ts = np.concatenate([solution0.t[None], solution.t])
        return ProbabilisticSolution(
            t=ts,
            u=estimate,
            solution_full=posterior,
            output_scale=output_scale,
            num_steps=solution.num_steps,
            auxiliary=solution.auxiliary,
            fun_evals=solution.fun_evals,
        )


class solver_dynamic(ProbabilisticSolver):
    """Create a solver that calibrates the output scale dynamically.

    Related:
    [`ProbabilisticSolver`](#probdiffeq.probdiffeq.ProbabilisticSolver).
    """

    def __init__(
        self,
        *,
        strategy: MarkovStrategy,
        prior: Callable,
        constraint: Constraint,
        ssm: ssm_impl.FactSsmImpl,
        constraint_init: Constraint | None = None,
        re_linearize_after_calibration=False,
    ) -> None:
        super().__init__(
            strategy=strategy,
            ssm=ssm,
            prior=prior,
            constraint=constraint,
            constraint_init=constraint_init,
        )
        self.re_linearize_after_calibration = re_linearize_after_calibration

    def init(self, t, u, *, damp) -> ProbabilisticSolution:
        u_pred, prediction = self.strategy.init_posterior(u=u)
        lin_state = self.constraint.init_linearization()

        output_scale = np.ones_like(self.ssm.prior.prototype_output_scale_calibrated())
        lin_fun = func.partial(self.constraint.linearize, damp=damp, t=t)
        fx, _lin_state = func.eval_shape(lin_fun, u_pred.marginals, lin_state)
        fx = tree.tree_map(np.zeros_like, fx)

        if self.constraint_init is not None:
            cstate_init = self.constraint_init.init_linearization()
            fx_init, _cstate = self.constraint_init.linearize(
                u_pred.marginals, cstate_init, damp=damp, t=t
            )
            zeros = tree.tree_map(np.zeros_like, fx_init.noise.mean_tree())
            updates = self.ssm.conditional.bayes_rule_tree(
                zeros, u_pred.marginals, fx_init, solve_triu=linalg.lstsq_svd
            )
            u, posterior = self.strategy.apply_updates(prediction, updates=updates)
        else:
            u, posterior = u_pred, prediction
        return ProbabilisticSolution(
            t=t,
            u=u,
            solution_full=posterior,
            auxiliary=lin_state,
            output_scale=output_scale,
            num_steps=0,
            fun_evals=fx,
        )

    def step(self, state: ProbabilisticSolution, *, dt: float, damp: float):
        lin_state = state.auxiliary

        # Calibrate the output scale
        ones = np.ones_like(self.ssm.prior.prototype_output_scale_calibrated())
        transition = self.prior(dt, ones)
        mean = state.u.marginals.mean_tree()
        u = self.ssm.conditional.apply_tree(mean, transition)

        # Linearize

        fx, lin_state = self.constraint.linearize(
            u, state=lin_state, damp=damp, t=state.t + dt
        )
        observed = self.ssm.conditional.marginalise(u, fx)
        zeros = tree.tree_map(np.zeros_like, fx.noise.mean_tree())
        output_scale = observed.residual_white_rms_tree(zeros)

        # Do the full extrapolation with the calibrated output scale
        # (Includes re-discretisation)
        transition = self.prior(dt, output_scale)
        u, prediction = self.strategy.predict(
            state.solution_full, transition=transition
        )

        # Relinearize
        if self.re_linearize_after_calibration:
            fx, lin_state = self.constraint.linearize(
                u.marginals, state=lin_state, damp=damp, t=state.t + dt
            )

        # Complete the update
        _, reverted = self.ssm.conditional.revert(
            u.marginals, fx, solve_triu=linalg.solve_triu
        )
        updates = reverted.noise
        u, posterior = self.strategy.apply_updates(prediction, updates=updates)

        # Return solution
        return ProbabilisticSolution(
            t=state.t + dt,
            u=u,
            solution_full=posterior,
            num_steps=state.num_steps + 1,
            auxiliary=lin_state,
            output_scale=output_scale,
            fun_evals=fx,  # return the initial linearization
        )

    def userfriendly_output(
        self, *, solution: ProbabilisticSolution, solution0: ProbabilisticSolution
    ):
        # This is the dynamic solver,
        # and all covariances have been calibrated already
        ones = np.ones_like(solution.output_scale)
        output_scale = ones[-1, ...]

        init = solution0.solution_full
        posterior = solution.solution_full
        estimate, posterior = self.strategy.finalize(
            posterior0=init, posterior=posterior, output_scale=output_scale
        )

        # TODO: stack the calibrated output scales?
        output_scale = ones
        ts = np.concatenate([solution0.t[None], solution.t])
        return ProbabilisticSolution(
            t=ts,
            u=estimate,
            solution_full=posterior,
            output_scale=output_scale,
            num_steps=solution.num_steps,
            auxiliary=solution.auxiliary,
            fun_evals=solution.fun_evals,
        )


class solver(ProbabilisticSolver):
    """Create a solver that does not calibrate the output scale automatically.

    This is the text-book implementation of probabilistic solvers.
    It is typically used in parameter estimation:

    - In combination with gradient-based optimisation of the output scale.
    - In combination with diffusion tempering.

    See the tutorials for example applications.


    Related:
    [`ProbabilisticSolver`](#probdiffeq.probdiffeq.ProbabilisticSolver).

    """

    def __init__(
        self,
        *,
        constraint: Constraint,
        prior: Callable,
        ssm: ssm_impl.FactSsmImpl,
        strategy: MarkovStrategy,
        constraint_init: Constraint | None = None,
    ) -> None:
        super().__init__(
            strategy=strategy,
            ssm=ssm,
            prior=prior,
            constraint=constraint,
            constraint_init=constraint_init,
        )

    def init(self, t: Array, u: TaylorCoeffTarget, *, damp) -> ProbabilisticSolution:
        u_pred, prediction = self.strategy.init_posterior(u=u)

        if self.constraint_init is not None:
            cstate_init = self.constraint_init.init_linearization()
            fx_init, _cstate = self.constraint_init.linearize(
                u_pred.marginals, cstate_init, damp=damp, t=t
            )
            zeros = tree.tree_map(np.zeros_like, fx_init.noise.mean_tree())
            updates = self.ssm.conditional.bayes_rule_tree(
                zeros, u_pred.marginals, fx_init, solve_triu=linalg.lstsq_svd
            )
            u, posterior = self.strategy.apply_updates(prediction, updates=updates)

        else:
            u, posterior = u_pred, prediction

        cstate = self.constraint.init_linearization()
        lin_fun = func.partial(self.constraint.linearize, damp=damp, t=t)
        fx, _cstate = func.eval_shape(lin_fun, u_pred.marginals, cstate)
        fx = tree.tree_map(np.zeros_like, fx)

        output_scale = np.ones_like(self.ssm.prior.prototype_output_scale_calibrated())
        return ProbabilisticSolution(
            t=t,
            u=u,
            solution_full=posterior,
            num_steps=0,
            auxiliary=cstate,
            output_scale=output_scale,
            fun_evals=fx,
        )

    def step(self, state: ProbabilisticSolution, *, dt, damp):
        # Discretize
        output_scale = np.ones_like(state.output_scale)
        transition = self.prior(dt, output_scale)

        # Predict
        u_pred, prediction = self.strategy.predict(
            state.solution_full, transition=transition
        )

        u = u_pred

        # Linearize
        fx, auxiliary = self.constraint.linearize(
            u.marginals, state.auxiliary, damp=damp, t=state.t + dt
        )

        # Update
        zeros = tree.tree_map(np.zeros_like, fx.noise.mean_tree())
        updates = self.ssm.conditional.bayes_rule_tree(
            zeros, u_pred.marginals, fx, solve_triu=linalg.solve_triu
        )
        u, posterior = self.strategy.apply_updates(prediction, updates=updates)

        # Return solution
        return ProbabilisticSolution(
            t=state.t + dt,
            u=u,
            solution_full=posterior,
            output_scale=output_scale,
            auxiliary=auxiliary,
            num_steps=state.num_steps + 1,
            fun_evals=fx,
        )

    def userfriendly_output(
        self, *, solution0: ProbabilisticSolution, solution: ProbabilisticSolution
    ) -> ProbabilisticSolution:
        assert solution.t.ndim > 0

        # This is the uncalibrated solver, so scale=1
        ones = np.ones_like(solution.output_scale)
        output_scale = np.ones_like(solution.output_scale[-1])

        init = solution0.solution_full
        posterior = solution.solution_full
        u, posterior = self.strategy.finalize(
            posterior0=init, posterior=posterior, output_scale=output_scale
        )

        output_scale = ones * output_scale[None, ...]

        ts = np.concatenate([solution0.t[None], solution.t])
        return ProbabilisticSolution(
            t=ts,
            u=u,
            solution_full=posterior,
            output_scale=output_scale,
            num_steps=solution.num_steps,
            auxiliary=solution.auxiliary,
            fun_evals=solution.fun_evals,
        )


def error_norm_scale_then_rms(*, norm_order=None) -> Callable:
    """Normalize an error by scaling followed by computing the root-mean-square norm.

    This is the recommended approach, and there is no reason to choose
    [`error_norm_rms_then_scale`](#probdiffeq.probdiffeq.error_norm_rms_then_scale),
    in situations where the present function applies.
    However, there are situations where it doesn't apply, for example,
    in residual-based error estimators for root constraints whose pytree
    structure differs from that of the target Taylor coefficients.

    See the custom information operator tutorial for details.
    """

    def normalize(error_abs, reference, atol, rtol):
        scale = atol + rtol * np.abs(reference)
        error_rel = error_abs / scale
        return rms(error_rel)

    def rms(s):
        return linalg.vector_norm(s, order=norm_order) / np.sqrt(s.size)

    return normalize


def error_norm_rms_then_scale(norm_order=None) -> Callable:
    """Normalize an error by computing the root-mean-square norm followed by scaling.

    Use this for residual-based error estimators in combination
    with custom root constraints.

    See the custom information operator tutorial for details.
    """

    def normalize(error_abs, reference, atol, rtol):
        norm_abs = rms(error_abs)
        norm_ref = rms(reference)
        return norm_abs / (atol + rtol * norm_ref)

    def rms(s):
        return linalg.vector_norm(s, order=norm_order) / np.sqrt(s.size)

    return normalize


class ErrorEstimator:
    """An interface for error estimators in probabilistic solvers.

    Related:
    [`error_residual_std`](#probdiffeq.probdiffeq.error_residual_std).

    """

    def init_error(self):
        """Initialize the error-estimation state."""
        raise NotImplementedError

    def estimate_error_norm(
        self,
        state: tuple,
        previous: ProbabilisticSolution,
        proposed: ProbabilisticSolution,
        *,
        dt: float,
        atol: float,
        rtol: float,
        damp: float,
    ):
        """Estimate the error norm.

        The error norm is a single scalar that already includes:

        - Absolute and relative tolerances
        - Error contraction rates

        In the acceptance/rejection step, this error norm is compared
        to one to determine whether a step has been successful.
        """
        raise NotImplementedError


class error_residual_std(ErrorEstimator):
    r"""Construct an error estimator based on a local residual's standard deviation.

    This is the common error estimate, proposed by Schober et al. (2019),
    extended by Bosch et al. (2021) to different linearization and calibration modes,
    and then generalised to state-space model factorisations by
    Krämer, Bosch, and Schmidt et al. (2022).
    Please consider citing these papers in your work if you use any of
    these error estimates.

    ??? note "BibTex for Schober et al. (2019)"
        ```bibtex
        @article{schober2019probabilistic,
            title={A probabilistic model for the numerical
            solution of initial value problems},
            author={Schober, Michael and S{\"a}rkk{\"a}, Simo and Hennig, Philipp},
            journal={Statistics and Computing},
            volume={29},
            number={1},
            pages={99--122},
            year={2019},
            publisher={Springer}
        }
        ```

    ??? note "BibTex for Bosch et al. (2021)"
        ```bibtex
            @inproceedings{bosch2021calibrated,
                title={Calibrated adaptive probabilistic ODE solvers},
                author={Bosch, Nathanael and Hennig, Philipp and Tronarp, Filip},
                booktitle={International Conference on
                Artificial Intelligence and Statistics},
                pages={3466--3474},
                year={2021},
                organization={PMLR}
            }
        ```

    ??? note "BibTex for Krämer, Bosch, and Schmidt et al. (2022)"
        ```bibtex
            @inproceedings{kramer2022probabilistic,
                title={Probabilistic ODE solutions in millions of dimensions},
                author={Kr{\"a}mer, Nicholas and Bosch, Nathanael and
                Schmidt, Jonathan and Hennig, Philipp},
                booktitle={International Conference on Machine Learning},
                pages={11634--11649},
                year={2022},
                organization={PMLR}
            }
        ```

    Related:
    [`ErrorEstimator`](#probdiffeq.probdiffeq.ErrorEstimator).

    """

    def __init__(
        self,
        *,
        constraint: Constraint,
        prior: Any,
        ssm: ssm_impl.FactSsmImpl,
        error_norm: Callable | None = None,
        re_linearize_before_error: bool = False,  # cache by default
        error_per_unit_step: bool = False,
    ) -> None:
        if error_norm is None:
            error_norm = error_norm_scale_then_rms()

        self.error_norm = error_norm
        self.constraint = constraint
        self.prior = prior
        self.ssm = ssm
        self.re_linearize_before_error = re_linearize_before_error
        self.error_per_unit_step = error_per_unit_step

    def init_error(self):
        return self.constraint.init_linearization()

    def estimate_error_norm(
        self,
        state,
        previous: ProbabilisticSolution,
        proposed: ProbabilisticSolution,
        *,
        dt: float,
        atol: float,
        rtol: float,
        damp: float,
    ) -> tuple[float, tuple]:
        # Discretize; The output scale is set to one
        # since the error is multiplied with a local scale estimate anyway
        output_scale = np.ones_like(self.ssm.prior.prototype_output_scale_calibrated())
        transition = self.prior(dt, output_scale)

        # Extrapolate from the zero-error state
        mean = previous.u.marginals.mean_tree()
        rv = self.ssm.conditional.apply_tree(mean, transition)

        # Optionally: re-linearize
        if self.re_linearize_before_error:
            linearized, state = self.constraint.linearize(
                rv, state, damp=damp, t=proposed.t
            )
        else:
            linearized = proposed.fun_evals

        # Extract the local residual std from the linearization
        observed = self.ssm.conditional.marginalise(rv, linearized)
        zeros = tree.tree_map(np.zeros_like, linearized.noise.mean_tree())
        output_scale = observed.residual_white_rms_tree(zeros)
        observed = observed.rescale_cholesky(output_scale)
        error = observed.std_tree()
        error, _ = tree.ravel_pytree(error)

        # Compute a reference
        previous_leaves = tree.tree_leaves(previous.u.mean)
        error_contraction_rate = len(previous_leaves)
        u0 = previous_leaves[0]
        u1 = tree.tree_leaves(proposed.u.mean)[0]
        reference = np.maximum(np.abs(u0), np.abs(u1))
        reference, _ = tree.ravel_pytree(reference)

        # Turn the unscaled absolute error into a relative one.
        # This is a generalisation of the typical residual-based
        # error estimates for probabilistic solvers in the sense that
        # it respects higher-order information. For first-order problems,
        # it is identical to Schober et al, Bosch et al., and so on.
        # For higher-order problems it is closer to Taylor-series based
        # (non-probabilistic) ODE solvers; for example, refer to
        # Tan et al. (2026; https://arxiv.org/pdf/2602.04086).
        n = self.constraint.root_order - 1
        if self.error_per_unit_step:
            n += 1

        error_abs = error * dt**n / np.factorial(n)
        error_norm = self.error_norm(error_abs, reference, atol=atol, rtol=rtol)

        # Scale the error norm with the error contraction rate and return
        error_power = error_norm ** (-1.0 / error_contraction_rate)
        return error_power, state


class error_state_std(ErrorEstimator):
    r"""Construct an error estimator based on a state's standard deviation.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This function is highly experimental and not safe to use.
        There is no guarantee that it works correctly (or at all).
        It might be deleted tomorrow and without any deprecation policy.

    """

    # TODO: make the experimental-warning into a decorator
    def __init__(
        self,
        *,
        constraint: Constraint,
        prior: Any,
        ssm: ssm_impl.FactSsmImpl,
        error_norm: Callable | None = None,
        re_linearize_before_error: bool = False,  # cache by default
        derivative_idx: int = 0,
        error_per_unit_step: bool = False,
    ) -> None:
        if error_norm is None:
            error_norm = error_norm_scale_then_rms()

        self.error_norm = error_norm
        self.constraint = constraint
        self.prior = prior
        self.ssm = ssm
        self.re_linearize_before_error = re_linearize_before_error
        self.derivative_idx = derivative_idx
        self.error_per_unit_step = error_per_unit_step

    def init_error(self):
        return self.constraint.init_linearization()

    def estimate_error_norm(
        self,
        state,
        previous: ProbabilisticSolution,
        proposed: ProbabilisticSolution,
        *,
        dt: float,
        atol: float,
        rtol: float,
        damp: float,
    ) -> tuple[float, tuple]:
        # Discretize; The output scale is set to one
        # since the error is multiplied with a local scale estimate anyway
        output_scale = np.ones_like(self.ssm.prior.prototype_output_scale_calibrated())
        transition = self.prior(dt, output_scale)

        # Extrapolate from the zero-error state
        mean = previous.u.marginals.mean_tree()
        rv = self.ssm.conditional.apply_tree(mean, transition)

        mean_leaves = tree.tree_leaves(mean)
        error_contraction_rate = len(mean_leaves)

        # Optionally: re-linearize
        if self.re_linearize_before_error:
            linearized, state = self.constraint.linearize(
                rv, state, damp=damp, t=proposed.t
            )
        else:
            linearized = proposed.fun_evals

        # Extract the local residual std from the linearization
        zeros = tree.tree_map(np.zeros_like, linearized.noise.mean_tree())
        output_scale, conditional = (
            self.ssm.conditional.bayes_rule_and_residual_white_rms_tree(
                zeros, rv, linearized, solve_triu=linalg.solve_triu
            )
        )

        # Measure error on the n-th state (usually, n=0 because why not)
        n = self.derivative_idx

        if self.error_per_unit_step:
            n += 1

        # *New:* Go back into solution space
        std = conditional.std_tree()[n]
        error, _ = tree.ravel_pytree(std)
        error = output_scale * error
        error, _ = tree.ravel_pytree(error)

        # Compute a reference
        u0, _ = tree.ravel_pytree(previous.u.mean[n])
        u1, _ = tree.ravel_pytree(proposed.u.mean[n])
        reference = np.maximum(np.abs(u0), np.abs(u1))

        # Turn the unscaled absolute error into a relative one.
        error_abs = error * dt**n / np.factorial(n)
        error_norm = self.error_norm(error_abs, reference, atol=atol, rtol=rtol)

        # Scale the error norm with the error contraction rate and return
        error_power = error_norm ** (-1.0 / error_contraction_rate)
        return error_power, state
