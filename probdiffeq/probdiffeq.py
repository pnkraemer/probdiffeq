"""Probabilistic solvers for differential equations.

See the tutorials for example use cases.
"""

from probdiffeq import taylor
from probdiffeq.backend import (
    flow,
    func,
    inspect,
    linalg,
    np,
    random,
    special,
    structs,
    tree,
)
from probdiffeq.backend.typing import (
    Any,
    Array,
    ArrayLike,
    Callable,
    Generic,
    Literal,
    Protocol,
    Sequence,
    TypeVar,
)
from probdiffeq.impl import impl

C = TypeVar("C", bound=Sequence)
T = TypeVar("T")


@tree.register_dataclass
@structs.dataclass
class CubaturePositiveWeights:
    """A datastructure for cubature rules that have positive weights.

    Related:
    [`cubature_gauss_hermite`](#probdiffeq.probdiffeq.cubature_gauss_hermite),
    [`cubature_third_order_spherical`](#probdiffeq.probdiffeq.cubature_third_order_spherical),
    [`cubature_unscented_transform`](#probdiffeq.probdiffeq.cubature_unscented_transform).

    """

    points: ArrayLike
    """Cubature points."""

    weights_sqrtm: ArrayLike
    """Square roots of cubature weights."""


def cubature_third_order_spherical(input_shape):
    """Third-order spherical cubature integration.

    Related:
    [`CubaturePositiveWeights`](#probdiffeq.probdiffeq.CubaturePositiveWeights).

    """
    assert len(input_shape) <= 1
    if len(input_shape) == 1:
        (d,) = input_shape
        points_mat, weights_sqrtm = _third_order_spherical_params(d=d)
        return CubaturePositiveWeights(points=points_mat, weights_sqrtm=weights_sqrtm)

    # If input_shape == (), compute weights via input_shape=(1,)
    # and 'squeeze' the points.
    points_mat, weights_sqrtm = _third_order_spherical_params(d=1)
    (S, _) = points_mat.shape
    points = np.reshape(points_mat, (S,))
    return CubaturePositiveWeights(points=points, weights_sqrtm=weights_sqrtm)


def _third_order_spherical_params(*, d):
    eye_d = np.eye(d) * np.sqrt(d)
    pts = np.concatenate((eye_d, -1 * eye_d))
    weights_sqrtm = np.ones((2 * d,)) / np.sqrt(2.0 * d)
    return pts, weights_sqrtm


def cubature_unscented_transform(input_shape, r=1.0):
    """Unscented transform.

    Related:
    [`CubaturePositiveWeights`](#probdiffeq.probdiffeq.CubaturePositiveWeights).

    """
    assert len(input_shape) <= 1
    if len(input_shape) == 1:
        (d,) = input_shape
        points_mat, weights_sqrtm = _unscented_transform_params(d=d, r=r)
        return CubaturePositiveWeights(points=points_mat, weights_sqrtm=weights_sqrtm)

    # If input_shape == (), compute weights via input_shape=(1,)
    # and 'squeeze' the points.
    points_mat, weights_sqrtm = _unscented_transform_params(d=1, r=r)
    (S, _) = points_mat.shape
    points = np.reshape(points_mat, (S,))
    return CubaturePositiveWeights(points=points, weights_sqrtm=weights_sqrtm)


def _unscented_transform_params(d, *, r):
    eye_d = np.eye(d) * np.sqrt(d + r)
    zeros = np.zeros((1, d))
    pts = np.concatenate((eye_d, zeros, -1 * eye_d))
    _scale = d + r
    weights_sqrtm1 = np.ones((d,)) / np.sqrt(2.0 * _scale)
    weights_sqrtm2 = np.sqrt(r / _scale)
    weights_sqrtm = np.hstack((weights_sqrtm1, weights_sqrtm2, weights_sqrtm1))
    return pts, weights_sqrtm


def cubature_gauss_hermite(input_shape, degree=5):
    """(Statistician's) Gauss-Hermite cubature.

    The number of cubature points is `prod(input_shape)**degree`.

    Related:
    [`CubaturePositiveWeights`](#probdiffeq.probdiffeq.CubaturePositiveWeights).

    """
    assert len(input_shape) == 1
    (dim,) = input_shape

    # Roots of the probabilist/statistician's Hermite polynomials (in Numpy...)
    _roots = special.roots_hermitenorm(n=degree, mu=True)
    pts, weights, sum_of_weights = _roots
    weights = weights / sum_of_weights

    # Transform into jax arrays and take square root of weights
    pts = np.asarray(pts)
    weights_sqrtm = np.sqrt(np.asarray(weights))

    # Build a tensor grid and return class
    tensor_pts = _tensor_points(pts, d=dim)
    tensor_weights_sqrtm = _tensor_weights(weights_sqrtm, d=dim)
    return CubaturePositiveWeights(
        points=tensor_pts, weights_sqrtm=tensor_weights_sqrtm
    )


# TODO: how does this generalise to an input_shape instead of an input_dimension?
#  via tree_map(lambda s: _tensor_points(x, s), input_shape)?


def _tensor_weights(*args, **kwargs):
    mesh = _tensor_points(*args, **kwargs)
    return np.prod_along_axis(mesh, axis=1)


def _tensor_points(x, /, *, d):
    x_mesh = np.meshgrid(*([x] * d))
    y_mesh = tree.tree_map(lambda s: np.reshape(s, (-1,)), x_mesh)
    return np.stack(y_mesh).T


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

    def __init__(self, *, jacfun=func.jacfwd):
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
    """

    def __init__(self, *, seed=1, num_probes=10):
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
    """

    def __init__(self, *, seed=1, num_probes=10):
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


def loss_lml_terminal_values(*, ssm, tcoeff_index=0):
    """Construct a log-marginal-likelihood loss for the terminal value."""

    def loss(u, /, *, marginals, std):
        u = tree.tree_map(np.asarray, u)

        std_expected = tree.tree_map(lambda _s: ssm.prototypes.standard_deviation(), u)
        std = tree.tree_map(np.asarray, std)
        shapes = tree.tree_map(lambda a, b: a.shape == b.shape, std, std_expected)
        shapes_equal = tree.tree_all(shapes)

        if not shapes_equal:
            msg = "The standard deviation container differs from what was expected."
            msg += f" Expected: shape={tree.tree_map(np.shape, std_expected)}."
            msg += f" Received: shape={tree.tree_map(np.shape, std)}."
            msg += f" For reference, data: shape={tree.tree_map(np.shape, u)}."
            raise ValueError(msg)

        model = ssm.conditional.to_derivative(tcoeff_index, std)
        marg = ssm.conditional.marginalise(marginals, model)
        return marg.logpdf(u)

    return loss


def loss_lml_timeseries(*, ssm, average_pdfs: bool = True, tcoeff_index=0):
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

        def batch_std(s):
            return func.vmap(lambda _s: ssm.prototypes.standard_deviation())(s)

        std_expected = tree.tree_map(batch_std, u)
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
            return ssm.conditional.to_derivative(tcoeff_index, s)

        model = func.vmap(make_model)(std)
        return posterior.eval_lml(u, model=model, ssm=ssm, average_pdfs=average_pdfs)

    return loss


class Constraint(Protocol):
    """An interface for constraints + linearization in probabilistic solvers.

    Related:
    [`constraint_ode_ts0`](#probdiffeq.probdiffeq.constraint_ode_ts0),
    [`constraint_ode_ts1`](#probdiffeq.probdiffeq.constraint_ode_ts1),
    [`constraint_ode_slr0`](#probdiffeq.probdiffeq.constraint_ode_slr0),
    [`constraint_ode_slr1`](#probdiffeq.probdiffeq.constraint_ode_slr1).
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


def constraint_ode_ts0(vf, /, *, ssm):
    """Create an ODE constraint with zeroth-order Taylor linearisation.

    Related:
    [`Constraint`](#probdiffeq.probdiffeq.Constraint).
    """
    ode_order = _verify_vector_field_signature_and_parse_order(vf)
    return ssm.linearize.ode_taylor_0th(vf, ode_order=ode_order)


def constraint_root_ts1(root, /, *, ssm, jacobian=None):
    """Construct a constraint based on a custom root.

    See the custom information operator tutorial for details.

    Related:
    [`Constraint`](#probdiffeq.probdiffeq.Constraint).
    """
    root_order = _verify_vector_field_signature_and_parse_order(root)
    if root_order == 1:
        msg = "Did you accidentally pass a vector field instead of a root-constraint?"
        raise ValueError(msg)

    if jacobian is None:
        # Use hutchinson Jacobian handling for backward compatibility.
        jacobian = jacobian_hutchinson_fwd()
    return ssm.linearize.root_taylor_1st(root, root_order=root_order, jacobian=jacobian)


def constraint_root_jet(root, /, *, ssm, jacobian=None):
    """Construct a constraint based on a custom root.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This function is highly experimental and not safe to use.
        There is no guarantee that it works correctly (or at all).
        It might be deleted tomorrow and without any deprecation policy.

    """
    root_order = _verify_vector_field_signature_and_parse_order(root)
    if root_order == 1:
        msg = "Did you accidentally pass a vector field instead of a root-constraint?"
        raise ValueError(msg)

    if jacobian is None:
        jacobian = jacobian_hutchinson_fwd()

    def root_jet(*tcoeffs_all, t):
        # TODO: if we apply the preconditioner before passing
        #   things in here, we can set is_tcoeff to True and possibly
        #   gain a bunch of numerical robustness
        ps, ss = taylor.jet_unpack_series(tcoeffs_all, root_order)
        primals, series = func.jet(lambda *y: root(*y, t=t), ps, ss, is_tcoeff=False)
        return [primals, *series]

    # TODO: once we have a second root constraint (eg slr1),
    #       offer the below as a function argument.
    return ssm.linearize.root_taylor_1st(
        root_jet, root_order=ssm.num_derivatives + 1, jacobian=jacobian
    )


def constraint_ode_ts1(vf, /, *, ssm, jacobian: JacobianHandler | None = None):
    """Create an ODE constraint with first-order Taylor linearisation.

    Related:
    [`Constraint`](#probdiffeq.probdiffeq.Constraint).

    The ODE vector field is assumed to be one of f(u, *, t), f(u, du, *, t), etc.
    The order of the ODE is read off the number of positional arguments before t.
    That is, for first-order ODEs, pass f(u, *, t),
    for second order ODEs, pass f(u, du, *, t),
    for third-order ODEs f(u, du, ddu, *, t), and so on.

    """
    ode_order = _verify_vector_field_signature_and_parse_order(vf)
    if jacobian is None:
        # Use hutchinson Jacobian handling for backward compatibility.
        jacobian = jacobian_hutchinson_fwd()
    return ssm.linearize.ode_taylor_1st(vf, ode_order=ode_order, jacobian=jacobian)


def constraint_ode_slr0(vf, /, *, ssm, cubature_fun=cubature_third_order_spherical):
    """Create an ODE constraint with zeroth-order statistical linear regression.

    Related:
    [`Constraint`](#probdiffeq.probdiffeq.Constraint).
    """
    ode_order = _verify_vector_field_signature_and_parse_order(vf)
    if ode_order > 1:
        msg = "SLR0 constraints cannot handle higher-order ODEs as of now."
        msg += " However, ode_order={ode_order} has been detected."
        msg += " Try a Taylor-series-based constraint instead."
        raise ValueError(msg)
    return ssm.linearize.ode_statistical_0th(vf, cubature_fun=cubature_fun)


def constraint_ode_slr1(vf, *, ssm, cubature_fun=cubature_third_order_spherical):
    """Create an ODE constraint with first-order statistical linear regression.

    Related:
    [`Constraint`](#probdiffeq.probdiffeq.Constraint).

    """
    ode_order = _verify_vector_field_signature_and_parse_order(vf)
    if ode_order > 1:
        msg = "SLR1 constraints cannot handle higher-order ODEs as of now."
        msg += " However, ode_order={ode_order} has been detected."
        msg += " Try a Taylor-series-based constraint instead."
        raise ValueError(msg)
    return ssm.linearize.ode_statistical_1st(vf, cubature_fun=cubature_fun)


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


@tree.register_dataclass
@structs.dataclass
class TaylorCoeffTarget(Generic[C, T]):
    """A probabilistic description of Taylor coefficients.

    Includes means, standard deviations, and marginals.
    Taylor coefficients are common solution targets
    for probabilistic differential equation solvers.
    """

    mean: C
    """A PyTree describing the mean of the Taylor coefficient."""

    std: C
    """A PyTree describing the standard deviation of the Taylor coefficient."""

    marginals: T
    """The full marginal distribution of the Taylor coefficient."""

    @classmethod
    def from_marginals(cls, marginals):
        mean = marginals.eval_mean()
        std = marginals.eval_standard_deviation()
        return cls(mean, std, marginals)


@tree.register_dataclass
@structs.dataclass
class MarkovSequence(Generic[T]):
    """A datastructure for Markov sequences as batches of joint distributions.

    This is the output type of smoother-based estimators.
    (Filter-based estimators do not return specialised types.)
    """

    marginal: T
    """The marginal distribution."""

    conditional: Any
    """The conditional distribution."""

    reverse: bool = structs.dataclass_field(metadata={"static": True})
    """The direction of factorisations."""

    def rescale_cholesky(self, factor, /):
        marg = self.marginal.rescale_cholesky(factor)
        cond = self.conditional.rescale_noise(factor)
        return MarkovSequence(marg, cond, reverse=self.reverse)

    def eval_marginals(self, *, ssm):
        """Extract the (time-)marginals from a Markov sequence.

        This is only needed in combination with smoothing-based strategies.
        """
        if self.marginal.mean.ndim == self.conditional.noise.mean.ndim:
            markov_seq = self._select_terminal()
            return markov_seq.eval_marginals(ssm=ssm)

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

    def eval_lml(self, u, *, model, ssm, average_pdfs: bool):
        assert self.reverse

        if self.marginal.mean.ndim == self.conditional.noise.mean.ndim:
            markov_seq = self._select_terminal()
            return markov_seq.eval_lml(
                u, model=model, ssm=ssm, average_pdfs=average_pdfs
            )

        # Process the terminal value
        u0 = tree.tree_map(lambda s: s[-1], u)
        model0 = tree.tree_map(lambda s: s[-1], model)
        marg, cond = ssm.conditional.revert(self.marginal, model0)
        updated = ssm.conditional.apply(u0, cond)
        pdf0 = marg.logpdf(u0)

        # Process the remaining values
        def body(rv_and_logpdf, prior_and_observation_and_data):
            rv, logpdf, num_data = rv_and_logpdf
            prior, observation, data = prior_and_observation_and_data

            predicted = ssm.conditional.marginalise(rv, prior)
            observed, noise = ssm.conditional.revert(predicted, observation)
            logpdf_n = observed.logpdf(data)
            corrected = ssm.conditional.apply(data, noise)

            # The mean of the PDFs usually leads to LML values that are more
            # "human-readable" (ie O(1) instead O(N)). This is technically not
            # a log-marginal-likelihood, but much nicer to work with.
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

    def sample(self, key, *, ssm, shape: tuple = ()):
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
            predicted = ssm.conditional.apply(smp0, cond)
            key, subkey = random.split(key, num=2)
            smp1 = predicted.sample(subkey)
            return (smp1, key), smp1

        key, subkey = random.split(key, num=2)
        smp = self.marginal.sample(subkey)
        _, smps = flow.scan(
            body, init=(smp, key), xs=self.conditional, reverse=self.reverse
        )

        # Currently, sampling is only implemented for reverse
        if self.reverse:
            return tree.tree_array_append(smps, smp)
        return tree.tree_array_prepend(smp, smps)


class MarkovStrategy(Generic[T]):
    """An interface for estimation strategies in Markovian state-space models.

    Related:
    [`strategy_filter`](#probdiffeq.probdiffeq.strategy_filter),
    [`strategy_smoother_fixedpoint`](#probdiffeq.probdiffeq.strategy_smoother_fixedpoint),
    [`strategy_smoother_fixedinterval`](#probdiffeq.probdiffeq.strategy_smoother_fixedinterval).
    """

    def __init__(
        self,
        ssm: Any,
        is_suitable_for_save_at: int,
        is_suitable_for_save_every_step: int,
        is_suitable_for_offgrid_marginals: int,
    ):
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
    ):
        """Interpolate between two points."""
        raise NotImplementedError

    def interpolate_at_t1(self, *, posterior_t1: T):
        """Interpolate at a checkpoint."""
        raise NotImplementedError

    def finalize(self, *, posterior0: T, posterior: T, output_scale) -> T:
        """Finalize the posterior before returning a solution."""
        raise NotImplementedError


@tree.register_dataclass
@structs.dataclass
class ProbabilisticSolution(Generic[C, T]):
    """A datastructure for probabilistic solutions of differential equations."""

    t: Array
    """The current time-step."""

    u: TaylorCoeffTarget[C, T]
    """The current ODE solution estimate."""

    solution_full: T | MarkovSequence[T]
    """The current posterior estimate."""

    # Todo: merge 'output_scale' and 'auxiliary' and "fun_evals"?
    output_scale: Any
    """The current output scale."""

    num_steps: int
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
        ssm: Any,
    ):
        self.ssm = ssm
        self.strategy = strategy
        self.prior = prior
        self.constraint = constraint

    @property
    def error_contraction_rate(self):
        """The error-contraction rate of the solver."""
        return self.ssm.num_derivatives + 1

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

        interp_res = _InterpRes(step_from=step_from, interp_from=interp_from)
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
        return sol, _InterpRes(step_from=acc, interp_from=prev)


def prior_wiener_integrated(
    tcoeffs: C,
    *,
    # Which of the Taylor coefficients are differential variables
    is_differential: C | None = None,
    nondifferential_eps: float = 1e-6,  # a small value
    # The state-space model factorisation
    ssm_fact: Literal["dense", "isotropic", "blockdiag"] = "dense",  # noqa: F821
    # Do we use a special output scale?
    output_scale: C = None,
    # How many extra derivatives to model in the state-space
    diffuse_derivatives: int = 0,
    diffuse_eps: float = 1.0,  # a large value
):
    """Construct an repeatedly-integrated Wiener process.

    Tip: Choose nonzero standard deviations
    to get visually more pleasing uncertainties and more numerical robustness for
    high-order solvers in low precision arithmetic. Outside of these cases,
    leave the standard deviations at zero to improve accuracy.
    """
    tcoeffs_std = _tcoeffs_std_from_differential_variables(
        tcoeffs,
        is_differential=is_differential,
        nondifferential_eps=nondifferential_eps,
        ssm_fact=ssm_fact,
    )

    return prior_wiener_integrated_diffuse(
        tcoeffs,
        tcoeffs_std,
        ssm_fact=ssm_fact,
        output_scale=output_scale,
        diffuse_derivatives=diffuse_derivatives,
        diffuse_eps=diffuse_eps,
    )


def _tcoeffs_std_from_differential_variables(
    tcoeffs, *, ssm_fact, is_differential, nondifferential_eps
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

    # If 'is_differential' hasn't been provided, simply return zeros everywhere
    if is_differential is None:
        return std_template

    # Before using is_differential, verify it has the correct structure and shape
    try:

        def shape_equal(A, B):
            return tree.tree_map(lambda a, b: a.shape == b.shape, A, B)

        assert tree.tree_all(shape_equal(is_differential, std_template))
    except (ValueError, AssertionError) as err:
        msg = "Input 'is_differential' has the wrong PyTree structure."
        msg += f" Expected: {tree.tree_map(np.shape, std_template)}."
        msg += f" Received: {tree.tree_map(np.shape, is_differential)}."
        raise ValueError(msg) from err

    # Wherever is_differential is True, initialize with zeros.
    # Elsewhere, initialize with a small positivec value.

    def std_init(s):
        if np.dtype(s) != np.dtype(bool):
            msg = "Boolean entries expected in `is_differential`."
            msg += f" Received: dtype={np.dtype(s)}"
            raise TypeError(msg)
        return np.where(s, 0.0, nondifferential_eps)

    return tree.tree_map(std_init, is_differential)


def prior_wiener_integrated_diffuse(
    tcoeffs_mean: C,
    tcoeffs_std: C,
    *,
    # The state-space model factorisation
    ssm_fact: Literal["dense", "isotropic", "blockdiag"] = "dense",  # noqa: F821
    # Do we use a special output scale?
    output_scale: C = None,
    # How many extra derivatives to model in the state-space
    diffuse_derivatives: int = 0,
    diffuse_eps: float = 1.0,  # a large value
):
    """Construct an diffuse repeatedly-integrated Wiener process.

    The diffuse process has a nonzero initial standard deviation.
    This is typically used to:
    - Either get more visually-pleasing uncertainties and gain
      numerical robustness for high-order solvers in low precision arithmetic.
    - Communicate to the solvers that the prior has not seen any data
      (and solvers can handle data at initialisation themselves.)

    Outside of these cases, use the usual integrated Wiener process.
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
    ssm = impl.choose(ssm_fact, tcoeffs_like=tcoeffs_mean)

    if output_scale is None:
        output_scale = tree.tree_map(np.ones_like, tcoeffs_std[0])
    else:
        output_scale = tree.tree_map(np.asarray, output_scale)

        def shape_equal(A, B):
            return tree.tree_map(lambda a, b: a.shape == b.shape, A, B)

        if not tree.tree_all(shape_equal(output_scale, tcoeffs_std[0])):
            msg = "Output scale has the wrong shape."
            msg += f" Expected: output_scale.shape={tcoeffs_std[0].shape}."
            msg += f" Received: output_scale.shape={output_scale.shape}."
            raise ValueError(msg)

    [output_scale_leaf] = tree.tree_leaves(output_scale)
    std_leaves, structure = tree.tree_flatten(tcoeffs_std)
    std_leaves_scaled = [output_scale_leaf * s for s in std_leaves]
    tcoeffs_std = tree.tree_unflatten(structure, std_leaves_scaled)

    discretize = ssm.conditional.transition_wiener_integrated(base_scale=output_scale)

    # Return the target
    marginal = ssm.normal.from_mean_and_std(tcoeffs_mean, tcoeffs_std)
    target = TaylorCoeffTarget.from_marginals(marginal)
    return target, discretize, ssm


def _add_diffuse_derivatives(
    tcoeffs_mean, tcoeffs_std, *, diffuse_eps, diffuse_derivatives
):
    zeros = tree.tree_map(np.zeros_like, tcoeffs_mean[0])
    tcoeffs_mean = [*tcoeffs_mean, *[zeros for _ in range(diffuse_derivatives)]]

    unknowns = tree.tree_map(lambda s: diffuse_eps * np.ones_like(s), tcoeffs_std[0])
    tcoeffs_std = [*tcoeffs_std, *[unknowns for _ in range(diffuse_derivatives)]]
    return tcoeffs_mean, tcoeffs_std


def prior_wiener_integrated_discrete(
    tcoeffs: C,
    grid: Array,
    *,
    is_differential: C | None = None,
    nondifferential_eps: float = 1e-6,  # a small value
    ssm_fact: Literal["dense", "isotropic", "blockdiag"] = "dense",  # noqa: F821
    output_scale: C = None,
    # How many extra derivatives to model in the state-space
    diffuse_derivatives: int = 0,
    diffuse_eps: float = 1.0,  # a large value
):
    """Compute a time-discretization of an integrated Wiener process."""
    init, discretize, ssm = prior_wiener_integrated(
        tcoeffs,
        is_differential=is_differential,
        nondifferential_eps=nondifferential_eps,
        ssm_fact=ssm_fact,
        output_scale=output_scale,
        diffuse_derivatives=diffuse_derivatives,
        diffuse_eps=diffuse_eps,
    )
    scales = np.ones_like(ssm.prototypes.output_scale_calibrated())
    discretize_vmap = func.vmap(discretize, in_axes=(0, None))
    conditionals = discretize_vmap(np.diff(grid), scales)
    markov_seq = MarkovSequence(init.marginals, conditionals, reverse=False)
    return markov_seq, ssm


@tree.register_dataclass
@structs.dataclass
class _InterpRes(Generic[T]):
    """A datastructure to store interpolation results.

    To ensure correct adaptive time-stepping, it is important
    to distinguish step-from variables from interpolate-from variables.

    For some solvers, e.g. fixed-point-smoother-based ones,
    both stepping and interpolating variables are adjusted during interpolation.
    """

    step_from: T
    """The new 'step_from' field.

    At time `max(t, s1.t)`.
    Use this as the right-most reference state
    in future interpolations, or continue time-stepping from here.
    """

    interp_from: T
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


class strategy_smoother_fixedinterval(MarkovStrategy[MarkovSequence]):
    """Construct a fixed-interval smoother.

    Use this strategy for fixed steps.
    For adaptive steps, consider using a fixed-point smoother instead.


    Related:
    [`MarkovStrategy`](#probdiffeq.probdiffeq.MarkovStrategy).
    """

    def __init__(self, ssm):
        super().__init__(
            ssm=ssm,
            is_suitable_for_save_at=False,
            is_suitable_for_save_every_step=True,
            is_suitable_for_offgrid_marginals=True,
        )

    def init_posterior(self, *, u: TaylorCoeffTarget):
        cond = self.ssm.conditional.identity(self.ssm.num_derivatives + 1)
        posterior = MarkovSequence(marginal=u.marginals, conditional=cond, reverse=True)
        return u, posterior

    def predict(
        self, posterior: MarkovSequence, *, transition
    ) -> tuple[TaylorCoeffTarget, MarkovSequence]:
        marginals, cond = self.ssm.conditional.revert(posterior.marginal, transition)
        posterior = MarkovSequence(
            marginal=marginals, conditional=cond, reverse=posterior.reverse
        )

        estimate = TaylorCoeffTarget.from_marginals(marginals)
        return estimate, posterior

    def apply_updates(self, prediction, *, updates):
        posterior = MarkovSequence(
            updates, prediction.conditional, reverse=prediction.reverse
        )
        marginals = updates
        estimate = TaylorCoeffTarget.from_marginals(marginals)
        return estimate, posterior

    def finalize(
        self, *, posterior0: MarkovSequence, posterior: MarkovSequence, output_scale
    ):
        assert output_scale.shape == self.ssm.prototypes.output_scale_calibrated().shape
        posterior0 = posterior0.rescale_cholesky(output_scale)
        posterior = posterior.rescale_cholesky(output_scale)

        # Marginalise
        marginals = posterior.eval_marginals(ssm=self.ssm)
        # marginals = self.markov_marginals(posterior, reverse=True)

        # Prepend the initial condition to the filtering distributions
        init = tree.tree_array_prepend(posterior0.marginal, posterior.marginal)
        posterior = MarkovSequence(
            marginal=init, conditional=posterior.conditional, reverse=posterior.reverse
        )

        # Extract targets
        estimate = TaylorCoeffTarget.from_marginals(marginals)
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
        interp_res = _InterpRes(step_from=solution_at_t1, interp_from=solution_at_t)

        # Extract targets
        marginals = solution_at_t.marginal
        estimate = TaylorCoeffTarget.from_marginals(marginals)
        return (estimate, solution_at_t), interp_res

    def interpolate_at_t1(self, posterior_t1):
        marginals = posterior_t1.marginal
        estimate = TaylorCoeffTarget.from_marginals(marginals)

        interp_res = _InterpRes(step_from=posterior_t1, interp_from=posterior_t1)
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

    def __init__(self, ssm):
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
        estimate = TaylorCoeffTarget.from_marginals(marginals)
        return estimate, marginals

    def apply_updates(self, prediction, *, updates):
        del prediction
        marginals = updates
        estimate = TaylorCoeffTarget.from_marginals(marginals)
        return estimate, marginals

    def finalize(self, *, posterior0, posterior, output_scale):
        assert output_scale.shape == self.ssm.prototypes.output_scale_calibrated().shape

        # No rescaling because no calibration at the initial step
        posterior0 = posterior0.rescale_cholesky(output_scale)

        # Calibrate
        posterior = posterior.rescale_cholesky(output_scale)

        # Stack
        posterior = tree.tree_array_prepend(posterior0, posterior)

        marginals = posterior
        estimate = TaylorCoeffTarget.from_marginals(marginals)
        return estimate, posterior

    def interpolate(
        self, *, posterior_t0, posterior_t1, transition_t0_t, transition_t_t1
    ):
        del transition_t_t1
        _, interpolated = self.predict(
            posterior=posterior_t0, transition=transition_t0_t
        )
        marginals = interpolated
        estimate = TaylorCoeffTarget.from_marginals(marginals)
        interp_res = _InterpRes(step_from=posterior_t1, interp_from=interpolated)
        return (estimate, interpolated), interp_res

    def interpolate_at_t1(self, *, posterior_t1):
        marginals = posterior_t1
        estimate = TaylorCoeffTarget.from_marginals(marginals)

        interp_res = _InterpRes(step_from=posterior_t1, interp_from=posterior_t1)
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

    def __init__(self, ssm):
        super().__init__(
            ssm=ssm,
            is_suitable_for_save_at=True,
            is_suitable_for_save_every_step=False,
            is_suitable_for_offgrid_marginals=False,
        )

    def init_posterior(self, *, u):
        cond = self.ssm.conditional.identity(self.ssm.num_derivatives + 1)
        posterior = MarkovSequence(u.marginals, cond, reverse=True)
        return u, posterior

    def predict(
        self, posterior: MarkovSequence, *, transition
    ) -> tuple[TaylorCoeffTarget, MarkovSequence]:
        rv = posterior.marginal
        bw0 = posterior.conditional
        marginals, cond = self.ssm.conditional.revert(rv, transition)
        cond = self.ssm.conditional.merge(bw0, cond)
        predicted = MarkovSequence(marginals, cond, reverse=posterior.reverse)

        estimate = TaylorCoeffTarget.from_marginals(marginals)
        return estimate, predicted

    def apply_updates(self, prediction: MarkovSequence, *, updates):
        posterior = MarkovSequence(
            updates, prediction.conditional, reverse=prediction.reverse
        )
        marginals = updates
        estimate = TaylorCoeffTarget.from_marginals(marginals)
        return estimate, posterior

    def finalize(
        self, *, posterior0: MarkovSequence, posterior: MarkovSequence, output_scale
    ):
        assert output_scale.shape == self.ssm.prototypes.output_scale_calibrated().shape
        posterior0 = posterior0.rescale_cholesky(output_scale)
        posterior = posterior.rescale_cholesky(output_scale)

        # Marginalise
        marginals = posterior.eval_marginals(ssm=self.ssm)

        # Prepend the initial condition to the filtering distributions
        init = tree.tree_array_prepend(posterior0.marginal, posterior.marginal)
        posterior = MarkovSequence(
            marginal=init, conditional=posterior.conditional, reverse=posterior.reverse
        )

        # Extract targets
        estimate = TaylorCoeffTarget.from_marginals(marginals)
        return estimate, posterior

    def interpolate_at_t1(self, *, posterior_t1: MarkovSequence):
        cond_identity = self.ssm.conditional.identity(self.ssm.num_derivatives + 1)
        resume_from = MarkovSequence(
            posterior_t1.marginal,
            conditional=cond_identity,
            reverse=posterior_t1.reverse,
        )
        interp_res = _InterpRes(step_from=resume_from, interp_from=resume_from)

        interpolated = posterior_t1
        marginals = interpolated.marginal
        estimate = TaylorCoeffTarget.from_marginals(marginals)
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
        conditional_id = self.ssm.conditional.identity(self.ssm.num_derivatives + 1)
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
        interp_res = _InterpRes(step_from=step_from, interp_from=previous_new)

        marginals = interpolated.marginal
        estimate = TaylorCoeffTarget.from_marginals(marginals)
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
        ssm: Any,
        strategy: MarkovStrategy,
        update_at_init: bool = False,
    ):
        super().__init__(strategy=strategy, ssm=ssm, prior=prior, constraint=constraint)
        self.update_at_init = update_at_init

    def init(self, t, u: TaylorCoeffTarget, *, damp) -> ProbabilisticSolution:
        u, prediction = self.strategy.init_posterior(u=u)
        cstate = self.constraint.init_linearization()

        output_scale_prior = np.ones_like(self.ssm.prototypes.output_scale_calibrated())

        # Update
        fx, cstate = self.constraint.linearize(u.marginals, cstate, damp=damp, t=t)
        if self.update_at_init:
            raise RuntimeError
            observed, reverted = self.ssm.conditional.revert(u.marginals, fx)
            updates = reverted.noise
            u, posterior = self.strategy.apply_updates(
                prediction=prediction, updates=updates
            )
            # Calibrate the output scale
            output_scale_running = self.ssm.stats.mahalanobis_norm_relative(
                0.0, observed
            )
            auxiliary = (cstate, output_scale_running, 1)
        fx = tree.tree_map(np.zeros_like, fx)
        posterior = prediction
        output_scale_running = np.zeros_like(output_scale_prior)
        auxiliary = (cstate, output_scale_running, 0)

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
        output_scale = np.ones_like(self.ssm.prototypes.output_scale_calibrated())
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
        observed, reverted = self.ssm.conditional.revert(u.marginals, fx)
        updates = reverted.noise
        u, posterior = self.strategy.apply_updates(
            prediction=prediction, updates=updates
        )

        # Calibrate the output scale
        new_term = observed.mahalanobis_norm_relative(0.0)
        output_scale_running = self.ssm.normal.update_moving_avg(
            output_scale_running, new_term, num=num_data
        )

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
        ssm: Any,
        re_linearize_after_calibration=False,
        update_at_init: bool = False,
    ):
        super().__init__(strategy=strategy, ssm=ssm, prior=prior, constraint=constraint)
        self.re_linearize_after_calibration = re_linearize_after_calibration
        self.update_at_init = update_at_init

    def init(self, t, u, *, damp) -> ProbabilisticSolution:
        u, prediction = self.strategy.init_posterior(u=u)
        lin_state = self.constraint.init_linearization()

        output_scale = np.ones_like(self.ssm.prototypes.output_scale_calibrated())

        fx, lin_state = self.constraint.linearize(
            u.marginals, lin_state, damp=damp, t=t
        )
        # TODO: avoid the linearization altogether if update is false.
        #  use jax.eval_shape instead.
        if self.update_at_init:
            raise RuntimeError
            _, reverted = self.ssm.conditional.revert(u.marginals, fx)
            updates = reverted.noise
            u, posterior = self.strategy.apply_updates(prediction, updates=updates)
        fx = tree.tree_map(np.zeros_like, fx)
        posterior = prediction
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
        ones = np.ones_like(self.ssm.prototypes.output_scale_calibrated())
        transition = self.prior(dt, ones)
        mean = state.u.marginals.eval_mean()
        u = self.ssm.conditional.apply(mean, transition)

        # Linearize

        fx, lin_state = self.constraint.linearize(
            u, state=lin_state, damp=damp, t=state.t + dt
        )
        observed = self.ssm.conditional.marginalise(u, fx)
        output_scale = observed.mahalanobis_norm_relative(0.0)

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
        _, reverted = self.ssm.conditional.revert(u.marginals, fx)
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
        ssm: Any,
        strategy: MarkovStrategy,
        constraint_init: Constraint | None = None,
    ):
        super().__init__(strategy=strategy, ssm=ssm, prior=prior, constraint=constraint)
        self.constraint_init = constraint_init

    def init(self, t: Array, u: TaylorCoeffTarget, *, damp) -> ProbabilisticSolution:
        u_pred, prediction = self.strategy.init_posterior(u=u)

        if self.constraint_init is not None:
            co = self.constraint.init_linearization()
            fx, _co = self.constraint_init.linearize(
                rv=u_pred.marginals, state=co, damp=damp, t=t
            )
            _, reverted = self.ssm.conditional.revert(u_pred.marginals, fx)
            u, posterior = self.strategy.apply_updates(
                prediction, updates=reverted.noise
            )
        else:
            u, posterior = u_pred, prediction

        cstate = self.constraint.init_linearization()
        # TODO: replace the below with jax.eval_shape
        # because we don't really want to evaluate anything here
        fx, cstate = self.constraint.linearize(
            rv=u_pred.marginals, state=cstate, damp=damp, t=t
        )
        fx = tree.tree_map(np.zeros_like, fx)
        output_scale = np.ones_like(self.ssm.prototypes.output_scale_calibrated())
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
        _, reverted = self.ssm.conditional.revert(u_pred.marginals, fx)
        u, posterior = self.strategy.apply_updates(prediction, updates=reverted.noise)

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


class solver_iterated(ProbabilisticSolver):
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
        ssm: Any,
        strategy: MarkovStrategy,
        constraint_init: Constraint | None = None,
        tol: float | None = None,
        maxiter: int = 10,
        while_loop=flow.while_loop,
    ):
        super().__init__(strategy=strategy, ssm=ssm, prior=prior, constraint=constraint)
        self.constraint_init = constraint_init

        float_eps = np.finfo_eps(np.asarray(1.0).dtype)
        self.tol = tol if tol is not None else float_eps**0.5
        self.maxiter = maxiter
        self.while_loop = while_loop

    # TODO: move this outside the init and reuse in the step
    @tree.register_dataclass
    @structs.dataclass
    class IState:
        """A data structure for carrying state in iterated solvers."""

        do_continue: float
        i: int
        linearize_at: Any
        cstate: Any  # constraint-state
        updated_u: Any
        updated_posterior: Any
        fx: Any = None

    def make_body_fun(self, u_pred, prediction, *, constraint, damp, t):
        """Create the body-function for the while-loop in an iterated solver."""

        def body_fun(carry: solver_iterated.IState) -> solver_iterated.IState:
            # _do_continue, i, lin_at, corrstate, _updated = carry
            # Linearize
            fx, cstate = constraint.linearize(
                carry.linearize_at, carry.cstate, damp=damp, t=t
            )

            # Update
            _, reverted = self.ssm.conditional.revert(u_pred.marginals, fx)
            u_upd, posterior = self.strategy.apply_updates(
                prediction, updates=reverted.noise
            )

            diff = u_upd.marginals.mean - carry.linearize_at.mean
            u = u_upd

            diff_norm = linalg.vector_norm(diff) / np.sqrt(diff.size)
            i = carry.i + 1
            do_continue = np.logical_and(diff_norm > self.tol, i < self.maxiter)
            return self.IState(
                do_continue=do_continue,
                i=i,
                linearize_at=u.marginals,
                cstate=cstate,
                updated_u=u_upd,
                updated_posterior=posterior,
                fx=fx if carry.fx is not None else None,
            )

        return body_fun

    def init(self, t: Array, u: TaylorCoeffTarget, *, damp) -> ProbabilisticSolution:
        u_pred, prediction = self.strategy.init_posterior(u=u)

        if self.constraint_init is not None:
            cstate = self.constraint_init.init_linearization()
            init = self.IState(
                do_continue=True,
                i=0,
                linearize_at=u_pred.marginals,
                cstate=cstate,
                updated_u=u_pred,
                updated_posterior=prediction,
            )
            body_fun = self.make_body_fun(
                u_pred, prediction, constraint=self.constraint_init, damp=damp, t=t
            )
            final = self.while_loop(lambda s: s.do_continue, body_fun, init)
            (u, posterior) = final.updated_u, final.updated_posterior
        else:
            u, posterior = u_pred, prediction

        cstate = self.constraint.init_linearization()
        fx, cstate = self.constraint.linearize(
            rv=u_pred.marginals, state=cstate, damp=damp, t=t
        )

        output_scale = np.ones_like(self.ssm.prototypes.output_scale_calibrated())

        # TODO: the number of function evaluations should also be reflected
        # in the solution object, so that we can communicate if an iterative
        # solver took excessive attempts or so
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

        # Iterated update
        body_fun = self.make_body_fun(
            u_pred, prediction, constraint=self.constraint, damp=damp, t=state.t + dt
        )
        init = self.IState(
            do_continue=True,
            i=0,
            linearize_at=u_pred.marginals,
            cstate=state.auxiliary,
            updated_u=u_pred,
            updated_posterior=prediction,
            fx=state.fun_evals,
        )
        final = self.while_loop(lambda s: s.do_continue, body_fun, init)
        return ProbabilisticSolution(
            t=state.t + dt,
            u=final.updated_u,
            solution_full=final.updated_posterior,
            output_scale=output_scale,
            auxiliary=final.cstate,
            num_steps=state.num_steps + 1,
            fun_evals=final.fx,
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
        ssm: Any,
        error_norm: Callable | None = None,
        re_linearize_before_error: bool = False,  # cache by default
    ):
        if error_norm is None:
            error_norm = error_norm_scale_then_rms()

        self.error_norm = error_norm
        self.constraint = constraint
        self.prior = prior
        self.ssm = ssm
        self.re_linearize_before_error = re_linearize_before_error

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
        output_scale = np.ones_like(self.ssm.prototypes.output_scale_calibrated())
        transition = self.prior(dt, output_scale)

        # Extrapolate from the zero-error state
        # TODO: should conditional.apply take the Pytree mean?
        mean = previous.u.marginals.eval_mean()
        rv = self.ssm.conditional.apply(mean, transition)

        # Optionally: re-linearize
        if self.re_linearize_before_error:
            linearized, state = self.constraint.linearize(
                rv, state, damp=damp, t=proposed.t
            )
        else:
            linearized = proposed.fun_evals

        # Extract the local residual stdev from the linearization
        observed = self.ssm.conditional.marginalise(rv, linearized)
        output_scale = observed.mahalanobis_norm_relative(0.0)
        observed = observed.rescale_cholesky(output_scale)
        error = observed.eval_standard_deviation()
        error, _ = tree.ravel_pytree(error)

        # Compute a reference
        u0 = tree.tree_leaves(previous.u.mean)[0]
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
        error_abs = error * dt**n / np.factorial(n)
        error_norm = self.error_norm(error_abs, reference, atol=atol, rtol=rtol)

        # Scale the error norm with the error contraction rate and return
        error_contraction_rate = self.ssm.num_derivatives + 1
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
        ssm: Any,
        error_norm: Callable | None = None,
        re_linearize_before_error: bool = False,  # cache by default
        derivative_idx: int = 0,
    ):
        if error_norm is None:
            error_norm = error_norm_scale_then_rms()

        self.error_norm = error_norm
        self.constraint = constraint
        self.prior = prior
        self.ssm = ssm
        self.re_linearize_before_error = re_linearize_before_error
        self.derivative_idx = derivative_idx

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
        output_scale = np.ones_like(self.ssm.prototypes.output_scale_calibrated())
        transition = self.prior(dt, output_scale)

        # Extrapolate from the zero-error state
        mean = previous.u.marginals.eval_mean()
        rv = self.ssm.conditional.apply(mean, transition)

        # Optionally: re-linearize
        if self.re_linearize_before_error:
            linearized, state = self.constraint.linearize(
                rv, state, damp=damp, t=proposed.t
            )
        else:
            linearized = proposed.fun_evals

        # Extract the local residual stdev from the linearization
        observed, conditional = self.ssm.conditional.revert(rv, linearized)
        output_scale = observed.mahalanobis_norm_relative(0.0)

        # Measure error on the n-th state (usually, n=0 because why not)
        n = self.derivative_idx

        # *New:* Go back into solution space
        stdev = conditional.noise.eval_standard_deviation()[n]
        error, _ = tree.ravel_pytree(stdev)
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
        error_contraction_rate = self.ssm.num_derivatives + 1
        error_power = error_norm ** (-1.0 / error_contraction_rate)
        return error_power, state
