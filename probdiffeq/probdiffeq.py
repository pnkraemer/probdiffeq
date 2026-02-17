"""Probabilistic solvers for differential equations.

See the tutorials for example use cases.
"""

from probdiffeq.backend import flow, func, linalg, np, random, special, structs, tree
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
from probdiffeq.util import filter_util

C = TypeVar("C", bound=Sequence)
T = TypeVar("T")


class VectorField(Protocol[T]):
    """A protocol for the vector-field type expected by the solvers.

    Examples of valid call signatures:

    - `f(u: T, /, *, t: ArrayLike) -> T`
    - `f(u: T, du: T, /, *, t: ArrayLike) -> T`
    - `f(u: T, du: T, ddu: T, /, *, t: ArrayLike) -> T`
    - `f(u: T, du: T, ddu: T, dddy: T, /, *, t: ArrayLike) -> T`

    If a type-checker complains, try jitting the vector field.
    If the error persists, open an issue.
    """

    def __call__(self, *ys: T, t: ArrayLike) -> T: ...


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


def constraint_ode_ts0(ssm, ode_order=1):
    """Create an ODE constraint with zeroth-order Taylor linearisation.

    Related:
    [`Constraint`](#probdiffeq.probdiffeq.Constraint).
    """
    return ssm.linearize.ode_taylor_0th(ode_order=ode_order)


def constraint_root_ts1(root, *, ssm, jacobian=None, ode_order=1):
    """Construct a constraint based on a custom root.

    See the custom information operator tutorial for details.

    Related:
    [`Constraint`](#probdiffeq.probdiffeq.Constraint).
    """
    if jacobian is None:
        # Use hutchinson Jacobian handling for backward compatibility.
        jacobian = jacobian_hutchinson_fwd()
    return ssm.linearize.root_taylor_1st(root, ode_order=ode_order, jacobian=jacobian)


def constraint_ode_ts1(*, ssm, jacobian: JacobianHandler | None = None, ode_order=1):
    """Create an ODE constraint with first-order Taylor linearisation.

    Related:
    [`Constraint`](#probdiffeq.probdiffeq.Constraint).
    """
    if jacobian is None:
        # Use hutchinson Jacobian handling for backward compatibility.
        jacobian = jacobian_hutchinson_fwd()
    return ssm.linearize.ode_taylor_1st(ode_order=ode_order, jacobian=jacobian)


def constraint_ode_slr0(*, ssm, cubature_fun=cubature_third_order_spherical):
    """Create an ODE constraint with zeroth-order statistical linear regression.

    Related:
    [`Constraint`](#probdiffeq.probdiffeq.Constraint).
    """
    return ssm.linearize.ode_statistical_0th(cubature_fun)


def constraint_ode_slr1(*, ssm, cubature_fun=cubature_third_order_spherical):
    """Create an ODE constraint with first-order statistical linear regression.

    Related:
    [`Constraint`](#probdiffeq.probdiffeq.Constraint).

    """
    return ssm.linearize.ode_statistical_1st(cubature_fun)


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

    def markov_marginals(self, markov_seq: MarkovSequence, *, reverse: bool):
        """Extract the (time-)marginals from a Markov sequence.

        This is only needed in combination with smoothing-based strategies.
        """
        if markov_seq.marginal.mean.ndim == markov_seq.conditional.noise.mean.ndim:
            markov_seq = self._markov_select_terminal(markov_seq)

        def step(x, cond):
            extrapolated = self.ssm.conditional.marginalise(x, cond)
            return extrapolated, extrapolated

        init, xs = markov_seq.marginal, markov_seq.conditional
        _, marginals = flow.scan(step, init=init, xs=xs, reverse=reverse)

        if reverse:
            # Append the terminal marginal to the computed ones
            return tree.tree_array_append(marginals, init)

        return tree.tree_array_prepend(init, marginals)

    def markov_sample(
        self, key, markov_seq: MarkovSequence, *, reverse: bool, shape: tuple = ()
    ):
        """Sample from a Markov sequence."""
        if markov_seq.marginal.mean.ndim == markov_seq.conditional.noise.mean.ndim:
            markov_seq = self._markov_select_terminal(markov_seq)
        # A smoother samples on the grid by sampling i.i.d values
        # from the terminal RV x_N and the backward noises z_(1:N)
        # and then combining them backwards as
        # x_(n-1) = l_n @ x_n + z_n, for n=1,...,N.
        markov_seq_shape = self._sample_shape(markov_seq)
        base_samples = random.normal(key, shape=shape + markov_seq_shape)
        return self._transform_unit_sample(markov_seq, base_samples, reverse=reverse)

    @staticmethod
    def _markov_select_terminal(markov_seq):
        """Discard all intermediate filtering solutions from a Markov sequence.

        This function is useful to convert a smoothing-solution into a Markov sequence
        that is compatible with sampling or marginalisation.
        """
        init = tree.tree_map(lambda x: x[-1, ...], markov_seq.marginal)
        return MarkovSequence(init, markov_seq.conditional)

    def _sample_shape(self, markov_seq):
        # The number of samples is one larger than the number of conditionals
        noise = markov_seq.conditional.noise
        n, *shape_single_sample = self.ssm.stats.hidden_shape(noise)
        return n + 1, *tuple(shape_single_sample)

    def _transform_unit_sample(self, markov_seq, base_sample, *, reverse):
        if base_sample.ndim > len(self._sample_shape(markov_seq)):
            transform = func.partial(self._transform_unit_sample, reverse=reverse)
            transform_vmap = func.vmap(transform, in_axes=(None, 0))
            return transform_vmap(markov_seq, base_sample)

        # Compute a single unit sample.

        def body_fun(samp_prev, conditionals_and_base_samples):
            conditional, base = conditionals_and_base_samples

            rv = self.ssm.conditional.apply(samp_prev, conditional)
            smp = self.ssm.stats.transform_unit_sample(base, rv)
            return smp, smp

        base_sample_init, base_sample_body = base_sample[0], base_sample[1:]

        # Compute a sample at the terminal value
        init_sample = self.ssm.stats.transform_unit_sample(
            base_sample_init, markov_seq.marginal
        )

        # Loop over backward models and the remaining base samples
        xs = (markov_seq.conditional, base_sample_body)
        _, samples = flow.scan(body_fun, init=init_sample, xs=xs, reverse=reverse)

        if reverse:
            samples = np.concatenate([samples, init_sample[None, ...]])
        else:
            samples = np.concatenate([init_sample[None, ...], samples])

        return func.vmap(self.ssm.stats.qoi_from_sample)(samples)

    def log_marginal_likelihood_terminal_values(
        self, u, /, *, standard_deviation, posterior: T
    ):
        """Compute the log-marginal-likelihood at the terminal value.

        Parameters
        ----------
        u
            Observation. Expected to have shape (d,) for an ODE with shape (d,).
        standard_deviation
            Standard deviation of the observation. Expected to be a scalar.
        posterior
            Posterior distribution.
            Expected to correspond to a solution of an ODE with shape (d,).
        """
        [u_leaves], u_structure = tree.tree_flatten(u)
        [_std_leaves], std_structure = tree.tree_flatten(standard_deviation)

        if u_structure != std_structure:
            msg = (
                f"Observation-noise tree structure {std_structure} "
                f"does not match the observation structure {u_structure}. "
            )
            raise ValueError(msg)

        # Generate an observation-model for the QOI
        model = self.ssm.conditional.to_derivative(0, u, standard_deviation)
        rv = posterior.marginal if isinstance(posterior, MarkovSequence) else posterior

        data = np.zeros_like(u_leaves)  # 'u' is baked into the observation model
        observed, _conditional = self.ssm.conditional.revert(rv, model)
        return self.ssm.stats.logpdf(data, observed)

    def log_marginal_likelihood(self, u, /, *, standard_deviation, posterior: T):
        """Compute the log-marginal-likelihood of observations of the IVP solution.

        Only works with smoothing-based estimators.

        !!! note
            Use `log_marginal_likelihood_terminal_values`
            to compute the log-likelihood at the terminal values.

        Parameters
        ----------
        u
            Observation. Expected to match the ODE's type/shape.
        standard_deviation
            Standard deviation of the observation. Expected to match 'u's
            Pytree structure, but every leaf must be a scalar.
        posterior
            Posterior distribution.
            Expected to correspond to a solution of an ODE with shape (d,).
        """
        [u_leaves], u_structure = tree.tree_flatten(u)
        [std_leaves], std_structure = tree.tree_flatten(standard_deviation)

        if u_structure != std_structure:
            msg = (
                f"Observation-noise tree structure {std_structure} "
                f"does not match the observation structure {u_structure}. "
            )
            raise ValueError(msg)

        qoi_flat, _ = tree.ravel_pytree(self.ssm.prototypes.qoi())
        if np.ndim(std_leaves) < 1 or np.ndim(u_leaves) != np.ndim(qoi_flat) + 1:
            msg = (
                f"Time-series solution expected. "
                f"ndim={np.ndim(u_leaves)}, shape={np.shape(u_leaves)} received."
            )
            raise ValueError(msg)

        if len(u_leaves) != len(np.asarray(std_leaves)):
            msg = (
                f"Observation-noise shape {np.shape(std_leaves)} "
                f"does not match the observation shape {np.shape(u_leaves)}. "
            )
            raise ValueError(msg)

        if not isinstance(posterior, MarkovSequence):
            msg1 = "Time-series marginal likelihoods "
            msg2 = "cannot be computed with a filtering solution."
            raise TypeError(msg1 + msg2)

        # Generate an observation-model for the QOI

        model_fun = func.vmap(self.ssm.conditional.to_derivative, in_axes=(None, 0, 0))
        models = model_fun(0, u, standard_deviation)

        # Select the terminal variable
        rv = tree.tree_map(lambda s: s[-1, ...], posterior.marginal)

        # Run the reverse Kalman filter
        estimator = filter_util.kalmanfilter_with_marginal_likelihood(ssm=self.ssm)
        result, _ = filter_util.estimate_rev(
            np.zeros_like(u_leaves),
            init=rv,
            prior_transitions=posterior.conditional,
            observation_model=models,
            estimator=estimator,
        )

        # Return only the logpdf
        return result.logpdf


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
        vector_field: VectorField,
        *,
        strategy: MarkovStrategy,
        prior: Callable,
        constraint: Constraint,
        ssm: Any,
    ):
        self.vector_field = vector_field
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
            u=interp_from.u,  # incorrect?
            output_scale=interp_from.output_scale,  # incorrect?
            auxiliary=interp_from.auxiliary,  # incorrect?
            num_steps=interp_from.num_steps,  # incorrect?
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
    tcoeffs_std: C | None = None,
    ssm_fact: Literal["dense", "isotropic", "blockdiag"] = "dense",  # noqa: F821
    output_scale: ArrayLike | None = None,
    increase_std_by_eps: bool = True,
):
    """Construct an repeatedly-integrated Wiener process.

    Tip: Choose nonzero standard deviations
    to get visually more pleasing uncertainties and more numerical robustness for
    high-order solvers in low precision arithmetic. Outside of these cases,
    leave the standard deviations at zero to improve accuracy.
    """
    # Choose a state-space model factorisation
    ssm = impl.choose(ssm_fact, tcoeffs_like=tcoeffs)

    # Set up the transitions

    # TODO: should the output_scale be an argument to solve()?
    if output_scale is None:
        output_scale = np.ones_like(ssm.prototypes.output_scale())

    output_scale = np.asarray(output_scale)
    discretize = ssm.conditional.ibm_transitions(base_scale=output_scale)

    if tcoeffs_std is None:
        error_like = np.zeros_like(ssm.prototypes.error_estimate())
        tcoeffs_std = tree.tree_map(lambda _: error_like, tcoeffs)

    # Increase the Taylor coefficient STD by a machine epsilon
    # because the solver initialisation carries out an update
    # and if the inputs are fully certain, this update yields NaNs.
    if increase_std_by_eps:
        eps = np.finfo_eps(output_scale.dtype)
        tcoeffs_std = tree.tree_map(lambda s: s + eps, tcoeffs_std)

    # Return the target
    marginal = ssm.normal.from_tcoeffs(tcoeffs, tcoeffs_std)
    u_mean = ssm.stats.qoi(marginal)
    std = ssm.stats.standard_deviation(marginal)
    u_std = ssm.stats.qoi_from_sample(std)
    target = TaylorCoeffTarget(u_mean, u_std, marginal)
    return target, discretize, ssm


def prior_wiener_integrated_discrete(
    ts,
    tcoeffs: C,
    *,
    tcoeffs_std: C | None = None,
    ssm_fact: Literal["dense", "isotropic", "blockdiag"] = "dense",  # noqa: F821
    output_scale: ArrayLike | None = None,
    increase_std_by_eps: bool = True,
):
    """Compute a time-discretization of an integrated Wiener process."""
    init, discretize, ssm = prior_wiener_integrated(
        tcoeffs,
        tcoeffs_std=tcoeffs_std,
        ssm_fact=ssm_fact,
        output_scale=output_scale,
        increase_std_by_eps=increase_std_by_eps,
    )
    scales = np.ones_like(ssm.prototypes.output_scale())
    discretize_vmap = func.vmap(discretize, in_axes=(0, None))
    conditionals = discretize_vmap(np.diff(ts), scales)
    markov_seq = MarkovSequence(init.marginals, conditionals)
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
        posterior = MarkovSequence(marginal=u.marginals, conditional=cond)
        return u, posterior

    def predict(
        self, posterior: MarkovSequence, *, transition
    ) -> tuple[TaylorCoeffTarget, MarkovSequence]:
        marginals, cond = self.ssm.conditional.revert(posterior.marginal, transition)
        posterior = MarkovSequence(marginal=marginals, conditional=cond)

        u = self.ssm.stats.qoi(marginals)
        std = self.ssm.stats.standard_deviation(marginals)
        u_std = self.ssm.stats.qoi_from_sample(std)
        estimate = TaylorCoeffTarget(u, u_std, marginals)
        return estimate, posterior

    def apply_updates(self, prediction, *, updates):
        posterior = MarkovSequence(updates, prediction.conditional)
        marginals = updates
        u = self.ssm.stats.qoi(marginals)
        std = self.ssm.stats.standard_deviation(marginals)
        u_std = self.ssm.stats.qoi_from_sample(std)
        return TaylorCoeffTarget(u, u_std, marginals), posterior

    def finalize(
        self, *, posterior0: MarkovSequence, posterior: MarkovSequence, output_scale
    ):
        assert output_scale.shape == self.ssm.prototypes.output_scale().shape

        init0 = self.ssm.stats.rescale_cholesky(posterior0.marginal, output_scale)
        conditional0 = self.ssm.conditional.rescale_noise(
            posterior0.conditional, output_scale
        )
        posterior0 = MarkovSequence(init0, conditional0)

        # Calibrate
        init = self.ssm.stats.rescale_cholesky(posterior.marginal, output_scale)
        conditional = self.ssm.conditional.rescale_noise(
            posterior.conditional, output_scale
        )
        posterior = MarkovSequence(init, conditional)

        # Marginalise
        marginals = self.markov_marginals(posterior, reverse=True)

        # Prepend the initial condition to the filtering distributions
        init = tree.tree_array_prepend(posterior0.marginal, posterior.marginal)
        posterior = MarkovSequence(marginal=init, conditional=posterior.conditional)

        # Extract targets
        u = self.ssm.stats.qoi(marginals)
        std = self.ssm.stats.standard_deviation(marginals)
        u_std = self.ssm.stats.qoi_from_sample(std)
        estimate = TaylorCoeffTarget(u, u_std, marginals)
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
        solution_at_t = MarkovSequence(rv_at_t, extrapolated_t.conditional)

        # The state at t1 gets a new backward model;
        # (it must remember how to get back to t, not to t0).
        solution_at_t1 = MarkovSequence(marginal_t1, conditional_t1_to_t)
        interp_res = _InterpRes(step_from=solution_at_t1, interp_from=solution_at_t)

        # Extract targets
        marginals = solution_at_t.marginal
        u = self.ssm.stats.qoi(marginals)
        std = self.ssm.stats.standard_deviation(marginals)
        u_std = self.ssm.stats.qoi_from_sample(std)
        estimate = TaylorCoeffTarget(u, u_std, marginals)
        return (estimate, solution_at_t), interp_res

    def interpolate_at_t1(self, posterior_t1):
        marginals = posterior_t1.marginal
        u = self.ssm.stats.qoi(marginals)
        std = self.ssm.stats.standard_deviation(marginals)
        u_std = self.ssm.stats.qoi_from_sample(std)
        estimate = TaylorCoeffTarget(u, u_std, marginals)

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
        u = self.ssm.stats.qoi(marginals)
        std = self.ssm.stats.standard_deviation(marginals)
        u_std = self.ssm.stats.qoi_from_sample(std)
        estimate = TaylorCoeffTarget(u, u_std, marginals)
        return estimate, marginals

    def apply_updates(self, prediction, *, updates):
        del prediction
        marginals = updates
        u = self.ssm.stats.qoi(marginals)
        std = self.ssm.stats.standard_deviation(marginals)
        u_std = self.ssm.stats.qoi_from_sample(std)
        estimate = TaylorCoeffTarget(u, u_std, marginals)
        return estimate, marginals

    def finalize(self, *, posterior0, posterior, output_scale):
        assert output_scale.shape == self.ssm.prototypes.output_scale().shape

        # No rescaling because no calibration at the initial step
        posterior0 = self.ssm.stats.rescale_cholesky(posterior0, output_scale)

        # Calibrate
        posterior = self.ssm.stats.rescale_cholesky(posterior, output_scale)

        # Stack
        posterior = tree.tree_array_prepend(posterior0, posterior)

        marginals = posterior
        u = self.ssm.stats.qoi(marginals)
        std = self.ssm.stats.standard_deviation(marginals)
        u_std = self.ssm.stats.qoi_from_sample(std)
        estimate = TaylorCoeffTarget(u, u_std, marginals)
        return estimate, posterior

    def interpolate(
        self, *, posterior_t0, posterior_t1, transition_t0_t, transition_t_t1
    ):
        del transition_t_t1
        _, interpolated = self.predict(
            posterior=posterior_t0, transition=transition_t0_t
        )

        u = self.ssm.stats.qoi(interpolated)
        std = self.ssm.stats.standard_deviation(interpolated)
        u_std = self.ssm.stats.qoi_from_sample(std)
        marginals = interpolated
        estimate = TaylorCoeffTarget(u, u_std, marginals)

        interp_res = _InterpRes(step_from=posterior_t1, interp_from=interpolated)
        return (estimate, interpolated), interp_res

    def interpolate_at_t1(self, *, posterior_t1):
        u = self.ssm.stats.qoi(posterior_t1)
        std = self.ssm.stats.standard_deviation(posterior_t1)
        u_std = self.ssm.stats.qoi_from_sample(std)
        marginals = posterior_t1
        estimate = TaylorCoeffTarget(u, u_std, marginals)

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
        posterior = MarkovSequence(u.marginals, cond)
        return u, posterior

    def predict(
        self, posterior: MarkovSequence, *, transition
    ) -> tuple[TaylorCoeffTarget, MarkovSequence]:
        rv = posterior.marginal
        bw0 = posterior.conditional
        marginals, cond = self.ssm.conditional.revert(rv, transition)
        cond = self.ssm.conditional.merge(bw0, cond)
        predicted = MarkovSequence(marginals, cond)

        u = self.ssm.stats.qoi(marginals)
        std = self.ssm.stats.standard_deviation(marginals)
        u_std = self.ssm.stats.qoi_from_sample(std)
        estimate = TaylorCoeffTarget(u, u_std, marginals)
        return estimate, predicted

    def apply_updates(self, prediction: MarkovSequence, *, updates):
        posterior = MarkovSequence(updates, prediction.conditional)
        rv = updates
        u = self.ssm.stats.qoi(rv)
        std = self.ssm.stats.standard_deviation(rv)
        u_std = self.ssm.stats.qoi_from_sample(std)
        marginals = rv
        estimate = TaylorCoeffTarget(u, u_std, marginals)
        return estimate, posterior

    def finalize(
        self, *, posterior0: MarkovSequence, posterior: MarkovSequence, output_scale
    ):
        assert output_scale.shape == self.ssm.prototypes.output_scale().shape
        init = self.ssm.stats.rescale_cholesky(posterior0.marginal, output_scale)
        conditional = self.ssm.conditional.rescale_noise(
            posterior0.conditional, output_scale
        )
        posterior0 = MarkovSequence(init, conditional)

        # Calibrate the time series
        init = self.ssm.stats.rescale_cholesky(posterior.marginal, output_scale)
        conditional = self.ssm.conditional.rescale_noise(
            posterior.conditional, output_scale
        )
        posterior = MarkovSequence(init, conditional)

        # Marginalise
        marginals = self.markov_marginals(posterior, reverse=True)

        # Prepend the initial condition to the filtering distributions
        init = tree.tree_array_prepend(posterior0.marginal, posterior.marginal)
        posterior = MarkovSequence(marginal=init, conditional=posterior.conditional)

        # Extract targets
        u = self.ssm.stats.qoi(marginals)
        std = self.ssm.stats.standard_deviation(marginals)
        u_std = self.ssm.stats.qoi_from_sample(std)
        estimate = TaylorCoeffTarget(u, u_std, marginals)
        return estimate, posterior

    def interpolate_at_t1(self, *, posterior_t1: MarkovSequence):
        cond_identity = self.ssm.conditional.identity(self.ssm.num_derivatives + 1)
        resume_from = MarkovSequence(posterior_t1.marginal, conditional=cond_identity)
        interp_res = _InterpRes(step_from=resume_from, interp_from=resume_from)

        interpolated = posterior_t1
        marginals = interpolated.marginal
        u = self.ssm.stats.qoi(marginals)
        std = self.ssm.stats.standard_deviation(marginals)
        u_std = self.ssm.stats.qoi_from_sample(std)
        estimate = TaylorCoeffTarget(u, u_std, marginals)
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
        previous_new = MarkovSequence(extrapolated_t.marginal, conditional_id)
        _, extrapolated_t1 = self.predict(
            posterior=previous_new, transition=transition_t_t1
        )

        # Marginalise from t1 to t to obtain the interpolated solution.
        marginal_t1 = posterior_t1.marginal
        conditional_t1_to_t = extrapolated_t1.conditional
        rv_at_t = self.ssm.conditional.marginalise(marginal_t1, conditional_t1_to_t)

        # Return the right combination of marginals and conditionals.
        interpolated = MarkovSequence(rv_at_t, extrapolated_t.conditional)
        step_from = MarkovSequence(
            posterior_t1.marginal, conditional=conditional_t1_to_t
        )
        interp_res = _InterpRes(step_from=step_from, interp_from=previous_new)

        marginals = interpolated.marginal
        u = self.ssm.stats.qoi(marginals)
        std = self.ssm.stats.standard_deviation(marginals)
        u_std = self.ssm.stats.qoi_from_sample(std)
        estimate = TaylorCoeffTarget(u, u_std, marginals)
        return (estimate, interpolated), interp_res


class solver_mle(ProbabilisticSolver):
    """Create a solver that uses maximum-likelihood calibration for the output scale.

    Related:
    [`ProbabilisticSolver`](#probdiffeq.probdiffeq.ProbabilisticSolver).

    """

    def __init__(
        self,
        vector_field: VectorField,
        *,
        constraint: Constraint,
        prior: Callable,
        ssm: Any,
        strategy: MarkovStrategy,
        increase_init_damp_by_eps: bool = True,
    ):
        super().__init__(
            vector_field, strategy=strategy, ssm=ssm, prior=prior, constraint=constraint
        )
        self.increase_init_damp_by_eps = increase_init_damp_by_eps

    def init(self, t, u: TaylorCoeffTarget, *, damp: float) -> ProbabilisticSolution:
        estimate, prediction = self.strategy.init_posterior(u=u)
        correction_state = self.constraint.init_linearization()

        output_scale_prior = np.ones_like(self.ssm.prototypes.output_scale())

        # Increase the damping by machine epsilon because often,
        # the initial taylor coefficients have zero standard deviation
        # in which case the correction below would yield NaNs.
        if self.increase_init_damp_by_eps:
            damp = np.asarray(damp)
            damp = damp + np.finfo_eps(damp)

        # Update
        f_wrapped = func.partial(self.vector_field, t=t)
        fx, correction_state = self.constraint.linearize(
            f_wrapped, estimate.marginals, correction_state, damp=damp
        )
        observed, reverted = self.ssm.conditional.revert(u.marginals, fx)
        updates = reverted.noise
        u, posterior = self.strategy.apply_updates(
            prediction=prediction, updates=updates
        )

        # Calibrate the output scale
        output_scale_running = self.ssm.stats.mahalanobis_norm_relative(0.0, observed)

        auxiliary = (correction_state, output_scale_running, 1)
        return ProbabilisticSolution(
            t=t,
            u=estimate,
            solution_full=posterior,
            auxiliary=auxiliary,
            output_scale=output_scale_prior,
            num_steps=0,
            fun_evals=fx,
        )

    def step(self, state, *, dt: float, damp: float):
        # Discretize
        output_scale = np.ones_like(self.ssm.prototypes.output_scale())
        transition = self.prior(dt, output_scale)

        # Predict
        u, prediction = self.strategy.predict(
            posterior=state.solution_full, transition=transition
        )

        # Linearize
        (lin_state, output_scale_running, num_data) = state.auxiliary
        f_wrapped = func.partial(self.vector_field, t=state.t + dt)
        fx, correction_state = self.constraint.linearize(
            f_wrapped, u.marginals, state=lin_state, damp=damp
        )

        # Do the full correction step
        observed, reverted = self.ssm.conditional.revert(u.marginals, fx)
        updates = reverted.noise
        u, posterior = self.strategy.apply_updates(
            prediction=prediction, updates=updates
        )

        # Calibrate the output scale
        new_term = self.ssm.stats.mahalanobis_norm_relative(0.0, observed)
        output_scale_running = self.ssm.stats.update_mean(
            output_scale_running, new_term, num=num_data
        )

        # Return the state
        auxiliary = (correction_state, output_scale_running, num_data + 1)
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
        vector_field: VectorField,
        *,
        strategy: MarkovStrategy,
        prior: Callable,
        constraint: Constraint,
        ssm: Any,
        re_linearize_after_calibration=False,
        increase_init_damp_by_eps: bool = True,
    ):
        super().__init__(
            vector_field, strategy=strategy, ssm=ssm, prior=prior, constraint=constraint
        )
        self.re_linearize_after_calibration = re_linearize_after_calibration
        self.increase_init_damp_by_eps = increase_init_damp_by_eps

    def init(self, t, u, *, damp) -> ProbabilisticSolution:
        u, prediction = self.strategy.init_posterior(u=u)
        lin_state = self.constraint.init_linearization()

        output_scale = np.ones_like(self.ssm.prototypes.output_scale())

        # Increase the damping by machine epsilon because often,
        # the initial taylor coefficients have zero standard deviation
        # in which case the correction below would yield NaNs.
        if self.increase_init_damp_by_eps:
            damp = np.asarray(damp)
            damp = damp + np.finfo_eps(damp)

        f_wrapped = func.partial(self.vector_field, t=t)
        fx, lin_state = self.constraint.linearize(
            f_wrapped, u.marginals, lin_state, damp=damp
        )

        _, reverted = self.ssm.conditional.revert(u.marginals, fx)
        updates = reverted.noise
        u, posterior = self.strategy.apply_updates(prediction, updates=updates)

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
        ones = np.ones_like(self.ssm.prototypes.output_scale())
        transition = self.prior(dt, ones)
        mean = self.ssm.stats.mean(state.u.marginals)
        u = self.ssm.conditional.apply(mean, transition)

        # Linearize
        f_wrapped = func.partial(self.vector_field, t=state.t + dt)
        fx, lin_state = self.constraint.linearize(
            f_wrapped, u, state=lin_state, damp=damp
        )
        observed = self.ssm.conditional.marginalise(u, fx)
        output_scale = self.ssm.stats.mahalanobis_norm_relative(0.0, rv=observed)

        # Do the full extrapolation with the calibrated output scale
        # (Includes re-discretisation)
        transition = self.prior(dt, output_scale)
        u, prediction = self.strategy.predict(
            state.solution_full, transition=transition
        )

        # Relinearize
        if self.re_linearize_after_calibration:
            fx, lin_state = self.constraint.linearize(
                f_wrapped, u.marginals, state=lin_state, damp=damp
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
        vector_field: VectorField,
        *,
        constraint: Constraint,
        prior: Callable,
        ssm: Any,
        strategy: MarkovStrategy,
        increase_init_damp_by_eps: bool = True,
    ):
        super().__init__(
            vector_field, strategy=strategy, ssm=ssm, prior=prior, constraint=constraint
        )
        self.increase_init_damp_by_eps = increase_init_damp_by_eps

    def init(
        self, t: Array, u: TaylorCoeffTarget, *, damp: float
    ) -> ProbabilisticSolution:
        u, prediction = self.strategy.init_posterior(u=u)

        correction_state = self.constraint.init_linearization()

        # Increase the damping by machine epsilon because often,
        # the initial taylor coefficients have zero standard deviation
        # in which case the correction below would yield NaNs.
        if self.increase_init_damp_by_eps:
            damp = np.asarray(damp)
            damp = damp + np.finfo_eps(damp)

        f_wrapped = func.partial(self.vector_field, t=t)
        fx, correction_state = self.constraint.linearize(
            f_wrapped, rv=u.marginals, state=correction_state, damp=damp
        )
        _, reverted = self.ssm.conditional.revert(u.marginals, fx)
        u, posterior = self.strategy.apply_updates(prediction, updates=reverted.noise)

        output_scale = np.ones_like(self.ssm.prototypes.output_scale())
        return ProbabilisticSolution(
            t=t,
            u=u,
            solution_full=posterior,
            num_steps=0,
            auxiliary=correction_state,
            output_scale=output_scale,
            fun_evals=fx,
        )

    def step(self, state: ProbabilisticSolution, *, dt, damp):
        # Discretize
        output_scale = np.ones_like(state.output_scale)
        transition = self.prior(dt, output_scale)

        # Predict
        u, prediction = self.strategy.predict(
            state.solution_full, transition=transition
        )

        # Linearize
        f_eval = func.partial(self.vector_field, t=state.t + dt)
        fx, auxiliary = self.constraint.linearize(
            f_eval, u.marginals, state.auxiliary, damp=damp
        )

        # Update
        _, reverted = self.ssm.conditional.revert(u.marginals, fx)
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


def errorest_error_norm_scale_then_rms(*, norm_order=None) -> Callable:
    """Normalize an error by scaling followed by computing the root-mean-square norm.

    This is the recommended approach, and there is no reason to choose
    [`errorest_error_norm_rms_then_scale`](#probdiffeq.probdiffeq.errorest_error_norm_rms_then_scale),
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


def errorest_error_norm_rms_then_scale(norm_order=None) -> Callable:
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
    [`errorest_local_residual`](#probdiffeq.probdiffeq.errorest_local_residual),
    [`errorest_local_residual_cached`](#probdiffeq.probdiffeq.errorest_local_residual_cached).

    """

    def init_errorest(self):
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


class errorest_local_residual_cached(ErrorEstimator):
    """Construct an error estimator based on a **cached** local residual.

    'Cached' refers to the fact that the vector field is not evaluated
    again, but the linearisation from the step itself is reused.

    See the docstring of the non-cached version for more details.

    Related:
    [`ErrorEstimator`](#probdiffeq.probdiffeq.ErrorEstimator),
    [`errorest_local_residual`](#probdiffeq.probdiffeq.errorest_local_residual).

    """

    def __init__(self, prior: Any, ssm: Any, error_norm: Callable | None = None):
        if error_norm is None:
            error_norm = errorest_error_norm_scale_then_rms()

        self.error_norm = error_norm

        self.prior = prior
        self.ssm = ssm

    def init_errorest(self) -> tuple:
        return ()

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
    ) -> tuple[float, tuple]:
        del damp  # unused because no additional linearisation
        del state  # unused because state-free method

        # Discretize; The output scale is set to one
        # since the error is multiplied with a local scale estimate anyway
        output_scale = np.ones_like(self.ssm.prototypes.output_scale())
        transition = self.prior(dt, output_scale)

        # Estimate the error
        mean = self.ssm.stats.mean(previous.u.marginals)
        mean_extra = self.ssm.conditional.apply(mean, transition)
        error = self._linearize_and_estimate(mean_extra, linearized=proposed.fun_evals)

        # Compute a reference
        u0 = tree.tree_leaves(previous.u.mean)[0]
        u1 = tree.tree_leaves(proposed.u.mean)[0]
        reference = np.maximum(np.abs(u0), np.abs(u1))

        # Turn the unscaled absolute error into a relative one
        error = tree.ravel_pytree(error)[0]
        reference = tree.ravel_pytree(reference)[0]
        error_abs = dt * error
        error_norm = self.error_norm(error_abs, reference, atol=atol, rtol=rtol)

        # Scale the error norm with the error contraction rate and return
        error_contraction_rate = self.ssm.num_derivatives + 1
        error_power = error_norm ** (-1.0 / error_contraction_rate)
        return error_power, ()

    def _linearize_and_estimate(self, rv, /, *, linearized):
        observed = self.ssm.conditional.marginalise(rv, linearized)
        output_scale = self.ssm.stats.mahalanobis_norm_relative(0.0, rv=observed)
        stdev = self.ssm.stats.standard_deviation(observed)

        error_estimate_unscaled = np.squeeze(stdev)
        return output_scale * error_estimate_unscaled


class errorest_local_residual(ErrorEstimator):
    r"""Construct an error estimator based on a local residual.

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
        vector_field: Any,
        constraint: Constraint,
        prior: Any,
        ssm: Any,
        error_norm: Callable | None = None,
    ):
        if error_norm is None:
            error_norm = errorest_error_norm_scale_then_rms()

        self.error_norm = error_norm
        self.vector_field = vector_field
        self.constraint = constraint
        self.prior = prior
        self.ssm = ssm

    def init_errorest(self):
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
        output_scale = np.ones_like(self.ssm.prototypes.output_scale())
        transition = self.prior(dt, output_scale)

        # Estimate the error
        mean = self.ssm.stats.mean(previous.u.marginals)
        mean_extra = self.ssm.conditional.apply(mean, transition)
        error, state = self._linearize_and_estimate(
            mean_extra, state, t=proposed.t, damp=damp
        )

        # Compute a reference
        u0 = tree.tree_leaves(previous.u.mean)[0]
        u1 = tree.tree_leaves(proposed.u.mean)[0]
        reference = np.maximum(np.abs(u0), np.abs(u1))

        # Turn the unscaled absolute error into a relative one
        error = tree.ravel_pytree(error)[0]
        reference = tree.ravel_pytree(reference)[0]
        error_abs = dt * error
        error_norm = self.error_norm(error_abs, reference, atol=atol, rtol=rtol)

        # Scale the error norm with the error contraction rate and return
        error_contraction_rate = self.ssm.num_derivatives + 1
        error_power = error_norm ** (-1.0 / error_contraction_rate)
        return error_power, state

    def _linearize_and_estimate(self, rv, state, /, t, *, damp):
        f_wrapped = func.partial(self.vector_field, t=t)
        linearized, state = self.constraint.linearize(f_wrapped, rv, state, damp=damp)

        observed = self.ssm.conditional.marginalise(rv, linearized)
        output_scale = self.ssm.stats.mahalanobis_norm_relative(0.0, rv=observed)
        stdev = self.ssm.stats.standard_deviation(observed)

        error_estimate_unscaled = np.squeeze(stdev)
        error_estimate = output_scale * error_estimate_unscaled
        return error_estimate, state
