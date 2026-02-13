"""Probabilistic IVP solvers."""

from probdiffeq.backend import (
    containers,
    control_flow,
    functools,
    linalg,
    random,
    special,
    tree_array_util,
    tree_util,
)
from probdiffeq.backend import numpy as np
from probdiffeq.backend.typing import (
    Any,
    Array,
    ArrayLike,
    Callable,
    Generic,
    Sequence,
    TypeVar,
)
from probdiffeq.impl import impl
from probdiffeq.util import filter_util

C = TypeVar("C", bound=Sequence)
T = TypeVar("T")


# TODO: if we can't expose Normal() or Conditional() types
#   from the implementation,
#  maybe we can make these into protocols?


@tree_util.register_dataclass
@containers.dataclass
class MarkovSeq(Generic[T]):
    """Markov sequence."""

    init: T
    conditional: Any


@tree_util.register_dataclass
@containers.dataclass
class ProbTaylorCoeffs(Generic[C, T]):
    """A probabilistic description of Taylor coefficients.

    Includes means, standard deviations, and marginals.
    Common solution target for probabilistic solvers.
    """

    mean: C
    std: C
    marginals: T


@tree_util.register_dataclass
@containers.dataclass
class ProbDiffEqSol(Generic[C, T]):
    """The probabilistic numerical solution of an initial value problem (IVP)."""

    t: Array
    """The current time-step."""

    u: ProbTaylorCoeffs[C, T]
    """The current ODE solution estimate."""

    posterior: T | MarkovSeq[T]
    """The current posterior estimate."""

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


def prior_wiener_integrated(
    tcoeffs: C,
    *,
    tcoeffs_std: C | None = None,
    ssm_fact: str,
    output_scale: ArrayLike | None = None,
    damp: float = 0.0,
):
    """Construct an adaptive(/continuous-time), multiply-integrated Wiener process.

    Increase damping to get visually more pleasing uncertainties
     and more numerical robustness for
     high-order solvers in low precision arithmetic
    """
    ssm = impl.choose(ssm_fact, tcoeffs_like=tcoeffs)

    # TODO: should the output_scale be an argument to solve()?
    # TODO: should the damping mirror the pytree structure of 'tcoeffs'?
    if output_scale is None:
        output_scale = np.ones_like(ssm.prototypes.output_scale())

    discretize = ssm.conditional.ibm_transitions(base_scale=output_scale)

    if tcoeffs_std is None:
        error_like = np.ones_like(ssm.prototypes.error_estimate())
        tcoeffs_std = tree_util.tree_map(lambda _: error_like, tcoeffs)
    marginal = ssm.normal.from_tcoeffs(tcoeffs, tcoeffs_std, damp=damp)
    u_mean = ssm.stats.qoi(marginal)
    std = ssm.stats.standard_deviation(marginal)
    u_std = ssm.stats.qoi_from_sample(std)
    target = ProbTaylorCoeffs(u_mean, u_std, marginal)
    return target, discretize, ssm


def prior_wiener_integrated_discrete(ts, *args, **kwargs):
    """Compute a time-discretized, multiply-integrated Wiener process."""
    init, discretize, ssm = prior_wiener_integrated(*args, **kwargs)
    scales = np.ones_like(ssm.prototypes.output_scale())
    discretize_vmap = functools.vmap(discretize, in_axes=(0, None))
    conditionals = discretize_vmap(np.diff(ts), scales)
    return init, conditionals, ssm


# TODO: AdaState carries the same two fields. Combine?
@tree_util.register_dataclass
@containers.dataclass
class _InterpRes(Generic[T]):
    step_from: T
    """The new 'step_from' field.

    At time `max(t, s1.t)`.
    Use this as the right-most reference state
    in future interpolations, or continue time-stepping from here.
    """

    interp_from: T
    """The new `interp_from` field.

    At time `t`. Use this as the right-most reference state
    in future interpolations, or continue time-stepping from here.

    The difference between `interpolated` and `interp_from` emerges in save_at* modes.
    `interpolated` belongs to the just-concluded time interval,
    and `interp_from` belongs to the to-be-started time interval.
    Concretely, this means that `interp_from` has a unit backward model
    and `interpolated` remembers how to step back to the previous target location.
    """


@tree_util.register_dataclass
@containers.dataclass
class _PositiveCubatureRule:
    """Cubature rule with positive weights."""

    points: ArrayLike
    weights_sqrtm: ArrayLike


def cubature_third_order_spherical(input_shape) -> _PositiveCubatureRule:
    """Third-order spherical cubature integration."""
    assert len(input_shape) <= 1
    if len(input_shape) == 1:
        (d,) = input_shape
        points_mat, weights_sqrtm = _third_order_spherical_params(d=d)
        return _PositiveCubatureRule(points=points_mat, weights_sqrtm=weights_sqrtm)

    # If input_shape == (), compute weights via input_shape=(1,)
    # and 'squeeze' the points.
    points_mat, weights_sqrtm = _third_order_spherical_params(d=1)
    (S, _) = points_mat.shape
    points = np.reshape(points_mat, (S,))
    return _PositiveCubatureRule(points=points, weights_sqrtm=weights_sqrtm)


def _third_order_spherical_params(*, d):
    eye_d = np.eye(d) * np.sqrt(d)
    pts = np.concatenate((eye_d, -1 * eye_d))
    weights_sqrtm = np.ones((2 * d,)) / np.sqrt(2.0 * d)
    return pts, weights_sqrtm


def cubature_unscented_transform(input_shape, r=1.0) -> _PositiveCubatureRule:
    """Unscented transform."""
    assert len(input_shape) <= 1
    if len(input_shape) == 1:
        (d,) = input_shape
        points_mat, weights_sqrtm = _unscented_transform_params(d=d, r=r)
        return _PositiveCubatureRule(points=points_mat, weights_sqrtm=weights_sqrtm)

    # If input_shape == (), compute weights via input_shape=(1,)
    # and 'squeeze' the points.
    points_mat, weights_sqrtm = _unscented_transform_params(d=1, r=r)
    (S, _) = points_mat.shape
    points = np.reshape(points_mat, (S,))
    return _PositiveCubatureRule(points=points, weights_sqrtm=weights_sqrtm)


def _unscented_transform_params(d, *, r):
    eye_d = np.eye(d) * np.sqrt(d + r)
    zeros = np.zeros((1, d))
    pts = np.concatenate((eye_d, zeros, -1 * eye_d))
    _scale = d + r
    weights_sqrtm1 = np.ones((d,)) / np.sqrt(2.0 * _scale)
    weights_sqrtm2 = np.sqrt(r / _scale)
    weights_sqrtm = np.hstack((weights_sqrtm1, weights_sqrtm2, weights_sqrtm1))
    return pts, weights_sqrtm


def cubature_gauss_hermite(input_shape, degree=5) -> _PositiveCubatureRule:
    """(Statistician's) Gauss-Hermite cubature.

    The number of cubature points is `prod(input_shape)**degree`.
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
    return _PositiveCubatureRule(points=tensor_pts, weights_sqrtm=tensor_weights_sqrtm)


# TODO: how does this generalise to an input_shape instead of an input_dimension?
#  via tree_map(lambda s: _tensor_points(x, s), input_shape)?


def _tensor_weights(*args, **kwargs):
    mesh = _tensor_points(*args, **kwargs)
    return np.prod_along_axis(mesh, axis=1)


def _tensor_points(x, /, *, d):
    x_mesh = np.meshgrid(*([x] * d))
    y_mesh = tree_util.tree_map(lambda s: np.reshape(s, (-1,)), x_mesh)
    return np.stack(y_mesh).T


@containers.dataclass
class _Strategy(Generic[T]):
    ssm: Any

    is_suitable_for_save_at: int
    is_suitable_for_save_every_step: int
    is_suitable_for_offgrid_marginals: int

    def init_posterior(self, *, u) -> T:
        """Initialise a state from a solution."""
        raise NotImplementedError

    def predict(self, *, posterior: T, transition) -> tuple[ProbTaylorCoeffs, T]:
        """Extrapolate (also known as prediction)."""
        raise NotImplementedError

    def apply_updates(self, *, prediction: T, updates) -> tuple[ProbTaylorCoeffs, T]:
        """Apply a correction to the prediction."""
        raise NotImplementedError

    def interpolate(
        self, *, posterior_t0: T, posterior_t1: T, transition_t0_t, transition_t_t1
    ):
        """Interpolate."""
        raise NotImplementedError

    def interpolate_at_t1(self, *, posterior_t1: T):
        """Process the state at a checkpoint."""
        raise NotImplementedError

    def finalize(self, *, posterior0: T, posterior: T, output_scale):
        """Postprocess the posterior before returning."""
        raise NotImplementedError

    def markov_marginals(self, markov_seq, *, reverse):
        """Extract the (time-)marginals from a Markov sequence."""
        if markov_seq.init.mean.ndim == markov_seq.conditional.noise.mean.ndim:
            markov_seq = self._markov_select_terminal(markov_seq)

        def step(x, cond):
            extrapolated = self.ssm.conditional.marginalise(x, cond)
            return extrapolated, extrapolated

        init, xs = markov_seq.init, markov_seq.conditional
        _, marginals = control_flow.scan(step, init=init, xs=xs, reverse=reverse)

        if reverse:
            # Append the terminal marginal to the computed ones
            return tree_array_util.tree_append(marginals, init)

        return tree_array_util.tree_prepend(init, marginals)

    def markov_sample(self, key, markov_seq, *, reverse, shape=()):
        """Sample from a Markov sequence."""
        if markov_seq.init.mean.ndim == markov_seq.conditional.noise.mean.ndim:
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
        init = tree_util.tree_map(lambda x: x[-1, ...], markov_seq.init)
        return MarkovSeq(init, markov_seq.conditional)

    def _sample_shape(self, markov_seq):
        # The number of samples is one larger than the number of conditionals
        noise = markov_seq.conditional.noise
        n, *shape_single_sample = self.ssm.stats.hidden_shape(noise)
        return n + 1, *tuple(shape_single_sample)

    def _transform_unit_sample(self, markov_seq, base_sample, /, reverse):
        if base_sample.ndim > len(self._sample_shape(markov_seq)):
            transform = functools.partial(self._transform_unit_sample, reverse=reverse)
            transform_vmap = functools.vmap(transform, in_axes=(None, 0))
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
            base_sample_init, markov_seq.init
        )

        # Loop over backward models and the remaining base samples
        xs = (markov_seq.conditional, base_sample_body)
        _, samples = control_flow.scan(
            body_fun, init=init_sample, xs=xs, reverse=reverse
        )

        if reverse:
            samples = np.concatenate([samples, init_sample[None, ...]])
        else:
            samples = np.concatenate([init_sample[None, ...], samples])

        return functools.vmap(self.ssm.stats.qoi_from_sample)(samples)

    def log_marginal_likelihood_terminal_values(
        self, u, /, *, standard_deviation, posterior
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
        [u_leaves], u_structure = tree_util.tree_flatten(u)
        [std_leaves], std_structure = tree_util.tree_flatten(standard_deviation)

        if u_structure != std_structure:
            msg = (
                f"Observation-noise tree structure {std_structure} "
                f"does not match the observation structure {u_structure}. "
            )
            raise ValueError(msg)

        # Generate an observation-model for the QOI
        model = self.ssm.conditional.to_derivative(0, u, standard_deviation)
        rv = posterior.init if isinstance(posterior, MarkovSeq) else posterior

        data = np.zeros_like(u_leaves)  # 'u' is baked into the observation model
        observed, _conditional = self.ssm.conditional.revert(rv, model)
        return self.ssm.stats.logpdf(data, observed)

    def log_marginal_likelihood(self, u, /, *, standard_deviation, posterior):
        """Compute the log-marginal-likelihood of observations of the IVP solution.

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
        [u_leaves], u_structure = tree_util.tree_flatten(u)
        [std_leaves], std_structure = tree_util.tree_flatten(standard_deviation)

        if u_structure != std_structure:
            msg = (
                f"Observation-noise tree structure {std_structure} "
                f"does not match the observation structure {u_structure}. "
            )
            raise ValueError(msg)

        qoi_flat, _ = tree_util.ravel_pytree(self.ssm.prototypes.qoi())
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

        if not isinstance(posterior, MarkovSeq):
            msg1 = "Time-series marginal likelihoods "
            msg2 = "cannot be computed with a filtering solution."
            raise TypeError(msg1 + msg2)

        # Generate an observation-model for the QOI

        model_fun = functools.vmap(
            self.ssm.conditional.to_derivative, in_axes=(None, 0, 0)
        )
        models = model_fun(0, u, standard_deviation)

        # Select the terminal variable
        rv = tree_util.tree_map(lambda s: s[-1, ...], posterior.init)

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


@containers.dataclass
class strategy_smoother(_Strategy[MarkovSeq]):
    """Construct a smoother."""

    def __init__(self, ssm):
        super().__init__(
            ssm=ssm,
            is_suitable_for_save_at=False,
            is_suitable_for_save_every_step=True,
            is_suitable_for_offgrid_marginals=True,
        )

    def init_posterior(self, *, u: ProbTaylorCoeffs):
        cond = self.ssm.conditional.identity(self.ssm.num_derivatives + 1)
        posterior = MarkovSeq(init=u.marginals, conditional=cond)
        return u, posterior

    def predict(
        self, *, posterior: MarkovSeq, transition
    ) -> tuple[ProbTaylorCoeffs, MarkovSeq]:
        marginals, cond = self.ssm.conditional.revert(posterior.init, transition)
        posterior = MarkovSeq(init=marginals, conditional=cond)

        u = self.ssm.stats.qoi(marginals)
        std = self.ssm.stats.standard_deviation(marginals)
        u_std = self.ssm.stats.qoi_from_sample(std)
        estimate = ProbTaylorCoeffs(u, u_std, marginals)
        return estimate, posterior

    def apply_updates(self, prediction, *, updates):
        posterior = MarkovSeq(updates, prediction.conditional)
        marginals = updates
        u = self.ssm.stats.qoi(marginals)
        std = self.ssm.stats.standard_deviation(marginals)
        u_std = self.ssm.stats.qoi_from_sample(std)
        return ProbTaylorCoeffs(u, u_std, marginals), posterior

    def finalize(self, *, posterior0: MarkovSeq, posterior: MarkovSeq, output_scale):
        assert output_scale.shape == self.ssm.prototypes.output_scale().shape

        init0 = self.ssm.stats.rescale_cholesky(posterior0.init, output_scale)
        conditional0 = self.ssm.conditional.rescale_noise(
            posterior0.conditional, output_scale
        )
        posterior0 = MarkovSeq(init0, conditional0)

        # Calibrate
        init = self.ssm.stats.rescale_cholesky(posterior.init, output_scale)
        conditional = self.ssm.conditional.rescale_noise(
            posterior.conditional, output_scale
        )
        posterior = MarkovSeq(init, conditional)

        # Marginalise
        marginals = self.markov_marginals(posterior, reverse=True)

        # Prepend the initial condition to the filtering distributions
        init = tree_array_util.tree_prepend(posterior0.init, posterior.init)
        posterior = MarkovSeq(init=init, conditional=posterior.conditional)

        # Extract targets
        u = self.ssm.stats.qoi(marginals)
        std = self.ssm.stats.standard_deviation(marginals)
        u_std = self.ssm.stats.qoi_from_sample(std)
        estimate = ProbTaylorCoeffs(u, u_std, marginals)
        return estimate, posterior

    def interpolate(
        self,
        *,
        posterior_t0: MarkovSeq,
        posterior_t1: MarkovSeq,
        transition_t0_t,
        transition_t_t1,
    ):
        """Interpolate.

        A smoother interpolates by_
        * Extrapolating from t0 to t, which gives the "filtering" marginal
        and the backward transition from t to t0.
        * Extrapolating from t to t1, which gives another "filtering" marginal
        and the backward transition from t1 to t.
        * Applying the new t1-to-t backward transition to compute the interpolation.
        This intermediate result is informed about its "right-hand side" datum.

        Subsequent interpolations continue from the value at 't'.
        ( TODO: they could also continue from t0)
        Subsequent IVP solver steps continue from the value at 't1' as before.
        """
        # Extrapolate from t0 to t, and from t to t1.

        _, extrapolated_t = self.predict(
            posterior=posterior_t0, transition=transition_t0_t
        )
        _, extrapolated_t1 = self.predict(
            posterior=extrapolated_t, transition=transition_t_t1
        )

        # Marginalise backwards from t1 to t to obtain the interpolated solution.
        marginal_t1 = posterior_t1.init
        conditional_t1_to_t = extrapolated_t1.conditional
        rv_at_t = self.ssm.conditional.marginalise(marginal_t1, conditional_t1_to_t)
        solution_at_t = MarkovSeq(rv_at_t, extrapolated_t.conditional)

        # The state at t1 gets a new backward model;
        # (it must remember how to get back to t, not to t0).
        solution_at_t1 = MarkovSeq(marginal_t1, conditional_t1_to_t)
        interp_res = _InterpRes(step_from=solution_at_t1, interp_from=solution_at_t)

        # Extract targets
        marginals = solution_at_t.init
        u = self.ssm.stats.qoi(marginals)
        std = self.ssm.stats.standard_deviation(marginals)
        u_std = self.ssm.stats.qoi_from_sample(std)
        estimate = ProbTaylorCoeffs(u, u_std, marginals)
        return (estimate, solution_at_t), interp_res

    def interpolate_at_t1(self, posterior_t1):
        marginals = posterior_t1.init
        u = self.ssm.stats.qoi(marginals)
        std = self.ssm.stats.standard_deviation(marginals)
        u_std = self.ssm.stats.qoi_from_sample(std)
        estimate = ProbTaylorCoeffs(u, u_std, marginals)

        interp_res = _InterpRes(step_from=posterior_t1, interp_from=posterior_t1)
        return (estimate, posterior_t1), interp_res


@containers.dataclass
class strategy_filter(_Strategy):
    """Construct a filter."""

    def __init__(self, ssm):
        super().__init__(
            ssm=ssm,
            is_suitable_for_save_at=True,
            is_suitable_for_save_every_step=True,
            is_suitable_for_offgrid_marginals=True,
        )

    def init_posterior(self, *, u: ProbTaylorCoeffs):
        return u, u.marginals

    def predict(self, posterior: T, *, transition) -> tuple[ProbTaylorCoeffs, T]:
        marginals = self.ssm.conditional.marginalise(posterior, transition)
        u = self.ssm.stats.qoi(marginals)
        std = self.ssm.stats.standard_deviation(marginals)
        u_std = self.ssm.stats.qoi_from_sample(std)
        estimate = ProbTaylorCoeffs(u, u_std, marginals)
        return estimate, marginals

    def apply_updates(self, prediction, *, updates):
        del prediction
        marginals = updates
        u = self.ssm.stats.qoi(marginals)
        std = self.ssm.stats.standard_deviation(marginals)
        u_std = self.ssm.stats.qoi_from_sample(std)
        estimate = ProbTaylorCoeffs(u, u_std, marginals)
        return estimate, marginals

    def finalize(self, *, posterior0, posterior, output_scale):
        assert output_scale.shape == self.ssm.prototypes.output_scale().shape

        # No rescaling because no calibration at the initial step
        posterior0 = self.ssm.stats.rescale_cholesky(posterior0, output_scale)

        # Calibrate
        posterior = self.ssm.stats.rescale_cholesky(posterior, output_scale)

        # Stack
        posterior = tree_array_util.tree_prepend(posterior0, posterior)

        marginals = posterior
        u = self.ssm.stats.qoi(marginals)
        std = self.ssm.stats.standard_deviation(marginals)
        u_std = self.ssm.stats.qoi_from_sample(std)
        estimate = ProbTaylorCoeffs(u, u_std, marginals)
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
        estimate = ProbTaylorCoeffs(u, u_std, marginals)

        interp_res = _InterpRes(step_from=posterior_t1, interp_from=interpolated)
        return (estimate, interpolated), interp_res

    def interpolate_at_t1(self, *, posterior_t1):
        u = self.ssm.stats.qoi(posterior_t1)
        std = self.ssm.stats.standard_deviation(posterior_t1)
        u_std = self.ssm.stats.qoi_from_sample(std)
        marginals = posterior_t1
        estimate = ProbTaylorCoeffs(u, u_std, marginals)

        interp_res = _InterpRes(step_from=posterior_t1, interp_from=posterior_t1)
        return (estimate, posterior_t1), interp_res


@containers.dataclass
class strategy_fixedpoint(_Strategy[MarkovSeq]):
    """Construct a fixedpoint-smoother."""

    def __init__(self, ssm):
        super().__init__(
            ssm=ssm,
            is_suitable_for_save_at=True,
            is_suitable_for_save_every_step=False,
            is_suitable_for_offgrid_marginals=False,
        )

    def init_posterior(self, *, u):
        cond = self.ssm.conditional.identity(self.ssm.num_derivatives + 1)
        posterior = MarkovSeq(u.marginals, cond)
        return u, posterior

    def predict(
        self, posterior: MarkovSeq, *, transition
    ) -> tuple[ProbTaylorCoeffs, MarkovSeq]:
        rv = posterior.init
        bw0 = posterior.conditional
        marginals, cond = self.ssm.conditional.revert(rv, transition)
        cond = self.ssm.conditional.merge(bw0, cond)
        predicted = MarkovSeq(marginals, cond)

        u = self.ssm.stats.qoi(marginals)
        std = self.ssm.stats.standard_deviation(marginals)
        u_std = self.ssm.stats.qoi_from_sample(std)
        estimate = ProbTaylorCoeffs(u, u_std, marginals)
        return estimate, predicted

    def apply_updates(self, prediction: MarkovSeq, *, updates):
        posterior = MarkovSeq(updates, prediction.conditional)
        rv = updates
        u = self.ssm.stats.qoi(rv)
        std = self.ssm.stats.standard_deviation(rv)
        u_std = self.ssm.stats.qoi_from_sample(std)
        marginals = rv
        estimate = ProbTaylorCoeffs(u, u_std, marginals)
        return estimate, posterior

    def finalize(self, *, posterior0: MarkovSeq, posterior: MarkovSeq, output_scale):
        assert output_scale.shape == self.ssm.prototypes.output_scale().shape
        init = self.ssm.stats.rescale_cholesky(posterior0.init, output_scale)
        conditional = self.ssm.conditional.rescale_noise(
            posterior0.conditional, output_scale
        )
        posterior0 = MarkovSeq(init, conditional)

        # Calibrate the time series
        init = self.ssm.stats.rescale_cholesky(posterior.init, output_scale)
        conditional = self.ssm.conditional.rescale_noise(
            posterior.conditional, output_scale
        )
        posterior = MarkovSeq(init, conditional)

        # Marginalise
        marginals = self.markov_marginals(posterior, reverse=True)

        # Prepend the initial condition to the filtering distributions
        init = tree_array_util.tree_prepend(posterior0.init, posterior.init)
        posterior = MarkovSeq(init=init, conditional=posterior.conditional)

        # Extract targets
        u = self.ssm.stats.qoi(marginals)
        std = self.ssm.stats.standard_deviation(marginals)
        u_std = self.ssm.stats.qoi_from_sample(std)
        estimate = ProbTaylorCoeffs(u, u_std, marginals)
        return estimate, posterior

    def interpolate_at_t1(self, *, posterior_t1: MarkovSeq):
        cond_identity = self.ssm.conditional.identity(self.ssm.num_derivatives + 1)
        resume_from = MarkovSeq(posterior_t1.init, conditional=cond_identity)
        interp_res = _InterpRes(step_from=resume_from, interp_from=resume_from)

        interpolated = posterior_t1
        marginals = interpolated.init
        u = self.ssm.stats.qoi(marginals)
        std = self.ssm.stats.standard_deviation(marginals)
        u_std = self.ssm.stats.qoi_from_sample(std)
        estimate = ProbTaylorCoeffs(u, u_std, marginals)
        return (estimate, interpolated), interp_res

    def interpolate(
        self,
        *,
        posterior_t0: MarkovSeq,
        posterior_t1: MarkovSeq,
        transition_t0_t,
        transition_t_t1,
    ):
        """
        Interpolate using a fixed-point smoother.

        Assuming `state_t0` has seen 'n' collocation points,
        and `state_t1` has seen 'n+1' collocation points,
        then interpolation at time `t` is computed as follows:

        1. Extrapolate from `t0` to `t`. This yields:
            - the marginal at `t` given `n` observations.
            - the backward transition from `t` to `t0` given `n` observations.

        2. Extrapolate from `t` to `t1`. This yields:
            - the marginal at `t1` given `n` observations
              (in contrast, `state_t1` has seen `n+1` observations)
            - the backward transition from `t1` to `t` given `n` observations.

        3. Apply the backward transition from `t1` to `t`
        to the marginal inside `state_t1`
        to obtain the marginal at `t` given `n+1` observations. Similarly,
        the interpolated solution inherits all auxiliary info from the `t_1` state.

        ---------------------------------------------------------------------

        Difference to standard smoother interpolation:

        In the fixed-point smoother, backward transitions are modified
        to ensure that future operations remain correct.
        Denote the location of the fixed-point with `t_f`. Then,
        the backward transition at `t` is merged with that at `t0`.
        This preserves knowledge of how to move from `t` to `t_f`.

        Then, `t` becomes the new fixed-point location. To ensure
        that future operations ``find their way back to t`:

        - Subsequent interpolations do not continue from the raw
        interpolated value. Instead, they continue from a nearly
        identical state where the backward transition is replaced
        by the identity.

        - Subsequent solver steps do not continue from the initial `t1`
        state. Instead, they continue from a version whose backward
        model is replaced with the `t-to-t1` transition.


        ---------------------------------------------------------------------

        As a result, each interpolation must return three distinct states:

            1. the interpolated solution,
            2. the state to continue interpolating from,
            3. the state to continue solver stepping from.

        These are intentionally different in the fixed-point smoother.
        Don't attempt to remove any of them. They're all important.
        """
        # Extrapolate from t0 to t, and from t to t1.
        # This yields all building blocks.
        _, extrapolated_t = self.predict(
            posterior=posterior_t0, transition=transition_t0_t
        )
        conditional_id = self.ssm.conditional.identity(self.ssm.num_derivatives + 1)
        previous_new = MarkovSeq(extrapolated_t.init, conditional_id)
        _, extrapolated_t1 = self.predict(
            posterior=previous_new, transition=transition_t_t1
        )

        # Marginalise from t1 to t to obtain the interpolated solution.
        marginal_t1 = posterior_t1.init
        conditional_t1_to_t = extrapolated_t1.conditional
        rv_at_t = self.ssm.conditional.marginalise(marginal_t1, conditional_t1_to_t)

        # Return the right combination of marginals and conditionals.
        interpolated = MarkovSeq(rv_at_t, extrapolated_t.conditional)
        step_from = MarkovSeq(posterior_t1.init, conditional=conditional_t1_to_t)
        interp_res = _InterpRes(step_from=step_from, interp_from=previous_new)

        marginals = interpolated.init
        u = self.ssm.stats.qoi(marginals)
        std = self.ssm.stats.standard_deviation(marginals)
        u_std = self.ssm.stats.qoi_from_sample(std)
        estimate = ProbTaylorCoeffs(u, u_std, marginals)
        return (estimate, interpolated), interp_res


@containers.dataclass
class _Information:
    """Information model interface."""

    linearization: Any

    def init_linearization(self, /):
        """Initialise the state from the solution."""
        return self.linearization.init()

    def linearize(self, vf_wrapped, rv, state):
        """Perform the correction step."""
        return self.linearization.apply(vf_wrapped, rv, state)


def constraint_ode_ts0(ssm, ode_order=1) -> _Information:
    """ODE constraint with zeroth-order Taylor linearisation."""
    return ssm.linearise.ode_taylor_0th(ode_order=ode_order)


# TODO: expose a "jacobian" option to choose between fwd and rev mode
def constraint_ode_ts1(
    *, ssm, ode_order=1, jvp_probes=10, jvp_probes_seed=1
) -> _Information:
    """ODE constraint with first-order Taylor linearisation."""
    assert jvp_probes > 0
    return ssm.linearise.ode_taylor_1st(
        ode_order=ode_order, jvp_probes=jvp_probes, jvp_probes_seed=jvp_probes_seed
    )


def constraint_ode_slr0(
    *, ssm, cubature_fun=cubature_third_order_spherical
) -> _Information:
    """ODE constraint with zeroth-order statistical linear regression."""
    return ssm.linearise.ode_statistical_0th(cubature_fun)


def constraint_ode_slr1(
    *, ssm, cubature_fun=cubature_third_order_spherical
) -> _Information:
    """ODE constraint with first-order statistical linear regression."""
    return ssm.linearise.ode_statistical_1st(cubature_fun)


@containers.dataclass
class _ProbabilisticSolver:
    vector_field: Callable
    """The ODE vector field."""

    strategy: _Strategy
    """The estimation strategy. 
    
    Usually filter vs fixedpoint (vs smoother).
    """

    prior: Callable
    """The prior. 
    
    Usually an integrated Wiener process.
    """

    ssm: Any
    """The state-space model implementation. 
    
    Constructed together with the prior.
    """

    constraint: Any
    """The constraint + correction model."""

    @property
    def error_contraction_rate(self):
        return self.ssm.num_derivatives + 1

    @property
    def is_suitable_for_offgrid_marginals(self):
        return self.strategy.is_suitable_for_offgrid_marginals

    @property
    def is_suitable_for_save_at(self):
        return self.strategy.is_suitable_for_save_at

    @property
    def is_suitable_for_save_every_step(self):
        return self.strategy.is_suitable_for_save_every_step

    def init(self, t, init) -> ProbDiffEqSol:
        raise NotImplementedError

    def step(self, state: ProbDiffEqSol, *, dt):
        raise NotImplementedError

    def userfriendly_output(
        self, *, solution0: ProbDiffEqSol, solution: ProbDiffEqSol
    ) -> ProbDiffEqSol:
        """Make the solutions user-friendly.

        This means calibrating, precomputing marginals, etc..
        """
        raise NotImplementedError

    def offgrid_marginals(self, t, *, solution):
        """Compute off-grid marginals on a dense grid via jax.numpy.searchsorted.

        !!! warning
            The elements in ts and the elements in the solution grid must be disjoint.
            Otherwise, anything can happen and the solution will be incorrect.
            At the moment, we do not check this.

        !!! warning
            The elements in ts must be strictly in (t0, t1).
            They must not lie outside the interval, and they must not coincide
            with the interval boundaries.
            At the moment, we do not check this.
        """
        assert t.shape == solution.t[0].shape
        # side="left" and side="right" are equivalent
        # because we _assume_ that the point sets are disjoint.
        index = np.searchsorted(solution.t, t)

        # Extract the LHS

        def _extract_previous(tree):
            return tree_util.tree_map(lambda s: s[index - 1, ...], tree)

        posterior_t0 = _extract_previous(solution.posterior)
        t0 = _extract_previous(solution.t)

        # Extract the RHS

        def _extract(tree):
            return tree_util.tree_map(lambda s: s[index, ...], tree)

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

    def interpolate(self, *, t, interp_from: ProbDiffEqSol, interp_to: ProbDiffEqSol):
        # Domain is (t0, t1]; thus, take the output scale from interp_to
        output_scale = interp_to.output_scale
        transition_t0_t = self.prior(t - interp_from.t, output_scale)
        transition_t_t1 = self.prior(interp_to.t - t, output_scale)

        # Interpolate
        tmp = self.strategy.interpolate(
            posterior_t0=interp_from.posterior,
            posterior_t1=interp_to.posterior,
            transition_t0_t=transition_t0_t,
            transition_t_t1=transition_t_t1,
        )
        (estimate, interpolated), step_and_interpolate_from = tmp

        step_from = ProbDiffEqSol(
            t=interp_to.t,
            # New:
            posterior=step_and_interpolate_from.step_from,
            # Old:
            u=interp_to.u,
            output_scale=interp_to.output_scale,
            auxiliary=interp_to.auxiliary,
            num_steps=interp_to.num_steps,
            fun_evals=interp_to.fun_evals,
        )

        interpolated = ProbDiffEqSol(
            t=t,
            # New:
            posterior=interpolated,
            u=estimate,
            # Taken from the rhs point
            output_scale=interp_to.output_scale,
            auxiliary=interp_to.auxiliary,
            num_steps=interp_to.num_steps,
            fun_evals=interp_to.fun_evals,
        )

        interp_from = ProbDiffEqSol(
            t=t,
            # New:
            posterior=step_and_interpolate_from.interp_from,
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
        self, *, t, interp_from: ProbDiffEqSol, interp_to: ProbDiffEqSol
    ):
        """Process the solution in case t=t_n."""
        del t
        tmp = self.strategy.interpolate_at_t1(posterior_t1=interp_to.posterior)
        (estimate, interpolated), step_and_interpolate_from = tmp

        prev = ProbDiffEqSol(
            t=interp_to.t,
            # New
            posterior=step_and_interpolate_from.interp_from,
            # Old
            u=interp_from.u,  # incorrect?
            output_scale=interp_from.output_scale,  # incorrect?
            auxiliary=interp_from.auxiliary,  # incorrect?
            num_steps=interp_from.num_steps,  # incorrect?
            fun_evals=interp_from.fun_evals,
        )
        sol = ProbDiffEqSol(
            t=interp_to.t,
            # New:
            posterior=interpolated,
            u=estimate,
            # Old:
            output_scale=interp_to.output_scale,
            auxiliary=interp_to.auxiliary,
            num_steps=interp_to.num_steps,
            fun_evals=interp_to.fun_evals,
        )
        acc = ProbDiffEqSol(
            t=interp_to.t,
            # New:
            posterior=step_and_interpolate_from.step_from,
            # Old
            u=interp_to.u,
            output_scale=interp_to.output_scale,
            auxiliary=interp_to.auxiliary,
            num_steps=interp_to.num_steps,
            fun_evals=interp_to.fun_evals,
        )
        return sol, _InterpRes(step_from=acc, interp_from=prev)


class solver_mle(_ProbabilisticSolver):
    """Create a solver that calibrates the output scale via maximum-likelihood.

    Warning: needs to be combined with a call to stats.calibrate()
    after solving if the MLE-calibration shall be *used*.
    """

    def init(self, t, u) -> ProbDiffEqSol:
        estimate, posterior = self.strategy.init_posterior(u=u)
        correction_state = self.constraint.init_linearization()

        output_scale_prior = np.ones_like(self.ssm.prototypes.output_scale())
        output_scale_running = 0 * output_scale_prior

        f_wrapped = functools.partial(self.vector_field, t=t)
        fun_evals, correction_state = self.constraint.linearize(
            f_wrapped, estimate.marginals, correction_state, damp=0.0
        )
        fun_evals = tree_util.tree_map(np.zeros_like, fun_evals)
        auxiliary = (correction_state, output_scale_running, 0)
        return ProbDiffEqSol(
            t=t,
            u=estimate,
            posterior=posterior,
            auxiliary=auxiliary,
            output_scale=output_scale_prior,
            num_steps=0,
            fun_evals=fun_evals,
        )

    def step(self, state, *, dt: float, damp: float):
        # Discretize
        output_scale = np.ones_like(self.ssm.prototypes.output_scale())
        transition = self.prior(dt, output_scale)

        # Predict
        u, prediction = self.strategy.predict(
            posterior=state.posterior, transition=transition
        )

        # Linearize
        (lin_state, output_scale_running, num_data) = state.auxiliary
        f_wrapped = functools.partial(self.vector_field, t=state.t + dt)
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
        return ProbDiffEqSol(
            t=state.t + dt,
            u=u,
            posterior=posterior,
            output_scale=state.output_scale,
            auxiliary=auxiliary,
            num_steps=state.num_steps + 1,
            fun_evals=fx,
        )

    def userfriendly_output(
        self, *, solution0: ProbDiffEqSol, solution: ProbDiffEqSol
    ) -> ProbDiffEqSol:
        assert solution.t.ndim > 0

        # This is the MLE solver, so we take the calibrated scale
        _, output_scale, _ = solution.auxiliary
        ones = np.ones_like(output_scale)
        output_scale = output_scale[-1]

        init = solution0.posterior
        posterior = solution.posterior
        estimate, posterior = self.strategy.finalize(
            posterior0=init, posterior=posterior, output_scale=output_scale
        )

        output_scale = ones * output_scale[None, ...]
        ts = np.concatenate([solution0.t[None], solution.t])
        return ProbDiffEqSol(
            t=ts,
            u=estimate,
            posterior=posterior,
            output_scale=output_scale,
            num_steps=solution.num_steps,
            auxiliary=solution.auxiliary,
            fun_evals=solution.fun_evals,
        )


class solver_dynamic(_ProbabilisticSolver):
    """Create a solver that calibrates the output scale dynamically."""

    def init(self, t, u) -> ProbDiffEqSol:
        estimate, posterior = self.strategy.init_posterior(u=u)
        lin_state = self.constraint.init_linearization()

        output_scale = np.ones_like(self.ssm.prototypes.output_scale())

        f_wrapped = functools.partial(self.vector_field, t=t)
        fx, lin_state = self.constraint.linearize(
            f_wrapped, estimate.marginals, lin_state, damp=0.0
        )
        fx = tree_util.tree_map(np.zeros_like, fx)

        return ProbDiffEqSol(
            t=t,
            u=estimate,
            posterior=posterior,
            auxiliary=lin_state,
            output_scale=output_scale,
            num_steps=0,
            fun_evals=fx,
        )

    def step(self, state: ProbDiffEqSol, *, dt: float, damp: float):
        lin_state = state.auxiliary

        # Calibrate the output scale
        ones = np.ones_like(self.ssm.prototypes.output_scale())
        transition = self.prior(dt, ones)
        mean = self.ssm.stats.mean(state.u.marginals)
        u = self.ssm.conditional.apply(mean, transition)

        t = state.t + dt

        # Linearize
        f_wrapped = functools.partial(self.vector_field, t=t)
        fx0, lin_state = self.constraint.linearize(
            f_wrapped, u, state=lin_state, damp=damp
        )
        observed = self.ssm.conditional.marginalise(u, fx0)
        output_scale = self.ssm.stats.mahalanobis_norm_relative(0.0, rv=observed)

        # Do the full extrapolation with the calibrated output scale
        # (Includes re-discretisation)
        transition = self.prior(dt, output_scale)
        u, prediction = self.strategy.predict(state.posterior, transition=transition)

        # Relinearise (TODO: optional? Skip entirely?)
        fx, lin_state = self.constraint.linearize(
            f_wrapped, u.marginals, state=lin_state, damp=damp
        )

        # Complete the update
        _, reverted = self.ssm.conditional.revert(u.marginals, fx)
        updates = reverted.noise
        u, posterior = self.strategy.apply_updates(prediction, updates=updates)

        # Return solution
        return ProbDiffEqSol(
            t=t,
            u=u,
            posterior=posterior,
            num_steps=state.num_steps + 1,
            auxiliary=lin_state,
            output_scale=output_scale,
            fun_evals=fx0,  # return the initial linearization
        )

    def userfriendly_output(self, *, solution: ProbDiffEqSol, solution0: ProbDiffEqSol):
        # This is the dynamic solver,
        # and all covariances have been calibrated already
        ones = np.ones_like(solution.output_scale)
        output_scale = ones[-1, ...]

        init = solution0.posterior
        posterior = solution.posterior
        estimate, posterior = self.strategy.finalize(
            posterior0=init, posterior=posterior, output_scale=output_scale
        )

        # TODO: stack the calibrated output scales?
        output_scale = ones
        ts = np.concatenate([solution0.t[None], solution.t])
        return ProbDiffEqSol(
            t=ts,
            u=estimate,
            posterior=posterior,
            output_scale=output_scale,
            num_steps=solution.num_steps,
            auxiliary=solution.auxiliary,
            fun_evals=solution.fun_evals,
        )


class solver(_ProbabilisticSolver):
    """Create a solver that does not calibrate the output scale automatically."""

    def init(self, t: ArrayLike, u: ProbTaylorCoeffs) -> ProbDiffEqSol:
        u, posterior = self.strategy.init_posterior(u=u)
        correction_state = self.constraint.init_linearization()
        output_scale = np.ones_like(self.ssm.prototypes.output_scale())

        f_wrapped = functools.partial(self.vector_field, t=t)
        fun_evals, correction_state = self.constraint.linearize(
            f_wrapped, rv=u.marginals, state=correction_state, damp=0.0
        )
        fun_evals = tree_util.tree_map(np.zeros_like, fun_evals)
        return ProbDiffEqSol(
            t=t,
            u=u,
            posterior=posterior,
            num_steps=0,
            auxiliary=correction_state,
            output_scale=output_scale,
            fun_evals=fun_evals,
        )

    def step(self, state: ProbDiffEqSol, *, dt, damp):
        # Discretize
        output_scale = np.ones_like(state.output_scale)
        transition = self.prior(dt, output_scale)

        # Predict
        u, prediction = self.strategy.predict(state.posterior, transition=transition)

        # Linearize
        f_eval = functools.partial(self.vector_field, t=state.t + dt)
        fx, auxiliary = self.constraint.linearize(
            f_eval, u.marginals, state.auxiliary, damp=damp
        )

        # Update
        _, reverted = self.ssm.conditional.revert(u.marginals, fx)
        u, posterior = self.strategy.apply_updates(prediction, updates=reverted.noise)

        # Return solution
        return ProbDiffEqSol(
            t=state.t + dt,
            u=u,
            posterior=posterior,
            output_scale=output_scale,
            auxiliary=auxiliary,
            num_steps=state.num_steps + 1,
            fun_evals=fx,
        )

    def userfriendly_output(
        self, *, solution0: ProbDiffEqSol, solution: ProbDiffEqSol
    ) -> ProbDiffEqSol:
        assert solution.t.ndim > 0

        # This is the uncalibrated solver, so scale=1
        ones = np.ones_like(solution.output_scale)
        output_scale = np.ones_like(solution.output_scale[-1])

        init = solution0.posterior
        posterior = solution.posterior
        u, posterior = self.strategy.finalize(
            posterior0=init, posterior=posterior, output_scale=output_scale
        )

        output_scale = ones * output_scale[None, ...]

        ts = np.concatenate([solution0.t[None], solution.t])
        return ProbDiffEqSol(
            t=ts,
            u=u,
            posterior=posterior,
            output_scale=output_scale,
            num_steps=solution.num_steps,
            auxiliary=solution.auxiliary,
            fun_evals=solution.fun_evals,
        )


@containers.dataclass
class _ErrorEstimator:
    pass


@containers.dataclass
class errorest_local_residual_cached(_ErrorEstimator):
    # Same as errorest_local_residual, but no additional
    # vector field evaluations.
    prior: Any
    ssm: Any
    norm_order: Any = None

    def init_errorest(self) -> tuple:
        return ()

    def estimate_error_norm(
        self,
        state: tuple,
        previous: ProbDiffEqSol,
        proposed: ProbDiffEqSol,
        *,
        dt: float,
        atol: float,
        rtol: float,
        damp: float,
    ) -> tuple[float, tuple]:
        del damp  # unused because no additional linearisation

        # Discretize; The output scale is set to one
        # since the error is multiplied with a local scale estimate anyway
        output_scale = np.ones_like(self.ssm.prototypes.output_scale())
        transition = self.prior(dt, output_scale)

        # Estimate the error
        mean = self.ssm.stats.mean(previous.u.marginals)
        mean_extra = self.ssm.conditional.apply(mean, transition)
        error = self._linearize_and_estimate(
            mean_extra, t=proposed.t, linearized=proposed.fun_evals
        )

        # Compute a reference
        u0 = tree_util.tree_leaves(previous.u.mean)[0]
        u1 = tree_util.tree_leaves(proposed.u.mean)[0]
        reference = np.maximum(np.abs(u0), np.abs(u1))

        # Turn the unscaled absolute error into a relative one
        error = tree_util.ravel_pytree(error)[0]
        reference = tree_util.ravel_pytree(reference)[0]
        error_abs = dt * error
        normalize = atol + rtol * np.abs(reference)
        error_rel = error_abs / normalize

        # Compute the of the error

        def rms(s):
            return linalg.vector_norm(s, order=self.norm_order) / np.sqrt(s.size)

        error_norm = rms(error_rel)

        # Scale the error norm with the error contraction rate and return
        error_contraction_rate = self.ssm.num_derivatives + 1
        error_power = error_norm ** (-1.0 / error_contraction_rate)
        return error_power, ()

    def _linearize_and_estimate(self, rv, /, t, *, linearized):
        observed = self.ssm.conditional.marginalise(rv, linearized)
        output_scale = self.ssm.stats.mahalanobis_norm_relative(0.0, rv=observed)
        stdev = self.ssm.stats.standard_deviation(observed)

        error_estimate_unscaled = np.squeeze(stdev)
        error_estimate = output_scale * error_estimate_unscaled
        return error_estimate


@containers.dataclass
class errorest_local_residual(_ErrorEstimator):
    vector_field: Any
    constraint: Any
    prior: Any
    ssm: Any
    norm_order: Any = None

    def init_errorest(self):
        return self.constraint.init_linearization()

    def estimate_error_norm(
        self,
        state,
        previous: ProbDiffEqSol,
        proposed: ProbDiffEqSol,
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
        u0 = tree_util.tree_leaves(previous.u.mean)[0]
        u1 = tree_util.tree_leaves(proposed.u.mean)[0]
        reference = np.maximum(np.abs(u0), np.abs(u1))

        # Turn the unscaled absolute error into a relative one
        error = tree_util.ravel_pytree(error)[0]
        reference = tree_util.ravel_pytree(reference)[0]
        error_abs = dt * error
        normalize = atol + rtol * np.abs(reference)
        error_rel = error_abs / normalize

        # Compute the of the error

        def rms(s):
            return linalg.vector_norm(s, order=self.norm_order) / np.sqrt(s.size)

        error_norm = rms(error_rel)

        # Scale the error norm with the error contraction rate and return
        error_contraction_rate = self.ssm.num_derivatives + 1
        error_power = error_norm ** (-1.0 / error_contraction_rate)
        return error_power, state

    def _linearize_and_estimate(self, rv, state, /, t, *, damp):
        f_wrapped = functools.partial(self.vector_field, t=t)
        linearized, state = self.constraint.linearize(f_wrapped, rv, state, damp=damp)

        observed = self.ssm.conditional.marginalise(rv, linearized)
        output_scale = self.ssm.stats.mahalanobis_norm_relative(0.0, rv=observed)
        stdev = self.ssm.stats.standard_deviation(observed)

        error_estimate_unscaled = np.squeeze(stdev)
        error_estimate = output_scale * error_estimate_unscaled
        return error_estimate, state
