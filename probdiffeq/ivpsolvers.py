"""Probabilistic IVP solvers."""

from probdiffeq import stats
from probdiffeq.backend import (
    containers,
    control_flow,
    functools,
    linalg,
    special,
    tree_array_util,
    tree_util,
)
from probdiffeq.backend import numpy as np
from probdiffeq.backend.typing import (
    Any,
    ArrayLike,
    Callable,
    Generic,
    NamedArg,
    Protocol,
    Sequence,
    TypeVar,
)
from probdiffeq.impl import impl

R = TypeVar("R")
C = TypeVar("C", bound=Sequence)
T = TypeVar("T")


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
    # TODO: should the output scale (and all 'damp'-like factors)
    #       mirror the pytree structure of 'tcoeffs'?
    if output_scale is None:
        output_scale = np.ones_like(ssm.prototypes.output_scale())

    discretize = ssm.conditional.ibm_transitions(base_scale=output_scale)

    if tcoeffs_std is None:
        error_like = np.ones_like(ssm.prototypes.error_estimate())
        tcoeffs_std = tree_util.tree_map(lambda _: error_like, tcoeffs)
    init = ssm.normal.from_tcoeffs(tcoeffs, tcoeffs_std, damp=damp)
    return init, discretize, ssm


def prior_wiener_integrated_discrete(ts, *args, **kwargs):
    """Compute a time-discretized, multiply-integrated Wiener process."""
    init, discretize, ssm = prior_wiener_integrated(*args, **kwargs)
    scales = np.ones_like(ssm.prototypes.output_scale())
    discretize_vmap = functools.vmap(discretize, in_axes=(0, None))
    conditionals = discretize_vmap(np.diff(ts), scales)
    return init, conditionals, ssm


@containers.dataclass
class _InterpRes(Generic[R]):
    step_from: R
    """The new 'step_from' field.

    At time `max(t, s1.t)`.
    Use this as the right-most reference state
    in future interpolations, or continue time-stepping from here.
    """

    interp_from: R
    """The new `interp_from` field.

    At time `t`. Use this as the right-most reference state
    in future interpolations, or continue time-stepping from here.

    The difference between `interpolated` and `interp_from` emerges in save_at* modes.
    `interpolated` belongs to the just-concluded time interval,
    and `interp_from` belongs to the to-be-started time interval.
    Concretely, this means that `interp_from` has a unit backward model
    and `interpolated` remembers how to step back to the previous target location.
    """


class _PositiveCubatureRule(containers.NamedTuple):
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


@tree_util.register_dataclass
@containers.dataclass
class Estimate:
    """Targets, standard deviations, and marginals."""

    u: Any
    u_std: Any
    marginals: Any


@containers.dataclass
class _Strategy:
    """Estimation-strategy interface."""

    ssm: Any

    is_suitable_for_save_at: int
    is_suitable_for_save_every_step: int
    is_suitable_for_offgrid_marginals: int

    def init_posterior(self, sol: Any, /) -> T:
        """Initialise a state from a solution."""
        raise NotImplementedError

    def predict(self, posterior: T, /, *, transition) -> tuple[Estimate, T]:
        """Extrapolate (also known as prediction)."""
        raise NotImplementedError

    def apply_updates(self, prediction: T, *, updates) -> tuple[Estimate, T]:
        raise NotImplementedError

    def interpolate(self, state_t0, state_t1, *, dt0, dt1, output_scale, prior):
        """Interpolate."""
        raise NotImplementedError

    def interpolate_at_t1(self, state_t0, state_t1, *, dt0, dt1, output_scale, prior):
        """Process the state at a checkpoint."""
        raise NotImplementedError


def strategy_smoother(*, ssm) -> _Strategy:
    """Construct a smoother."""

    @containers.dataclass
    class Smoother(_Strategy):
        def init_posterior(self, marginals):
            cond = self.ssm.conditional.identity(ssm.num_derivatives + 1)
            posterior = stats.MarkovSeq(init=marginals, conditional=cond)

            u = self.ssm.stats.qoi(marginals)
            std = self.ssm.stats.standard_deviation(marginals)
            u_std = self.ssm.stats.qoi_from_sample(std)
            estimate = Estimate(u, u_std, marginals)
            return estimate, posterior
            # # # Special case for implementing offgrid-marginals...
            # # if isinstance(sol, stats.MarkovSeq):
            # #     rv = sol.init
            # #     cond = sol.conditional
            # # else:
            # # rv = sol
            # # cond = self.ssm.conditional.identity(ssm.num_derivatives + 1)
            # return rv, cond

        def predict(self, posterior, *, transition):
            marginals, cond = self.ssm.conditional.revert(posterior.init, transition)
            posterior = stats.MarkovSeq(init=marginals, conditional=cond)

            u = self.ssm.stats.qoi(marginals)
            std = self.ssm.stats.standard_deviation(marginals)
            u_std = self.ssm.stats.qoi_from_sample(std)
            estimate = Estimate(u, u_std, marginals)
            return estimate, posterior

        def apply_updates(self, prediction, *, updates):
            posterior = stats.MarkovSeq(updates, prediction.conditional)
            marginals = updates
            u = self.ssm.stats.qoi(marginals)
            std = self.ssm.stats.standard_deviation(marginals)
            u_std = self.ssm.stats.qoi_from_sample(std)
            return Estimate(u, u_std, marginals), posterior

        def finalize(
            self, *, posterior0: stats.MarkovSeq, posterior: stats.MarkovSeq, scale
        ):
            # Calibrate
            init = ssm.stats.rescale_cholesky(posterior0.init, scale[-1, ...])
            conditional = ssm.conditional.rescale_noise(posterior0.conditional, scale)
            posterior0 = stats.MarkovSeq(init, conditional)
            init = ssm.stats.rescale_cholesky(posterior.init, scale)
            conditional = ssm.conditional.rescale_noise(posterior.conditional, scale)
            posterior = stats.MarkovSeq(init, conditional)

            # Marginalise
            posterior_no_filter_marginals = stats.markov_select_terminal(posterior)
            marginals = stats.markov_marginals(
                posterior_no_filter_marginals, reverse=True, ssm=self.ssm
            )

            # Append the terminal marginal to the computed ones
            marginal_t1 = tree_util.tree_map(lambda s: s[-1, ...], posterior.init)
            marginals = tree_array_util.tree_append(marginals, marginal_t1)

            # Prepend the initial condition to the filtering distributions
            init = tree_array_util.tree_prepend(posterior0.init, posterior.init)
            posterior = stats.MarkovSeq(init=init, conditional=posterior.conditional)

            # Extract targets
            u = ssm.stats.qoi(marginals)
            std = ssm.stats.standard_deviation(marginals)
            u_std = ssm.stats.qoi_from_sample(std)
            estimate = Estimate(u, u_std, marginals)
            return estimate, posterior

        def interpolate(
            self,
            *,
            state_t0: stats.MarkovSeq,
            state_t1: stats.MarkovSeq,
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

            _, extrapolated_t = self.predict(state_t0, transition=transition_t0_t)
            _, extrapolated_t1 = self.predict(
                extrapolated_t, transition=transition_t_t1
            )

            # Marginalise backwards from t1 to t to obtain the interpolated solution.
            marginal_t1 = state_t1.init
            conditional_t1_to_t = extrapolated_t1.conditional
            rv_at_t = self.ssm.conditional.marginalise(marginal_t1, conditional_t1_to_t)
            solution_at_t = stats.MarkovSeq(rv_at_t, extrapolated_t.conditional)

            # The state at t1 gets a new backward model;
            # (it must remember how to get back to t, not to t0).
            solution_at_t1 = stats.MarkovSeq(marginal_t1, conditional_t1_to_t)
            interp_res = _InterpRes(step_from=solution_at_t1, interp_from=solution_at_t)

            # Extract targets
            marginals = solution_at_t.init
            u = ssm.stats.qoi(marginals)
            std = ssm.stats.standard_deviation(marginals)
            u_std = ssm.stats.qoi_from_sample(std)
            estimate = Estimate(u, u_std, marginals)
            return (estimate, solution_at_t), interp_res

        def interpolate_at_t1(self, state_t1):
            marginals = state_t1.init
            u = ssm.stats.qoi(marginals)
            std = ssm.stats.standard_deviation(marginals)
            u_std = ssm.stats.qoi_from_sample(std)
            estimate = Estimate(u, u_std, marginals)

            interp_res = _InterpRes(step_from=state_t1, interp_from=state_t1)
            return (estimate, state_t1), interp_res

    return Smoother(
        ssm=ssm,
        is_suitable_for_save_at=False,
        is_suitable_for_save_every_step=True,
        is_suitable_for_offgrid_marginals=True,
    )


def strategy_filter(*, ssm) -> _Strategy:
    """Construct a filter."""

    @containers.dataclass
    class Filter(_Strategy):
        def init_posterior(self, marginals, /):
            u = self.ssm.stats.qoi(marginals)
            std = self.ssm.stats.standard_deviation(marginals)
            u_std = self.ssm.stats.qoi_from_sample(std)
            estimate = Estimate(u, u_std, marginals)
            return estimate, marginals

        def predict(self, posterior, *, transition):
            marginals = self.ssm.conditional.marginalise(posterior, transition)
            u = self.ssm.stats.qoi(marginals)
            std = self.ssm.stats.standard_deviation(marginals)
            u_std = self.ssm.stats.qoi_from_sample(std)
            estimate = Estimate(u, u_std, marginals)
            return estimate, marginals

        def apply_updates(self, prediction, *, updates):
            del prediction
            marginals = updates
            u = self.ssm.stats.qoi(marginals)
            std = self.ssm.stats.standard_deviation(marginals)
            u_std = self.ssm.stats.qoi_from_sample(std)
            estimate = Estimate(u, u_std, marginals)
            return estimate, marginals

        def finalize(self, *, posterior0, posterior, scale):
            # Calibrate
            posterior0 = ssm.stats.rescale_cholesky(posterior0, scale[-1, ...])
            posterior = ssm.stats.rescale_cholesky(posterior, scale)

            # Stack
            posterior = tree_array_util.tree_prepend(posterior0, posterior)

            marginals = posterior
            u = ssm.stats.qoi(marginals)
            std = ssm.stats.standard_deviation(marginals)
            u_std = ssm.stats.qoi_from_sample(std)
            estimate = Estimate(u, u_std, marginals)
            return estimate, posterior

        def interpolate(self, state_t0, state_t1, transition_t0_t, transition_t_t1):
            del transition_t_t1
            _, interpolated = self.predict(state_t0, transition=transition_t0_t)

            u = ssm.stats.qoi(interpolated)
            std = ssm.stats.standard_deviation(interpolated)
            u_std = ssm.stats.qoi_from_sample(std)
            marginals = interpolated
            estimate = Estimate(u, u_std, marginals)

            interp_res = _InterpRes(step_from=state_t1, interp_from=interpolated)
            return (estimate, interpolated), interp_res

        def interpolate_at_t1(self, state_t1):
            u = ssm.stats.qoi(state_t1)
            std = ssm.stats.standard_deviation(state_t1)
            u_std = ssm.stats.qoi_from_sample(std)
            marginals = state_t1
            estimate = Estimate(u, u_std, marginals)

            interp_res = _InterpRes(step_from=state_t1, interp_from=state_t1)
            return (estimate, state_t1), interp_res

    return Filter(
        ssm=ssm,
        is_suitable_for_save_at=True,
        is_suitable_for_save_every_step=True,
        is_suitable_for_offgrid_marginals=True,
    )


def strategy_fixedpoint(*, ssm) -> _Strategy:
    """Construct a fixedpoint-smoother."""

    @containers.dataclass
    class FixedPoint(_Strategy):
        def init_posterior(self, marginals, /):
            cond = self.ssm.conditional.identity(ssm.num_derivatives + 1)
            posterior = stats.MarkovSeq(marginals, cond)

            u = self.ssm.stats.qoi(marginals)
            std = self.ssm.stats.standard_deviation(marginals)
            u_std = self.ssm.stats.qoi_from_sample(std)
            estimate = Estimate(u, u_std, marginals)
            return estimate, posterior

        def predict(self, posterior: stats.MarkovSeq, *, transition):
            rv = posterior.init
            bw0 = posterior.conditional
            marginals, cond = self.ssm.conditional.revert(rv, transition)
            cond = self.ssm.conditional.merge(bw0, cond)
            predicted = stats.MarkovSeq(marginals, cond)

            u = self.ssm.stats.qoi(marginals)
            std = self.ssm.stats.standard_deviation(marginals)
            u_std = self.ssm.stats.qoi_from_sample(std)
            estimate = Estimate(u, u_std, marginals)
            return estimate, predicted

        def apply_updates(self, prediction: stats.MarkovSeq, *, updates):
            posterior = stats.MarkovSeq(updates, prediction.conditional)
            rv = updates
            u = self.ssm.stats.qoi(rv)
            std = self.ssm.stats.standard_deviation(rv)
            u_std = self.ssm.stats.qoi_from_sample(std)
            marginals = rv
            estimate = Estimate(u, u_std, marginals)
            return estimate, posterior

        def finalize(
            self, *, posterior0: stats.MarkovSeq, posterior: stats.MarkovSeq, scale
        ):
            # Calibrate
            init = ssm.stats.rescale_cholesky(posterior0.init, scale[-1, ...])
            conditional = ssm.conditional.rescale_noise(posterior0.conditional, scale)
            posterior0 = stats.MarkovSeq(init, conditional)
            init = ssm.stats.rescale_cholesky(posterior.init, scale)
            conditional = ssm.conditional.rescale_noise(posterior.conditional, scale)
            posterior = stats.MarkovSeq(init, conditional)

            # Marginalise
            posterior_no_filter_marginals = stats.markov_select_terminal(posterior)
            marginals = stats.markov_marginals(
                posterior_no_filter_marginals, reverse=True, ssm=self.ssm
            )

            # Append the terminal marginal to the computed ones
            marginal_t1 = tree_util.tree_map(lambda s: s[-1, ...], posterior.init)
            marginals = tree_array_util.tree_append(marginals, marginal_t1)

            # Prepend the initial condition to the filtering distributions
            init = tree_array_util.tree_prepend(posterior0.init, posterior.init)
            posterior = stats.MarkovSeq(init=init, conditional=posterior.conditional)

            # Extract targets
            u = ssm.stats.qoi(marginals)
            std = ssm.stats.standard_deviation(marginals)
            u_std = ssm.stats.qoi_from_sample(std)
            estimate = Estimate(u, u_std, marginals)
            return estimate, posterior
            # # def prepend(a, A):
            # #     a = np.asarray(a)
            # #     A = np.asarray(A)
            # #     return np.concatenate([a[None, ...], A])

            # # result = tree_util.tree_map(prepend, state.step_from, solution)
            # # if isinstance(posterior, stats.MarkovSeq):
            #         # #     # Compute marginals
            # # #     posterior_no_filter_marginals = stats.markov_select_terminal(posterior)
            # # #     marginals = stats.markov_marginals(
            # # #         posterior_no_filter_marginals, reverse=True, ssm=ssm
            # # #     )

            # # #     # Prepend the marginal at t1 to the computed marginals
            # # #     marginal_t1 = tree_util.tree_map(lambda s: s[-1, ...], posterior.init)
            # # #     marginals = tree_array_util.tree_append(marginals, marginal_t1)

            # # #     # Prepend the marginal at t1 to the inits
            # # #     init = tree_array_util.tree_prepend(ssm_init, posterior.init)
            # # #     posterior = stats.MarkovSeq(init=init, conditional=posterior.conditional)
            # # # else:
            # # #     posterior = tree_array_util.tree_prepend(ssm_init, posterior)
            # # #     marginals = posterior

            # # TODO: should we always select the last item or not?
            # # Calibrate
            # init = ssm.stats.rescale_cholesky(posterior.init, scale)
            # conditional = ssm.conditional.rescale_noise(posterior.conditional, scale)

            # init = tree_util.tree_map(lambda x: x[-1, ...], init)
            # posterior = stats.MarkovSeq(init, conditional)

            # def smooth_step(x, cond):
            #     extrapolated = ssm.conditional.marginalise(x, cond)
            #     return extrapolated, extrapolated

            # init, xs = posterior.init, posterior.conditional
            # _, marginals = control_flow.scan(
            #     smooth_step, init=init, xs=xs, reverse=True
            # )

            # u = ssm.stats.qoi(marginals)
            # std = ssm.stats.standard_deviation(marginals)
            # u_std = ssm.stats.qoi_from_sample(std)
            # estimate = Estimate(u, u_std, marginals)
            # return estimate, posterior

        def interpolate_at_t1(self, state_t1: stats.MarkovSeq):
            cond_identity = self.ssm.conditional.identity(ssm.num_derivatives + 1)
            interpolated = stats.MarkovSeq(state_t1.init, conditional=cond_identity)
            interp_res = _InterpRes(step_from=interpolated, interp_from=interpolated)

            marginals = interpolated.init
            u = ssm.stats.qoi(marginals)
            std = ssm.stats.standard_deviation(marginals)
            u_std = ssm.stats.qoi_from_sample(std)
            estimate = Estimate(u, u_std, marginals)
            return (estimate, interpolated), interp_res

        def interpolate(
            self,
            *,
            state_t0: stats.MarkovSeq,
            state_t1: stats.MarkovSeq,
            transition_t0_t,
            transition_t_t1,
        ):
            """
            Interpolate using a fixed-point smoother.

            Assuming `state_t0` has seen 'n' collocation points, and `state_t1` has seen 'n+1'
            collocation points, then interpolation at time `t` is computed as follows:

            1. Extrapolate from `t0` to `t`. This yields:
                - the marginal at `t` given `n` observations.
                - the backward transition from `t` to `t0` given `n` observations.

            2. Extrapolate from `t` to `t1`. This yields:
                - the marginal at `t1` given `n` observations (in contrast, `state_t1` has seen `n+1` observations)
                - the backward transition from `t1` to `t` given `n` observations.

            3. Apply the backward transition from `t1` to `t` to the marginal inside `state_t1`
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
            _, extrapolated_t = self.predict(state_t0, transition=transition_t0_t)
            conditional_id = self.ssm.conditional.identity(ssm.num_derivatives + 1)
            previous_new = stats.MarkovSeq(extrapolated_t.init, conditional_id)
            _, extrapolated_t1 = self.predict(previous_new, transition=transition_t_t1)

            # Marginalise from t1 to t to obtain the interpolated solution.
            marginal_t1 = state_t1.init
            conditional_t1_to_t = extrapolated_t1.conditional
            rv_at_t = self.ssm.conditional.marginalise(marginal_t1, conditional_t1_to_t)

            # Return the right combination of marginals and conditionals.
            interpolated = stats.MarkovSeq(rv_at_t, extrapolated_t.conditional)
            step_from = stats.MarkovSeq(state_t1.init, conditional=conditional_t1_to_t)
            interp_res = _InterpRes(step_from=step_from, interp_from=previous_new)

            marginals = interpolated.init
            u = ssm.stats.qoi(marginals)
            std = ssm.stats.standard_deviation(marginals)
            u_std = ssm.stats.qoi_from_sample(std)
            estimate = Estimate(u, u_std, marginals)
            return (estimate, interpolated), interp_res

    return FixedPoint(
        ssm=ssm,
        is_suitable_for_save_at=True,
        is_suitable_for_save_every_step=False,
        is_suitable_for_offgrid_marginals=False,
    )


@containers.dataclass
class _Correction:
    """Correction model interface."""

    name: str
    ode_order: int
    ssm: Any
    linearize: Any
    vector_field: Callable
    re_linearize: bool

    def init(self, /):
        """Initialise the state from the solution."""
        return self.linearize.init()

    def estimate_error(self, rv, correction_state, /, t):
        """Estimate the error."""
        f_wrapped = functools.partial(self.vector_field, t=t)
        cond, correction_state = self.linearize.update(f_wrapped, rv, correction_state)
        observed = self.ssm.conditional.marginalise(rv, cond)

        zero_data = np.zeros(())
        output_scale = self.ssm.stats.mahalanobis_norm_relative(zero_data, rv=observed)
        stdev = self.ssm.stats.standard_deviation(observed)
        error_estimate_unscaled = np.squeeze(stdev)
        error_estimate = output_scale * error_estimate_unscaled
        return error_estimate, observed, (correction_state, cond)

    def updates(self, rv, correction_state, /, t):
        """Perform the correction step."""
        linearization_state, cond = correction_state

        if self.re_linearize:
            f_wrapped = functools.partial(self.vector_field, t=t)
            cond, linearization_state = self.linearize.update(
                f_wrapped, rv, linearization_state
            )

        observed, reverted = self.ssm.conditional.revert(rv, cond)
        corrected = reverted.noise
        return corrected, observed, linearization_state


def correction_ts0(vector_field, *, ssm, ode_order=1, damp: float = 0.0) -> _Correction:
    """Zeroth-order Taylor linearisation."""
    linearize = ssm.linearise.ode_taylor_0th(ode_order=ode_order, damp=damp)
    return _Correction(
        name="TS0",
        vector_field=vector_field,
        ode_order=ode_order,
        ssm=ssm,
        linearize=linearize,
        re_linearize=False,
    )


def correction_ts1(
    vector_field,
    *,
    ssm,
    ode_order=1,
    damp: float = 0.0,
    jvp_probes=10,
    jvp_probes_seed=1,
) -> _Correction:
    """First-order Taylor linearisation."""
    assert jvp_probes > 0
    linearize = ssm.linearise.ode_taylor_1st(
        ode_order=ode_order,
        damp=damp,
        jvp_probes=jvp_probes,
        jvp_probes_seed=jvp_probes_seed,
    )
    return _Correction(
        name="TS1",
        vector_field=vector_field,
        ode_order=ode_order,
        ssm=ssm,
        linearize=linearize,
        re_linearize=False,
    )


def correction_slr0(
    vector_field, *, ssm, cubature_fun=cubature_third_order_spherical, damp: float = 0.0
) -> _Correction:
    """Zeroth-order statistical linear regression."""
    linearize = ssm.linearise.ode_statistical_0th(cubature_fun, damp=damp)
    return _Correction(
        ssm=ssm,
        vector_field=vector_field,
        ode_order=1,
        linearize=linearize,
        name="SLR0",
        re_linearize=True,
    )


def correction_slr1(
    vector_field, *, ssm, cubature_fun=cubature_third_order_spherical, damp: float = 0.0
) -> _Correction:
    """First-order statistical linear regression."""
    linearize = ssm.linearise.ode_statistical_1st(cubature_fun, damp=damp)
    return _Correction(
        ssm=ssm,
        vector_field=vector_field,
        ode_order=1,
        linearize=linearize,
        name="SLR1",
        re_linearize=True,
    )


@containers.dataclass
class _Calibration:
    """Calibration implementation."""

    init: Callable
    update: Callable
    extract: Callable


@tree_util.register_dataclass
@containers.dataclass
class ProbSolverState:
    """Solver state."""

    t: Any
    """The current time-step."""

    estimate: Estimate
    """The current ODE solution estimate."""

    posterior: Any
    """The current hidden variable in the SSM."""

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


@tree_util.register_dataclass
@containers.dataclass
class _ErrorEstimate:
    estimate: ArrayLike
    reference: ArrayLike


@containers.dataclass
class _ProbabilisticSolver:
    strategy: _Strategy
    prior: Callable
    ssm: Any
    correction: _Correction

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

    def init(self, t, init) -> ProbSolverState:
        raise NotImplementedError

    def step(self, state: ProbSolverState, *, dt):
        raise NotImplementedError

    def userfriendly_output(
        self, *, solution: ProbSolverState, solution0: ProbSolverState
    ):
        raise NotImplementedError

    # def extract(self, state: ProbSolverState, /):
    #     posterior = self.strategy.extract(state.rv, state.strategy_state)
    #     t = state.t

    #     _output_scale_prior, output_scale = self.calibration.extract(state.output_scale)
    #     return t, (posterior, output_scale)

    def offgrid_marginals(self, *, t, marginals_t1, posterior_t0, t0, t1, output_scale):
        """Compute offgrid_marginals."""
        if not self.is_suitable_for_offgrid_marginals:
            raise NotImplementedError

        # TODO: how is this function different to interpolate() now?

        _estimate, state_t0 = self.strategy.init_posterior(posterior_t0)
        _estimate, state_t1 = self.strategy.init_posterior(marginals_t1)

        transition_t0_t = self.prior(t - t0, output_scale)
        transition_t_t1 = self.prior(t1 - t, output_scale)
        (estimate, _posterior), _interp_res = self.strategy.interpolate(
            state_t0=state_t0,
            state_t1=state_t1,
            transition_t0_t=transition_t0_t,
            transition_t_t1=transition_t_t1,
        )
        return estimate.u, estimate.marginals  # todo: return "estimate"?

    def interpolate(
        self, *, t, interp_from: ProbSolverState, interp_to: ProbSolverState
    ) -> _InterpRes:
        # Domain is (t0, t1]; thus, take the output scale from interp_to
        output_scale = interp_to.output_scale
        auxiliary = interp_to.auxiliary

        # Interpolate
        transition_t0_t = self.prior(t - interp_from.t, output_scale)
        transition_t_t1 = self.prior(interp_to.t - t, output_scale)
        tmp = self.strategy.interpolate(
            state_t0=interp_from.posterior,
            state_t1=interp_to.posterior,
            transition_t0_t=transition_t0_t,
            transition_t_t1=transition_t_t1,
        )
        (estimate, interpolated), step_and_interpolate_from = tmp

        step_from = ProbSolverState(
            t=interp_to.t,
            # New:
            posterior=step_and_interpolate_from.step_from,
            # Old:
            estimate=interp_to.estimate,
            output_scale=interp_to.output_scale,
            auxiliary=interp_to.auxiliary,
            num_steps=interp_to.num_steps,
        )

        interpolated = ProbSolverState(
            t=t,
            # New:
            posterior=interpolated,
            estimate=estimate,
            # Taken from the rhs point
            output_scale=interp_to.output_scale,
            auxiliary=interp_to.auxiliary,
            num_steps=interp_to.num_steps,
        )

        interp_from = ProbSolverState(
            t=t,
            # New:
            posterior=step_and_interpolate_from.interp_from,
            # Old:
            estimate=interp_from.estimate,
            output_scale=interp_from.output_scale,
            auxiliary=interp_from.auxiliary,
            num_steps=interp_from.num_steps,
        )

        interp_res = _InterpRes(step_from=step_from, interp_from=interp_from)
        return interpolated, interp_res

    def interpolate_at_t1(
        self, *, t, interp_from: ProbSolverState, interp_to: ProbSolverState
    ) -> _InterpRes:
        """Process the solution in case t=t_n."""
        del t
        tmp = self.strategy.interpolate_at_t1(state_t1=interp_to.posterior)
        (estimate, interpolated), step_and_interpolate_from = tmp

        prev = ProbSolverState(
            t=interp_to.t,
            # New
            posterior=step_and_interpolate_from.interp_from,
            # Old
            estimate=interp_from.estimate,  # incorrect?
            output_scale=interp_from.output_scale,  # incorrect?
            auxiliary=interp_from.auxiliary,  # incorrect?
            num_steps=interp_from.num_steps,  # incorrect?
        )
        sol = ProbSolverState(
            t=interp_to.t,
            # New:
            posterior=interpolated,
            estimate=estimate,
            # Old:
            output_scale=interp_to.output_scale,
            auxiliary=interp_to.auxiliary,
            num_steps=interp_to.num_steps,
        )
        acc = ProbSolverState(
            t=interp_to.t,
            # New:
            posterior=step_and_interpolate_from.step_from,
            # Old
            estimate=interp_to.estimate,
            output_scale=interp_to.output_scale,
            auxiliary=interp_to.auxiliary,
            num_steps=interp_to.num_steps,
        )
        return sol, _InterpRes(step_from=acc, interp_from=prev)


class solver_mle(_ProbabilisticSolver):
    """Create a solver that calibrates the output scale via maximum-likelihood.

    Warning: needs to be combined with a call to stats.calibrate()
    after solving if the MLE-calibration shall be *used*.
    """

    def init(self, t, init) -> ProbSolverState:
        estimate, posterior = self.strategy.init_posterior(init)
        correction_state = self.correction.init()

        output_scale_prior = np.ones_like(self.ssm.prototypes.output_scale())
        output_scale_running = 0 * output_scale_prior
        auxiliary = (correction_state, output_scale_running, 0)
        return ProbSolverState(
            t=t,
            estimate=estimate,
            posterior=posterior,
            auxiliary=auxiliary,
            output_scale=output_scale_prior,
            num_steps=0,
        )

    def step(self, state, *, dt):
        u_step_from = state.estimate.u
        (correction_state, output_scale_running, num_data) = state.auxiliary

        # Discretize
        transition = self.prior(dt, state.output_scale)

        # Estimate the error
        mean = self.ssm.stats.mean(state.estimate.marginals)
        mean_extra = self.ssm.conditional.apply(mean, transition)
        t = state.t + dt
        error, _, correction_state = self.correction.estimate_error(
            mean_extra, correction_state, t=t
        )

        # Do the full prediction step (reuse previous discretisation)
        hidden, prediction = self.strategy.predict(
            state.posterior, transition=transition
        )

        # Do the full correction step
        updates, observed, correction_state = self.correction.updates(
            hidden.marginals, correction_state, t=t
        )
        estimate, posterior = self.strategy.apply_updates(prediction, updates=updates)

        # Calibrate the output scale
        new_term = self.ssm.stats.mahalanobis_norm_relative(0.0, observed)
        output_scale_running = self.ssm.stats.update_mean(
            output_scale_running, new_term, num=num_data
        )

        # Normalise the error
        auxiliary = (correction_state, output_scale_running, num_data + 1)

        state = ProbSolverState(
            t=t,
            estimate=estimate,
            posterior=posterior,
            output_scale=state.output_scale,
            auxiliary=auxiliary,
            num_steps=state.num_steps + 1,
        )
        u0 = tree_util.tree_leaves(u_step_from)[0]
        u1 = tree_util.tree_leaves(estimate.u)[0]
        reference = np.maximum(np.abs(u0), np.abs(u1))
        error = _ErrorEstimate(dt * error, reference=reference)
        return error, state

    def userfriendly_output(
        self, *, solution0: ProbSolverState, solution: ProbSolverState
    ) -> ProbSolverState:
        # This is the MLE solver, so we take the calibrated scale
        _, output_scale, _ = solution.auxiliary

        init = solution0.posterior
        posterior = solution.posterior
        estimate, posterior = self.strategy.finalize(
            posterior0=init, posterior=posterior, scale=output_scale
        )

        ts = np.concatenate([solution0.t[None], solution.t])
        return ProbSolverState(
            t=ts,
            estimate=estimate,
            posterior=posterior,
            output_scale=output_scale,
            num_steps=solution.num_steps,
            auxiliary=solution.auxiliary,
        )

        # posterior = solution.posterior
        # _, output_scale, _ = solution.auxiliary  # calibrated scale!
        # estimate, posterior = self.strategy.finalize(posterior, scale=output_scale)
        # return ProbSolverState(
        #     t=solution.t,
        #     estimate=estimate,
        #     posterior=posterior,
        #     output_scale=output_scale,
        #     auxiliary=solution.auxiliary,
        #     num_steps=solution.num_steps,
        # )

        # if isinstance(posterior, stats.MarkovSeq):
        #     # Compute marginals
        #     posterior_no_filter_marginals = stats.markov_select_terminal(posterior)
        #     marginals = stats.markov_marginals(
        #         posterior_no_filter_marginals, reverse=True, ssm=ssm
        #     )

        #     # Prepend the marginal at t1 to the computed marginals
        #     marginal_t1 = tree_util.tree_map(lambda s: s[-1, ...], posterior.init)
        #     marginals = tree_array_util.tree_append(marginals, marginal_t1)

        #     # Prepend the marginal at t1 to the inits
        #     init = tree_array_util.tree_prepend(ssm_init, posterior.init)
        #     posterior = stats.MarkovSeq(init=init, conditional=posterior.conditional)
        # else:
        #     posterior = tree_array_util.tree_prepend(ssm_init, posterior)
        #     marginals = posterior
        # return marginals, posterior

        # assert False


# def _calibration_running(ssm):
#     def init():
#         prior = np.ones_like(ssm.prototypes.output_scale())
#         return prior, prior, 0.0

#     def update(state, /, observed):
#         prior, calibrated, num_data = state

#         new_term = ssm.stats.mahalanobis_norm_relative(0.0, observed)
#         calibrated = ssm.stats.update_mean(calibrated, new_term, num=num_data)
#         return prior, calibrated, num_data + 1.0

#     def extract(state, /):
#         prior, calibrated, _num_data = state
#         return prior, calibrated
#
# return _Calibration(init=init, update=update, extract=extract)


def solver_dynamic(strategy, *, correction, prior, ssm):
    """Create a solver that calibrates the output scale dynamically."""

    def step_dynamic(state, /, *, dt, calibration):
        u_step_from = tree_util.ravel_pytree(ssm.unravel(state.rv.mean)[0])[0]

        # Estimate error and calibrate the output scale
        ones = np.ones_like(ssm.prototypes.output_scale())
        transition = prior(dt, ones)
        mean = ssm.stats.mean(state.rv)
        hidden = ssm.conditional.apply(mean, transition)

        t = state.t + dt
        error, observed, correction_state = correction.estimate_error(
            hidden, state.correction_state, t=t
        )
        output_scale = calibration.update(state.output_scale, observed=observed)

        # Do the full extrapolation with the calibrated output scale
        scale, _ = calibration.extract(output_scale)
        transition = prior(dt, scale)
        hidden, extra = strategy.extrapolate(
            state.rv, state.strategy_state, transition=transition
        )

        # Do the full correction step
        hidden, _, correction_state = correction.correct(hidden, correction_state, t=t)

        # Return solution
        state = ProbSolverState(
            t=t,
            rv=hidden,
            strategy_state=extra,
            correction_state=correction_state,
            output_scale=output_scale,
        )

        # Normalise the error
        u_proposed = tree_util.ravel_pytree(ssm.unravel(state.rv.mean)[0])[0]
        reference = np.maximum(np.abs(u_proposed), np.abs(u_step_from))
        error = _ErrorEstimate(dt * error, reference=reference)
        return error, state

    return _ProbabilisticSolver(
        prior=prior,
        ssm=ssm,
        strategy=strategy,
        correction=correction,
        calibration=_calibration_most_recent(ssm=ssm),
        name="Dynamic probabilistic solver",
        step_implementation=step_dynamic,
    )


def _calibration_most_recent(*, ssm) -> _Calibration:
    def init():
        return np.ones_like(ssm.prototypes.output_scale())

    def update(_state, /, observed):
        return ssm.stats.mahalanobis_norm_relative(0.0, observed)

    def extract(state, /):
        return state, state

    return _Calibration(init=init, update=update, extract=extract)


class solver(_ProbabilisticSolver):
    """Create a solver that does not calibrate the output scale automatically."""

    def init(self, t, init) -> ProbSolverState:
        estimate, posterior = self.strategy.init_posterior(init)
        correction_state = self.correction.init()
        output_scale = np.ones_like(self.ssm.prototypes.output_scale())
        return ProbSolverState(
            t=t,
            estimate=estimate,
            posterior=posterior,
            num_steps=0,
            auxiliary=(correction_state,),
            output_scale=output_scale,
        )

    def step(self, state: ProbSolverState, *, dt):
        # Save a reference value for error estimation

        u_step_from = state.estimate.u

        # Discretise
        transition = self.prior(dt, state.output_scale)

        # Estimate the error.
        # I hate this code and want to get rid of it.
        # With cleaner error estimation, the solvers would be
        # so much simpler to implement!
        mean = self.ssm.stats.mean(state.estimate.marginals)
        hidden = self.ssm.conditional.apply(mean, transition)

        t = state.t + dt
        (correction_state,) = state.auxiliary
        error, _, correction_state = self.correction.estimate_error(
            hidden, correction_state, t=t
        )

        # Do the full extrapolation step (reuse the transition)
        hidden, prediction = self.strategy.predict(
            state.posterior, transition=transition
        )
        updates, _, correction_state = self.correction.updates(
            hidden.marginals, correction_state, t=t
        )
        estimate, posterior = self.strategy.apply_updates(prediction, updates=updates)
        state = ProbSolverState(
            t=t,
            estimate=estimate,
            posterior=posterior,
            output_scale=state.output_scale,
            auxiliary=(correction_state,),
            num_steps=state.num_steps + 1,
        )

        # Normalise the error
        u0 = tree_util.tree_leaves(u_step_from)[0]
        u1 = tree_util.tree_leaves(estimate.u)[0]
        reference = np.maximum(np.abs(u0), np.abs(u1))
        error = _ErrorEstimate(dt * error, reference=reference)
        return error, state

    def userfriendly_output(
        self, *, solution0: ProbSolverState, solution: ProbSolverState
    ) -> ProbSolverState:
        # This is the uncalibrated solver, so scale=1
        output_scale = np.ones_like(solution.output_scale)

        init = solution0.posterior
        posterior = solution.posterior
        estimate, posterior = self.strategy.finalize(
            posterior0=init, posterior=posterior, scale=output_scale
        )

        ts = np.concatenate([solution0.t[None], solution.t])
        return ProbSolverState(
            t=ts,
            estimate=estimate,
            posterior=posterior,
            output_scale=output_scale,
            num_steps=solution.num_steps,
            auxiliary=solution.auxiliary,
        )


def _calibration_none(*, ssm) -> _Calibration:
    def init():
        return np.ones_like(ssm.prototypes.output_scale())

    def update(_state, /, observed):
        raise NotImplementedError

    def extract(state, /):
        return state, state

    return _Calibration(init=init, update=update, extract=extract)


def adaptive(
    slvr,
    /,
    *,
    ssm,
    atol=1e-4,
    rtol=1e-2,
    control=None,
    norm_ord=None,
    clip_dt: bool = False,
    eps: float | None = None,
):
    """Make an IVP solver adaptive."""
    if control is None:
        control = control_proportional_integral()
    if eps is None:
        eps = 10 * np.finfo_eps(float)
    return _AdaSolver(
        slvr,
        ssm=ssm,
        atol=atol,
        rtol=rtol,
        control=control,
        norm_ord=norm_ord,
        clip_dt=clip_dt,
        eps=eps,
    )


class _AnythingWithTimeAttribute(Protocol):
    t: float


A = TypeVar("A", bound=_AnythingWithTimeAttribute)


@tree_util.register_dataclass
@containers.dataclass
class _AdaState(Generic[A]):
    dt: float
    step_from: A
    interp_from: A
    control: Any


class _AdaSolver:
    """Adaptive IVP solvers."""

    def __init__(
        self,
        slvr: _ProbabilisticSolver,
        /,
        *,
        atol,
        rtol,
        control,
        norm_ord,
        ssm,
        clip_dt: bool,
        eps: float,
    ):
        self.solver = slvr
        self.atol = atol
        self.rtol = rtol
        self.control = control
        self.norm_ord = norm_ord
        self.ssm = ssm
        self.clip_dt = clip_dt
        self.eps = eps

    def __repr__(self):
        return (
            f"\n{self.__class__.__name__}("
            f"\n\tsolver={self.solver},"
            f"\n\tatol={self.atol},"
            f"\n\trtol={self.rtol},"
            f"\n\tcontrol={self.control},"
            f"\n\tnorm_order={self.norm_ord},"
            "\n)"
        )

    @functools.jit
    def init(self, t, initial_condition, dt) -> _AdaState:
        """Initialise the IVP solver state."""
        state_solver = self.solver.init(t, initial_condition)
        state_control = self.control.init(dt)
        return _AdaState(
            dt=dt,
            step_from=state_solver,
            interp_from=state_solver,
            control=state_control,
        )

    @functools.jit
    def rejection_loop(self, state0: _AdaState, *, t1) -> _AdaState:
        class _RejectionState(containers.NamedTuple):
            """State for rejection loops.

            (Keep decreasing step-size until error norm is small.
            This is one part of an IVP solver step.)
            """

            dt: float
            error_norm_proposed: float
            control: Any
            proposed: Any
            step_from: Any

        def init(s0: _AdaState) -> _RejectionState:
            def _ones_like(tree):
                return tree_util.tree_map(np.ones_like, tree)

            smaller_than_1 = 0.9  # the cond() must return True
            return _RejectionState(
                error_norm_proposed=smaller_than_1,
                dt=s0.dt,
                control=s0.control,
                proposed=_ones_like(s0.step_from),  # irrelevant
                step_from=s0.step_from,
            )

        def cond_fn(state: _RejectionState) -> bool:
            # error_norm_proposed is EEst ** (-1/rate), thus "<"
            return state.error_norm_proposed < 1.0

        def body_fn(state: _RejectionState) -> _RejectionState:
            """Attempt a step.

            Perform a step with an IVP solver and
            propose a future time-step based on tolerances and error estimates.
            """
            dt = state.dt

            # Some controllers like to clip the terminal value instead of interpolating.
            # This must happen _before_ the step.
            if self.clip_dt:
                dt = np.minimum(dt, t1 - state.step_from.t)

            # Perform the actual step.
            error_estimate, state_proposed = self.solver.step(
                state=state.step_from, dt=dt
            )

            # Propose a new step
            error_power = self._error_scale_and_normalize(error_estimate)
            dt, state_control = self.control.apply(
                dt, state.control, error_power=error_power
            )
            return _RejectionState(
                dt=dt,
                error_norm_proposed=error_power,  # new
                proposed=state_proposed,  # new
                control=state_control,  # new
                step_from=state.step_from,
            )

        def extract(s: _RejectionState) -> _AdaState:
            return _AdaState(
                dt=s.dt,
                step_from=s.proposed,
                interp_from=s.step_from,
                control=s.control,
            )

        init_val = init(state0)
        state_new = control_flow.while_loop(cond_fn, body_fn, init_val)
        return extract(state_new)

    def _error_scale_and_normalize(self, error: _ErrorEstimate):
        assert isinstance(error, _ErrorEstimate)
        normalize = self.atol + self.rtol * np.abs(error.reference)
        error_relative = error.estimate / normalize

        dim = np.atleast_1d(error.reference).size
        error_norm = linalg.vector_norm(error_relative, order=self.norm_ord)
        error_norm_rel = error_norm / np.sqrt(dim)
        return error_norm_rel ** (-1.0 / self.solver.error_contraction_rate)

    def extract_before_t1(self, state: _AdaState, t):
        del t
        # solution_solver = self.solver.extract(state.step_from)
        # extracted = solution_solver
        return state, state.step_from

    def extract_at_t1(self, state: _AdaState, t):
        # TODO: make the "at t1" decision inside solver.interpolate(),
        #  which collapses the next two functions together
        interpolated, interp_res = self.solver.interpolate_at_t1(
            t=t, interp_from=state.interp_from, interp_to=state.step_from
        )
        state = _AdaState(
            dt=state.dt,
            step_from=interp_res.step_from,
            interp_from=interp_res.interp_from,
            control=state.control,
        )
        return state, interpolated

    def extract_after_t1(self, state: _AdaState, t):
        """We have overstepped, so we interpolate before returning."""
        interpolated, interp_res = self.solver.interpolate(
            t=t, interp_from=state.interp_from, interp_to=state.step_from
        )
        state = _AdaState(
            dt=state.dt,
            step_from=interp_res.step_from,
            interp_from=interp_res.interp_from,
            control=state.control,
        )
        return state, interpolated

    # def _extract_interpolate(self, interpolated, interp_res: _InterpRes, state):
    #     state = _AdaState(
    #         dt=state.dt,
    #         step_from=interp_res.step_from,
    #         interp_from=interp_res.interp_from,
    #         control=state.control,
    #         stats=state.stats,
    #     )

    #     solution_solver = self.solver.extract(interpolated)
    #     return state, (solution_solver, (state.dt, state.control), state.stats)

    @staticmethod
    def register_pytree_node():
        def _asolver_flatten(asolver):
            children = (asolver.atol, asolver.rtol, asolver.eps)
            aux = (
                asolver.solver,
                asolver.control,
                asolver.norm_ord,
                asolver.ssm,
                asolver.clip_dt,
            )
            return children, aux

        def _asolver_unflatten(aux, children):
            atol, rtol, eps = children
            (slvr, control, norm_ord, ssm, clip_dt) = aux
            return _AdaSolver(
                slvr,
                atol=atol,
                rtol=rtol,
                control=control,
                norm_ord=norm_ord,
                ssm=ssm,
                clip_dt=clip_dt,
                eps=eps,
            )

        tree_util.register_pytree_node(
            _AdaSolver, flatten_func=_asolver_flatten, unflatten_func=_asolver_unflatten
        )


_AdaSolver.register_pytree_node()


@containers.dataclass
class _Controller(Generic[T]):
    """Control algorithm."""

    init: Callable[[float], T]
    """Initialise the controller state."""

    apply: Callable[[float, T, NamedArg(float, "error_power")], tuple[float, T]]
    r"""Propose a time-step $\Delta t$."""


def control_proportional_integral(
    *,
    safety=0.95,
    factor_min=0.2,
    factor_max=10.0,
    power_integral_unscaled=0.3,
    power_proportional_unscaled=0.4,
) -> _Controller[float]:
    """Construct a proportional-integral-controller with time-clipping."""

    def init(_dt: float, /) -> float:
        return 1.0

    def apply(dt: float, error_power_prev: float, /, *, error_power):
        # Equivalent: error_power = error_norm ** (-1.0 / error_contraction_rate)
        a1 = error_power**power_integral_unscaled
        a2 = (error_power / error_power_prev) ** power_proportional_unscaled
        scale_factor_unclipped = safety * a1 * a2

        scale_factor_clipped_min = np.minimum(scale_factor_unclipped, factor_max)
        scale_factor = np.maximum(factor_min, scale_factor_clipped_min)

        # >= 1.0 because error_power is 1/scaled_error_norm
        error_power_prev = np.where(error_power >= 1.0, error_power, error_power_prev)

        dt_proposed = scale_factor * dt
        return dt_proposed, error_power_prev

    return _Controller(init=init, apply=apply)


def control_integral(
    *, safety=0.95, factor_min=0.2, factor_max=10.0
) -> _Controller[None]:
    """Construct an integral-controller."""

    def init(_dt, /) -> None:
        return None

    def apply(dt, _state, /, *, error_power):
        # error_power = error_norm ** (-1.0 / error_contraction_rate)
        scale_factor_unclipped = safety * error_power

        scale_factor_clipped_min = np.minimum(scale_factor_unclipped, factor_max)
        scale_factor = np.maximum(factor_min, scale_factor_clipped_min)
        return scale_factor * dt, None

    return _Controller(init=init, apply=apply)
