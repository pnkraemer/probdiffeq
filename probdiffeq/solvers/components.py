"""IVP solver components."""

from probdiffeq.backend import abc, containers, functools, special, tree_util
from probdiffeq.backend import numpy as np
from probdiffeq.backend.typing import Array
from probdiffeq.impl import impl
from probdiffeq.solvers import markov


def ibm_adaptive(num_derivatives, output_scale=None):
    """Construct an adaptive(/continuous-time), multiply-integrated Wiener process."""
    output_scale = output_scale or np.ones_like(impl.prototypes.output_scale())
    discretise = impl.ssm_util.ibm_transitions(num_derivatives, output_scale)
    return discretise, num_derivatives


def ibm_discretised(ts, *, num_derivatives, output_scale=None):
    """Compute a time-discretised, multiply-integrated Wiener process."""
    discretise, _ = ibm_adaptive(num_derivatives, output_scale=output_scale)
    transitions, (p, p_inv) = functools.vmap(discretise)(np.diff(ts))

    preconditioner_apply_vmap = functools.vmap(impl.ssm_util.preconditioner_apply_cond)
    conditionals = preconditioner_apply_vmap(transitions, p, p_inv)

    output_scale = np.ones_like(impl.prototypes.output_scale())
    init = impl.ssm_util.standard_normal(num_derivatives + 1, output_scale=output_scale)
    return markov.MarkovSeq(init, conditionals)


class _Correction(abc.ABC):
    """Correction model interface."""

    def __init__(self, ode_order):
        self.ode_order = ode_order

    @abc.abstractmethod
    def init(self, x, /):
        """Initialise the state from the solution."""
        raise NotImplementedError

    @abc.abstractmethod
    def estimate_error(self, ssv, corr, /, vector_field, t):
        """Perform all elements of the correction until the error estimate."""
        raise NotImplementedError

    @abc.abstractmethod
    def complete(self, ssv, corr, /):
        """Complete what has been left out by `estimate_error`."""
        raise NotImplementedError

    @abc.abstractmethod
    def extract(self, ssv, corr, /):
        """Extract the solution from the state."""
        raise NotImplementedError


class _ODEConstraintTaylor(_Correction):
    def __init__(self, ode_order, linearise_fun, string_repr):
        super().__init__(ode_order=ode_order)

        self.linearise = linearise_fun
        self.string_repr = string_repr

    def __repr__(self):
        return self.string_repr

    def init(self, ssv, /):
        obs_like = impl.prototypes.observed()
        return ssv, obs_like

    def estimate_error(self, hidden_state, _corr, /, vector_field, t):
        def f_wrapped(s):
            return vector_field(*s, t=t)

        A, b = self.linearise(f_wrapped, hidden_state.mean)
        observed = impl.transform.marginalise(hidden_state, (A, b))

        error_estimate = _estimate_error(observed)
        return error_estimate, observed, (A, b)

    def complete(self, hidden_state, corr, /):
        A, b = corr
        observed, (_gain, corrected) = impl.transform.revert(hidden_state, (A, b))
        return corrected, observed

    def extract(self, ssv, _corr, /):
        return ssv


class _ODEConstraintStatistical(_Correction):
    def __init__(self, ode_order, linearise_fun, string_repr):
        super().__init__(ode_order=ode_order)

        self.linearise = linearise_fun
        self.string_repr = string_repr

    def __repr__(self):
        return self.string_repr

    def init(self, ssv, /):
        obs_like = impl.prototypes.observed()
        return ssv, obs_like

    def estimate_error(self, hidden_state, _corr, /, vector_field, t):
        f_wrapped = functools.partial(vector_field, t=t)
        A, b = self.linearise(f_wrapped, hidden_state)
        observed = impl.conditional.marginalise(hidden_state, (A, b))

        error_estimate = _estimate_error(observed)
        return error_estimate, observed, (A, b, f_wrapped)

    def complete(self, hidden_state, corr, /):
        # Re-linearise (because the linearisation point changed)
        *_, f_wrapped = corr
        A, b = self.linearise(f_wrapped, hidden_state)

        # Condition
        observed, (_gain, corrected) = impl.conditional.revert(hidden_state, (A, b))
        return corrected, observed

    def extract(self, hidden_state, _corr, /):
        return hidden_state


def _estimate_error(observed, /):
    # TODO: the functions involved in error estimation are still a bit patchy.
    #  for instance, they assume that they are called in exactly this error estimation
    #  context. Same for prototype_qoi etc.
    zero_data = np.zeros(())
    output_scale = impl.stats.mahalanobis_norm_relative(zero_data, rv=observed)
    error_estimate_unscaled = np.squeeze(impl.stats.standard_deviation(observed))
    return output_scale * error_estimate_unscaled


def ts0(*, ode_order=1) -> _ODEConstraintTaylor:
    """Zeroth-order Taylor linearisation."""
    return _ODEConstraintTaylor(
        ode_order=ode_order,
        linearise_fun=impl.linearise.ode_taylor_0th(ode_order=ode_order),
        string_repr=f"<TS0 with ode_order={ode_order}>",
    )


def ts1(*, ode_order=1) -> _ODEConstraintTaylor:
    """First-order Taylor linearisation."""
    return _ODEConstraintTaylor(
        ode_order=ode_order,
        linearise_fun=impl.linearise.ode_taylor_1st(ode_order=ode_order),
        string_repr=f"<TS1 with ode_order={ode_order}>",
    )


def slr0(cubature_fun=None) -> _ODEConstraintStatistical:
    """Zeroth-order statistical linear regression."""
    cubature_fun = cubature_fun or third_order_spherical
    linearise_fun = impl.linearise.ode_statistical_1st(cubature_fun)
    return _ODEConstraintStatistical(
        ode_order=1,
        linearise_fun=linearise_fun,
        string_repr=f"<SLR1 with ode_order={1}>",
    )


def slr1(cubature_fun=None) -> _ODEConstraintStatistical:
    """First-order statistical linear regression."""
    cubature_fun = cubature_fun or third_order_spherical
    linearise_fun = impl.linearise.ode_statistical_0th(cubature_fun)
    return _ODEConstraintStatistical(
        ode_order=1,
        linearise_fun=linearise_fun,
        string_repr=f"<SLR0 with ode_order={1}>",
    )


class PositiveCubatureRule(containers.NamedTuple):
    """Cubature rule with positive weights."""

    points: Array
    weights_sqrtm: Array


def third_order_spherical(input_shape) -> PositiveCubatureRule:
    """Third-order spherical cubature integration."""
    assert len(input_shape) <= 1
    if len(input_shape) == 1:
        (d,) = input_shape
        points_mat, weights_sqrtm = _third_order_spherical_params(d=d)
        return PositiveCubatureRule(points=points_mat, weights_sqrtm=weights_sqrtm)

    # If input_shape == (), compute weights via input_shape=(1,)
    # and 'squeeze' the points.
    points_mat, weights_sqrtm = _third_order_spherical_params(d=1)
    (S, _) = points_mat.shape
    points = np.reshape(points_mat, (S,))
    return PositiveCubatureRule(points=points, weights_sqrtm=weights_sqrtm)


def _third_order_spherical_params(*, d):
    eye_d = np.eye(d) * np.sqrt(d)
    pts = np.concatenate((eye_d, -1 * eye_d))
    weights_sqrtm = np.ones((2 * d,)) / np.sqrt(2.0 * d)
    return pts, weights_sqrtm


def unscented_transform(input_shape, r=1.0) -> PositiveCubatureRule:
    """Unscented transform."""
    assert len(input_shape) <= 1
    if len(input_shape) == 1:
        (d,) = input_shape
        points_mat, weights_sqrtm = _unscented_transform_params(d=d, r=r)
        return PositiveCubatureRule(points=points_mat, weights_sqrtm=weights_sqrtm)

    # If input_shape == (), compute weights via input_shape=(1,)
    # and 'squeeze' the points.
    points_mat, weights_sqrtm = _unscented_transform_params(d=1, r=r)
    (S, _) = points_mat.shape
    points = np.reshape(points_mat, (S,))
    return PositiveCubatureRule(points=points, weights_sqrtm=weights_sqrtm)


def _unscented_transform_params(d, *, r):
    eye_d = np.eye(d) * np.sqrt(d + r)
    zeros = np.zeros((1, d))
    pts = np.concatenate((eye_d, zeros, -1 * eye_d))
    _scale = d + r
    weights_sqrtm1 = np.ones((d,)) / np.sqrt(2.0 * _scale)
    weights_sqrtm2 = np.sqrt(r / _scale)
    weights_sqrtm = np.hstack((weights_sqrtm1, weights_sqrtm2, weights_sqrtm1))
    return pts, weights_sqrtm


def gauss_hermite(input_shape, degree=5) -> PositiveCubatureRule:
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
    return PositiveCubatureRule(points=tensor_pts, weights_sqrtm=tensor_weights_sqrtm)


# how does this generalise to an input_shape instead of an input_dimension?
# via tree_map(lambda s: _tensor_points(x, s), input_shape)?


def _tensor_weights(*args, **kwargs):
    mesh = _tensor_points(*args, **kwargs)
    return np.prod_along_axis(mesh, axis=1)


def _tensor_points(x, /, *, d):
    x_mesh = np.meshgrid(*([x] * d))
    y_mesh = tree_util.tree_map(lambda s: np.reshape(s, (-1,)), x_mesh)
    return np.stack(y_mesh).T
