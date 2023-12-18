"""Discrete filtering and smoothing."""

from probdiffeq.backend import containers, control_flow, tree_util
from probdiffeq.backend.typing import Any
from probdiffeq.impl import impl


# TODO: fixedpointsmoother and kalmanfilter should be estimate()
#  with two different methods. This saves a lot of code.
def estimate_fwd(data, /, init, prior_transitions, observation_model, *, estimator):
    """Estimate forward-in-time."""
    initialise, step = estimator

    # Incorporate final data point
    information_terminal = _select((data, observation_model), idx_or_slice=0)
    init = initialise(init, *information_terminal)

    # Scan over the remaining data points
    idx_or_slice = slice(1, len(data), 1)
    information = _select((data, observation_model), idx_or_slice=idx_or_slice)
    xs = (prior_transitions, *information)
    return control_flow.scan(step, init=init, xs=xs, reverse=False)


def estimate_rev(data, /, init, prior_transitions, observation_model, *, estimator):
    """Estimate reverse-in-time."""
    initialise, step = estimator

    # Incorporate final data point
    information_terminal = _select((data, observation_model), idx_or_slice=-1)
    init = initialise(init, *information_terminal)

    # Scan over the remaining data points
    information = _select((data, observation_model), idx_or_slice=slice(0, -1, 1))
    xs = (prior_transitions, *information)
    return control_flow.scan(step, init=init, xs=xs, reverse=True)


def fixedpointsmoother_precon():
    """Construct a discrete, preconditioned fixedpoint-smoother."""

    class _FPState(containers.NamedTuple):
        rv: Any
        conditional: Any

    def _initialise(init, data, observation_model) -> _FPState:
        rv, cond = init
        _observed, conditional = impl.conditional.revert(rv, observation_model)
        corrected = impl.conditional.apply(data, conditional)
        return _FPState(corrected, cond)

    def _step(state: _FPState, cond_and_data_and_obs) -> tuple[_FPState, _FPState]:
        (conditional, (p, p_inv)), data, observation = cond_and_data_and_obs
        rv, conditional_rev = state

        # Extrapolate
        rv = impl.ssm_util.preconditioner_apply(rv, p_inv)
        rv, conditional_new = impl.conditional.revert(rv, conditional)
        rv = impl.ssm_util.preconditioner_apply(rv, p)
        conditional_new = impl.ssm_util.preconditioner_apply_cond(
            conditional_new, p, p_inv
        )
        conditional_rev_new = impl.conditional.merge(conditional_rev, conditional_new)

        # Correct
        _observed, reverse = impl.conditional.revert(rv, observation)
        corrected = impl.conditional.apply(data, reverse)

        # Scan-compatible output
        state = _FPState(corrected, conditional_rev_new)
        return state, state

    return _initialise, _step


def _select(tree, idx_or_slice):
    return tree_util.tree_map(lambda s: s[idx_or_slice, ...], tree)


def kalmanfilter_with_marginal_likelihood():
    """Construct a Kalman-filter-implementation of computing the marginal likelihood."""

    class _KFState(containers.NamedTuple):
        rv: Any
        num_data_points: float
        logpdf: float

    def _initialise(rv, data, model) -> _KFState:
        observed, conditional = impl.conditional.revert(rv, model)
        corrected = impl.conditional.apply(data, conditional)
        logpdf = impl.stats.logpdf(data, observed)
        return _KFState(corrected, 1.0, logpdf)

    def _step(state: _KFState, cond_and_data_and_obs) -> tuple[_KFState, _KFState]:
        conditional, data, observation = cond_and_data_and_obs
        rv, num_data, logpdf = state

        # Extrapolate-correct
        rv = impl.conditional.marginalise(rv, conditional)
        observed, reverse = impl.conditional.revert(rv, observation)
        corrected = impl.conditional.apply(data, reverse)

        # Update logpdf
        logpdf_new = impl.stats.logpdf(data, observed)
        logpdf_mean = (logpdf * num_data + logpdf_new) / (num_data + 1)
        state = _KFState(corrected, num_data + 1.0, logpdf_mean)

        # Scan-compatible output
        return state, state

    return _initialise, _step
