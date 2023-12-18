"""Correction models."""

from probdiffeq.backend import abc, functools
from probdiffeq.backend import numpy as np
from probdiffeq.impl import impl
from probdiffeq.solvers.strategies.components import cubature


class Correction(abc.ABC):
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


class _ODEConstraintTaylor(Correction):
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


class _ODEConstraintStatistical(Correction):
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
    cubature_fun = cubature_fun or cubature.third_order_spherical
    linearise_fun = impl.linearise.ode_statistical_1st(cubature_fun)
    return _ODEConstraintStatistical(
        ode_order=1,
        linearise_fun=linearise_fun,
        string_repr=f"<SLR1 with ode_order={1}>",
    )


def slr1(cubature_fun=None) -> _ODEConstraintStatistical:
    """First-order statistical linear regression."""
    cubature_fun = cubature_fun or cubature.third_order_spherical
    linearise_fun = impl.linearise.ode_statistical_0th(cubature_fun)
    return _ODEConstraintStatistical(
        ode_order=1,
        linearise_fun=linearise_fun,
        string_repr=f"<SLR0 with ode_order={1}>",
    )
