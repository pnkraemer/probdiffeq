"""Interface for estimation strategies."""

from probdiffeq import _interp
from probdiffeq.backend import abc, containers, tree_util
from probdiffeq.backend import numpy as np
from probdiffeq.backend.typing import Any, Generic, TypeVar
from probdiffeq.impl import impl

T = TypeVar("T")
R = TypeVar("R")
S = TypeVar("S")


class ExtrapolationImpl(abc.ABC, Generic[T, R, S]):
    """Extrapolation model interface."""

    @abc.abstractmethod
    def initial_condition(self, tcoeffs, /) -> T:
        """Compute an initial condition from a set of Taylor coefficients."""
        raise NotImplementedError

    @abc.abstractmethod
    def init(self, solution: T, /) -> tuple[R, S]:
        """Initialise a state from a solution."""
        raise NotImplementedError

    @abc.abstractmethod
    def begin(self, state: R, aux: S, /, dt) -> tuple[R, S]:
        """Begin the extrapolation."""
        raise NotImplementedError

    @abc.abstractmethod
    def complete(self, state: R, aux: S, /, output_scale) -> tuple[R, S]:
        """Complete the extrapolation."""
        raise NotImplementedError

    @abc.abstractmethod
    def extract(self, state: R, aux: S, /) -> T:
        """Extract a solution from a state."""
        raise NotImplementedError

    @abc.abstractmethod
    def interpolate(self, state_t0, marginal_t1, *, dt0, dt1, output_scale):
        """Interpolate."""
        raise NotImplementedError

    @abc.abstractmethod
    def right_corner(self, state: R, aux: S, /) -> _interp.InterpRes[tuple[R, S]]:
        """Process the state at checkpoint t=t_n."""
        raise NotImplementedError


class _State(containers.NamedTuple):
    t: Any
    hidden: Any
    aux_extra: Any
    aux_corr: Any


class Strategy:
    """Estimation strategy."""

    def __init__(
        self,
        extrapolation: ExtrapolationImpl,
        correction,
        *,
        string_repr,
        is_suitable_for_save_at,
        is_suitable_for_save_every_step,
        is_suitable_for_offgrid_marginals,
    ):
        # Content
        self.extrapolation = extrapolation
        self.correction = correction

        # Some meta-information
        self.string_repr = string_repr
        self.is_suitable_for_save_at = is_suitable_for_save_at
        self.is_suitable_for_save_every_step = is_suitable_for_save_every_step
        self.is_suitable_for_offgrid_marginals = is_suitable_for_offgrid_marginals

    def __repr__(self):
        return self.string_repr

    def initial_condition(self, taylor_coefficients, /):
        """Construct an initial condition from a set of Taylor coefficients."""
        return self.extrapolation.initial_condition(taylor_coefficients)

    def init(self, t, posterior, /) -> _State:
        """Initialise a state from a posterior."""
        rv, extra = self.extrapolation.init(posterior)
        rv, corr = self.correction.init(rv)
        return _State(t=t, hidden=rv, aux_extra=extra, aux_corr=corr)

    def predict_error(self, state: _State, /, *, dt, vector_field):
        """Predict the error of an upcoming step."""
        hidden, extra = self.extrapolation.begin(state.hidden, state.aux_extra, dt=dt)
        t = state.t + dt
        error, observed, corr = self.correction.estimate_error(
            hidden, state.aux_corr, vector_field=vector_field, t=t
        )
        state = _State(t=t, hidden=hidden, aux_extra=extra, aux_corr=corr)
        return error, observed, state

    def complete(self, state, /, *, output_scale):
        """Complete the step after the error has been predicted."""
        hidden, extra = self.extrapolation.complete(
            state.hidden, state.aux_extra, output_scale=output_scale
        )
        hidden, corr = self.correction.complete(hidden, state.aux_corr)
        return _State(t=state.t, hidden=hidden, aux_extra=extra, aux_corr=corr)

    def extract(self, state: _State, /):
        """Extract the solution from a state."""
        hidden = self.correction.extract(state.hidden, state.aux_corr)
        sol = self.extrapolation.extract(hidden, state.aux_extra)
        return state.t, sol

    def case_right_corner(self, state_t1: _State) -> _interp.InterpRes:
        """Process the solution in case t=t_n."""
        _tmp = self.extrapolation.right_corner(state_t1.hidden, state_t1.aux_extra)
        step_from, solution, interp_from = _tmp

        def _state(x):
            t = state_t1.t
            corr_like = tree_util.tree_map(np.empty_like, state_t1.aux_corr)
            return _State(t=t, hidden=x[0], aux_extra=x[1], aux_corr=corr_like)

        step_from = _state(step_from)
        solution = _state(solution)
        interp_from = _state(interp_from)
        return _interp.InterpRes(step_from, solution, interp_from)

    def case_interpolate(
        self, t, *, s0: _State, s1: _State, output_scale
    ) -> _interp.InterpRes[_State]:
        """Process the solution in case t>t_n."""
        # Interpolate
        step_from, solution, interp_from = self.extrapolation.interpolate(
            state_t0=(s0.hidden, s0.aux_extra),
            marginal_t1=s1.hidden,
            dt0=t - s0.t,
            dt1=s1.t - t,
            output_scale=output_scale,
        )

        # Turn outputs into valid states

        def _state(t_, x):
            corr_like = tree_util.tree_map(np.empty_like, s0.aux_corr)
            return _State(t=t_, hidden=x[0], aux_extra=x[1], aux_corr=corr_like)

        step_from = _state(s1.t, step_from)
        solution = _state(t, solution)
        interp_from = _state(t, interp_from)
        return _interp.InterpRes(step_from, solution, interp_from)

    def offgrid_marginals(self, *, t, marginals_t1, posterior_t0, t0, t1, output_scale):
        """Compute offgrid_marginals."""
        if not self.is_suitable_for_offgrid_marginals:
            raise NotImplementedError

        dt0 = t - t0
        dt1 = t1 - t
        state_t0 = self.init(t0, posterior_t0)

        _acc, (marginals, _aux), _prev = self.extrapolation.interpolate(
            state_t0=(state_t0.hidden, state_t0.aux_extra),
            marginal_t1=marginals_t1,
            dt0=dt0,
            dt1=dt1,
            output_scale=output_scale,
        )

        u = impl.hidden_model.qoi(marginals)
        return u, marginals


def _tree_flatten(strategy):
    children = ()
    aux = (
        # Content
        strategy.extrapolation,
        strategy.correction,
        # Meta-info
        strategy.string_repr,
        strategy.is_suitable_for_offgrid_marginals,
        strategy.is_suitable_for_save_every_step,
        strategy.is_suitable_for_save_at,
    )
    return children, aux


def _tree_unflatten(aux, _children):
    extra, corr, string, suitable_offgrid, suitable_every, suitable_saveat = aux
    return Strategy(
        extrapolation=extra,
        correction=corr,
        string_repr=string,
        is_suitable_for_save_at=suitable_saveat,
        is_suitable_for_save_every_step=suitable_every,
        is_suitable_for_offgrid_marginals=suitable_offgrid,
    )


tree_util.register_pytree_node(Strategy, _tree_flatten, _tree_unflatten)
