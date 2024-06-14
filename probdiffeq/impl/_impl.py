"""State-space model impl."""

import warnings

from probdiffeq.backend import containers
from probdiffeq.backend.typing import Optional
from probdiffeq.impl import (
    _conditional,
    _hidden_model,
    _linearise,
    _prototypes,
    _ssm_util,
    _stats,
    _transform,
    _variable,
)


class Impl:
    """User-facing implementation 'package'.

    Wrap a factorised implementations and garnish it with error messages
    and a "selection" functionality.
    """

    def __init__(self) -> None:
        self._fact: Optional[FactorisedImpl] = None
        self._fact_name: str = "None"

    def select(self, which, **kwargs):
        if self._fact is not None:
            msg = f"An implementation has already been selected: '{self._fact_name}'."
            warnings.warn(msg, stacklevel=1)
        self._fact = choose(which, **kwargs)
        self._fact_name = which

    @property
    def impl_name(self):
        return self._fact_name

    @property
    def linearise(self) -> _linearise.LinearisationBackend:
        if self._fact is None:
            msg = "Select a factorisation first."
            raise ValueError(msg)
        return self._fact.linearise

    @property
    def conditional(self) -> _conditional.ConditionalBackend:
        if self._fact is None:
            msg = "Select a factorisation first."
            raise ValueError(msg)
        return self._fact.conditional

    @property
    def transform(self) -> _transform.TransformBackend:
        if self._fact is None:
            msg = "Select a factorisation first."
            raise ValueError(msg)
        return self._fact.transform

    @property
    def ssm_util(self) -> _ssm_util.SSMUtilBackend:
        if self._fact is None:
            msg = "Select a factorisation first."
            raise ValueError(msg)
        return self._fact.ssm_util

    @property
    def prototypes(self) -> _prototypes.PrototypeBackend:
        if self._fact is None:
            msg = "Select a factorisation first."
            raise ValueError(msg)
        return self._fact.prototypes

    @property
    def variable(self) -> _variable.VariableBackend:
        if self._fact is None:
            msg = "Select a factorisation first."
            raise ValueError(msg)
        return self._fact.variable

    @property
    def hidden_model(self) -> _hidden_model.HiddenModelBackend:
        if self._fact is None:
            msg = "Select a factorisation first."
            raise ValueError(msg)
        return self._fact.hidden_model

    @property
    def stats(self) -> _stats.StatsBackend:
        if self._fact is None:
            msg = "Select a factorisation first."
            raise ValueError(msg)
        return self._fact.stats


@containers.dataclass
class FactorisedImpl:
    linearise: _linearise.LinearisationBackend
    transform: _transform.TransformBackend
    conditional: _conditional.ConditionalBackend
    ssm_util: _ssm_util.SSMUtilBackend
    prototypes: _prototypes.PrototypeBackend
    variable: _variable.VariableBackend
    hidden_model: _hidden_model.HiddenModelBackend
    stats: _stats.StatsBackend


def choose(which: str, /, *, ode_shape=None) -> FactorisedImpl:
    if which == "scalar":
        return _select_scalar()

    if ode_shape is None:
        msg = "Please provide an ODE shape."
        raise ValueError(msg)

    if which == "dense":
        return _select_dense(ode_shape=ode_shape)
    if which == "isotropic":
        return _select_isotropic(ode_shape=ode_shape)
    if which == "blockdiag":
        return _select_blockdiag(ode_shape=ode_shape)

    msg1 = f"Implementation '{which}' unknown. "
    msg2 = "Choose an implementation out of {scalar, dense, isotropic, blockdiag}."
    raise ValueError(msg1 + msg2)


def _select_dense(*, ode_shape):
    prototypes = _prototypes.DensePrototype(ode_shape=ode_shape)
    ssm_util = _ssm_util.DenseSSMUtil(ode_shape=ode_shape)
    linearise = _linearise.DenseLinearisation(ode_shape=ode_shape)
    stats = _stats.DenseStats(ode_shape=ode_shape)
    conditional = _conditional.DenseConditional()
    transform = _transform.DenseTransform()
    variable = _variable.DenseVariable(ode_shape=ode_shape)
    hidden_model = _hidden_model.DenseHiddenModel(ode_shape=ode_shape)
    return FactorisedImpl(
        linearise=linearise,
        transform=transform,
        conditional=conditional,
        ssm_util=ssm_util,
        prototypes=prototypes,
        variable=variable,
        hidden_model=hidden_model,
        stats=stats,
    )
