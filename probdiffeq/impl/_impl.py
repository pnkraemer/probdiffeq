"""State-space model impl."""

import warnings

from probdiffeq.backend import containers
from probdiffeq.backend.typing import Optional
from probdiffeq.impl import (
    _conditional,
    _linearise,
    _normal,
    _prototypes,
    _stats,
    _transform,
)


class Impl:
    """User-facing implementation 'package'.

    Wrap a factorised implementations and garnish it with error messages
    and a `select()` functionality.
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
        if self._fact is not None:
            return self._fact.linearise

        raise ValueError(self.error_msg())

    @property
    def conditional(self) -> _conditional.ConditionalBackend:
        if self._fact is not None:
            return self._fact.conditional
        raise ValueError(self.error_msg())

    @property
    def transform(self) -> _transform.TransformBackend:
        if self._fact is not None:
            return self._fact.transform
        raise ValueError(self.error_msg())

    @property
    def normal(self) -> _normal.NormalBackend:
        if self._fact is not None:
            return self._fact.ssm_util
        raise ValueError(self.error_msg())

    @property
    def prototypes(self) -> _prototypes.PrototypeBackend:
        if self._fact is not None:
            return self._fact.prototypes
        raise ValueError(self.error_msg())

    @property
    def stats(self) -> _stats.StatsBackend:
        if self._fact is not None:
            return self._fact.stats
        raise ValueError(self.error_msg())

    @staticmethod
    def error_msg():
        return "Select a factorisation first."


@containers.dataclass
class FactorisedImpl:
    prototypes: _prototypes.PrototypeBackend
    ssm_util: _normal.NormalBackend
    stats: _stats.StatsBackend
    linearise: _linearise.LinearisationBackend
    conditional: _conditional.ConditionalBackend
    transform: _transform.TransformBackend


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


def _select_scalar() -> FactorisedImpl:
    prototypes = _prototypes.ScalarPrototype()
    ssm_util = _normal.ScalarNormal()
    stats = _stats.ScalarStats()
    linearise = _linearise.ScalarLinearisation()
    conditional = _conditional.ScalarConditional()
    transform = _transform.ScalarTransform()
    return FactorisedImpl(
        prototypes=prototypes,
        ssm_util=ssm_util,
        stats=stats,
        linearise=linearise,
        conditional=conditional,
        transform=transform,
    )


def _select_dense(*, ode_shape) -> FactorisedImpl:
    prototypes = _prototypes.DensePrototype(ode_shape=ode_shape)
    ssm_util = _normal.DenseNormal(ode_shape=ode_shape)
    linearise = _linearise.DenseLinearisation(ode_shape=ode_shape)
    stats = _stats.DenseStats(ode_shape=ode_shape)
    conditional = _conditional.DenseConditional(ode_shape=ode_shape)
    transform = _transform.DenseTransform()
    return FactorisedImpl(
        linearise=linearise,
        transform=transform,
        conditional=conditional,
        ssm_util=ssm_util,
        prototypes=prototypes,
        stats=stats,
    )


def _select_isotropic(*, ode_shape) -> FactorisedImpl:
    prototypes = _prototypes.IsotropicPrototype(ode_shape=ode_shape)
    ssm_util = _normal.IsotropicNormal(ode_shape=ode_shape)
    stats = _stats.IsotropicStats(ode_shape=ode_shape)
    linearise = _linearise.IsotropicLinearisation()
    conditional = _conditional.IsotropicConditional(ode_shape=ode_shape)
    transform = _transform.IsotropicTransform()
    return FactorisedImpl(
        prototypes=prototypes,
        ssm_util=ssm_util,
        stats=stats,
        linearise=linearise,
        conditional=conditional,
        transform=transform,
    )


def _select_blockdiag(*, ode_shape) -> FactorisedImpl:
    prototypes = _prototypes.BlockDiagPrototype(ode_shape=ode_shape)
    ssm_util = _normal.BlockDiagNormal(ode_shape=ode_shape)
    stats = _stats.BlockDiagStats(ode_shape=ode_shape)
    linearise = _linearise.BlockDiagLinearisation()
    conditional = _conditional.BlockDiagConditional(ode_shape=ode_shape)
    transform = _transform.BlockDiagTransform(ode_shape=ode_shape)
    return FactorisedImpl(
        prototypes=prototypes,
        ssm_util=ssm_util,
        stats=stats,
        linearise=linearise,
        conditional=conditional,
        transform=transform,
    )
