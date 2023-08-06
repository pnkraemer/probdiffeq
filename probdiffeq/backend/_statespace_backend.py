"""State-space model backend."""

from probdiffeq.backend import _cond, _factorisations, _linearise, _random, _ssm_util


class BackEnd:
    def __init__(self):
        self._fact: _factorisations.Factorisation = None

    def select(self, which, **kwargs):
        self._fact = _factorisations.choose(which, **kwargs)

    @property
    def linearise_ode(self) -> _linearise.LineariseODEBackEnd:
        return self._fact.linearise_ode()

    @property
    def random(self) -> _random.RandomVariableBackEnd:
        if self._fact is None:
            raise ValueError("Select a factorisation first.")
        return self._fact.random()

    @property
    def cond(self) -> _cond.ConditionalBackEnd:
        return self._fact.cond()

    @property
    def ssm_util(self) -> _ssm_util.SSMUtilBackEnd:
        return self._fact.ssm_util()
