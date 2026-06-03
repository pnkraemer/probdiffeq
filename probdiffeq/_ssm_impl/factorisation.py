from probdiffeq._ssm_impl import interfaces
from probdiffeq.backend import structs

__all__ = ["FactSsmImpl"]


@structs.dataclass
class FactSsmImpl:
    """Implementation of factorized Markovian state-space models."""

    linearize: interfaces.AbstractLinearizationFactory
    """An implementation of linearization constructors."""

    prior: interfaces.AbstractPriorFactory
    """An implementation of constructing prior distributions."""

    conditional: interfaces.AbstractConditional
    """An implementation of manipulating conditionals."""

    @classmethod
    def from_dense(cls, /):
        from probdiffeq._ssm_impl import factorisation_via_dense as dense

        prior = dense.DensePriorFactory()
        linearize = dense.DenseLinearizationFactory()
        conditional = dense.DenseConditional()
        return cls(linearize=linearize, conditional=conditional, prior=prior)

    @classmethod
    def from_isotropic(cls, /):
        from probdiffeq._ssm_impl import factorisation_via_isotropic as isotropic

        prior = isotropic.IsotropicPriorFactory()
        linearize = isotropic.IsotropicLinearizationFactory()
        conditional = isotropic.IsotropicConditional()
        return cls(linearize=linearize, conditional=conditional, prior=prior)

    @classmethod
    def from_blockdiag(cls, /):
        from probdiffeq._ssm_impl import factorisation_via_blockdiag as blockdiag

        prior = blockdiag.BlockDiagPriorFactory()
        linearize = blockdiag.BlockDiagLinearizationFactory()
        conditional = blockdiag.BlockDiagConditional()
        return cls(linearize=linearize, conditional=conditional, prior=prior)
