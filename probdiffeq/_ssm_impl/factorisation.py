from probdiffeq._ssm_util import ssm_api
from probdiffeq.backend import structs

__all__ = ["FactSsmImpl"]


@structs.dataclass
class FactSsmImpl:
    """Implementation of factorized Markovian state-space models."""

    linearize: ssm_api.AbstractLinearizationFactory
    """An implementation of linearization constructors."""

    prior: ssm_api.AbstractPriorFactory
    """An implementation of constructing prior distributions."""

    conditional: ssm_api.AbstractConditional
    """An implementation of manipulating conditionals."""

    @classmethod
    def from_dense(cls, /):
        from probdiffeq._ssm_util import dense_models

        prior = dense_models.DensePriorFactory()
        linearize = dense_models.DenseLinearizationFactory()
        conditional = dense_models.DenseConditional()
        return cls(linearize=linearize, conditional=conditional, prior=prior)

    @classmethod
    def from_isotropic(cls, /):
        from probdiffeq._ssm_util import isotropic_models

        prior = isotropic_models.IsotropicPriorFactory()
        linearize = isotropic_models.IsotropicLinearizationFactory()
        conditional = isotropic_models.IsotropicConditional()
        return cls(linearize=linearize, conditional=conditional, prior=prior)

    @classmethod
    def from_blockdiag(cls, /):
        from probdiffeq._ssm_util import blockdiag_models

        prior = blockdiag_models.BlockDiagPriorFactory()
        linearize = blockdiag_models.BlockDiagLinearizationFactory()
        conditional = blockdiag_models.BlockDiagConditional()
        return cls(linearize=linearize, conditional=conditional, prior=prior)
