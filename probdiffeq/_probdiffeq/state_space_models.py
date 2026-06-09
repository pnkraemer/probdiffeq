"""State-space model implementations.

Examples
--------
>>> from probdiffeq import probdiffeq

>>> from probdiffeq import probdiffeq

>>> ssm = probdiffeq.state_space_model()
>>> print(ssm)
FactSsmImpl(linearize=DenseLinearizationFactory(), prior=DensePriorFactory())

Switching to a different factorisation is easy via the `ssm_fact` argument:

>>> ssm = probdiffeq.state_space_model(probdiffeq.SsmFactName.BLOCKDIAG)
>>> print(ssm)
FactSsmImpl(linearize=BlockDiagLinearizationFactory(), prior=BlockDiagPriorFactory())

Instead of the Enum, you can also just pass the string:

>>> ssm = probdiffeq.state_space_model("isotropic")
>>> print(ssm)
FactSsmImpl(linearize=IsotropicLinearizationFactory(), prior=IsotropicPriorFactory())

"""

from probdiffeq import ssm_impl
from probdiffeq.backend import structs


class SsmFactName(structs.Enum):
    """The factorisation of the state-space model to use."""

    DENSE = "dense"
    """Indicate a dense factorisation of the state-space model.

    This is the most general factorisation, but also the most computationally expensive.
    Since all algorithms work for this factorisation, it is the default choice.
    """

    ISOTROPIC = "isotropic"
    """Indicate an isotropic factorisation of the state-space model.

    This is the least general factorisation, but also the least computationally expensive.
    Some functionality (e.g. DAE solvers) is not available for this factorisation,
    but it tends to e the most efficient choice when it is applicable.
    """

    BLOCKDIAG = "blockdiag"
    """Indicate a block-diagonal factorisation of the state-space model.

    This is a middle ground between the dense and isotropic factorisations, and can be more computationally efficient than the dense factorisation while still being more general than the isotropic factorisation.
    """


def state_space_model(
    ssm_fact: SsmFactName | str = SsmFactName.DENSE,
) -> ssm_impl.FactSsmImpl:
    """Construct an implementation of a factorised state-space model.

    Parameters
    ----------
    ssm_fact:
        Either an `SsmFactName` or one of:
        `"dense"`, `"blockdiag"`, `"isotropic"`.
    """
    ssm_fact = SsmFactName(ssm_fact)

    if ssm_fact == SsmFactName.DENSE:
        return ssm_impl.FactSsmImpl.from_dense()

    if ssm_fact == SsmFactName.BLOCKDIAG:
        return ssm_impl.FactSsmImpl.from_blockdiag()

    if ssm_fact == SsmFactName.ISOTROPIC:
        return ssm_impl.FactSsmImpl.from_isotropic()
    raise ValueError(ssm_fact)
