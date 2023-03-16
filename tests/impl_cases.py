"""Test cases for implementations."""

import pytest_cases

from probdiffeq.implementations import recipes


@pytest_cases.case(id="IsoTS0")
def case_ts0_iso():
    def impl_factory(*, num_derivatives, ode_shape):
        return recipes.IsoTS0.from_params(num_derivatives=num_derivatives)

    return impl_factory


@pytest_cases.case(id="BlockDiagTS0")
def case_ts0_blockdiag():
    return recipes.BlockDiagTS0.from_params


@pytest_cases.case(id="DenseTS1")
def case_ts1_dense():
    return recipes.DenseTS1.from_params


@pytest_cases.case(id="DenseTS0")
def case_ts0_dense():
    return recipes.DenseTS0.from_params


# todo: parametrize with different cubature rules
@pytest_cases.case(id="DenseSLR1")
def case_slr1_dense(cubature_rule=None):
    def impl_factory(**kwargs):
        return recipes.DenseSLR1.from_params(cubature_rule=cubature_rule, **kwargs)

    return impl_factory


# todo: parametrize with different cubature rules
@pytest_cases.case(id="DenseSLR0")
def case_slr0_dense(cubature_rule=None):
    def impl_factory(**kwargs):
        return recipes.DenseSLR0.from_params(cubature_rule=cubature_rule, **kwargs)

    return impl_factory


# todo: parametrize with different cubature rules
@pytest_cases.case(id="BlockDiagSLR1")
def case_slr1_blockdiag(cubature_rule=None):
    def impl_factory(**kwargs):
        return recipes.BlockDiagSLR1.from_params(cubature_rule=cubature_rule, **kwargs)

    return impl_factory
