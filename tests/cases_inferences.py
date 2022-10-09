"""Cases."""

import pytest_cases

from odefilter import implementations, inferences, information


@pytest_cases.parametrize(
    "information_op",
    [information.IsotropicEK0FirstOrder()],
    ids=["IsotropicEK0"],
)
@pytest_cases.parametrize("num_derivatives", [2])
def case_dynamic_isotropic_filter(num_derivatives, information_op):
    return inferences.DynamicFilter(
        implementation=implementations.IsotropicImplementation.from_num_derivatives(
            num_derivatives=num_derivatives
        ),
        information=information_op,
    )


@pytest_cases.parametrize(
    "information_op",
    [information.IsotropicEK0FirstOrder()],
    ids=["IsotropicEK0"],
)
@pytest_cases.parametrize("num_derivatives", [2])
def case_dynamic_smoother(num_derivatives, information_op):
    return inferences.DynamicSmoother(
        implementation=implementations.IsotropicImplementation.from_num_derivatives(
            num_derivatives=num_derivatives
        ),
        information=information_op,
    )


@pytest_cases.parametrize(
    "information_op", [information.EK1(ode_dimension=2)], ids=["EK1"]
)
@pytest_cases.parametrize("num_derivatives", [2])
def case_dynamic_filter(num_derivatives, information_op):
    return inferences.DynamicFilter(
        implementation=implementations.DenseImplementation.from_num_derivatives(
            num_derivatives=num_derivatives, ode_dimension=2
        ),
        information=information_op,
    )
