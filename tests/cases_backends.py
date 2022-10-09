"""Cases."""

import pytest_cases

from odefilter import backends, implementations, information


@pytest_cases.parametrize(
    "information_op",
    [information.IsotropicEK0FirstOrder()],
    ids=["IsotropicEK0FirstOrder"],
)
@pytest_cases.parametrize("num_derivatives", [2])
def case_backend_dynamic_isotropic_filter(num_derivatives, information_op):
    return backends.DynamicFilter(
        implementation=implementations.IsotropicImplementation.from_num_derivatives(
            num_derivatives=num_derivatives
        ),
        information=information_op,
    )


@pytest_cases.parametrize(
    "information_op",
    [information.IsotropicEK0FirstOrder()],
    ids=["IsotropicEK0FirstOrder"],
)
@pytest_cases.parametrize("num_derivatives", [2])
def case_backend_dynamic_smoother(num_derivatives, information_op):
    return backends.DynamicSmoother(
        implementation=implementations.IsotropicImplementation.from_num_derivatives(
            num_derivatives=num_derivatives
        ),
        information=information_op,
    )


@pytest_cases.parametrize("information_op", [information.EK1(ode_dimension=1)])
@pytest_cases.parametrize("num_derivatives", [2])
def case_backend_dynamic_filter(num_derivatives, information_op):
    return backends.DynamicFilter(
        implementation=implementations.DenseImplementation.from_num_derivatives(
            num_derivatives=num_derivatives, ode_dimension=1
        ),
        information=information_op,
    )
