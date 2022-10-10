"""Tests for recipes."""

import pytest_cases

from odefilter import recipes, solvers

# They all use PIControl and TaylorMode, unless specified otherwise
#
# isotropic_ekf0(num_derivatives)
# isotropic_eks0(num_derivatives)
# batch_ekf0(num_derivatives, ode_dimension)
# batch_eks0(num_derivatives, ode_dimension)
# batch_ekf1(num_derivatives, ode_dimension)
# batch_eks1(num_derivatives, ode_dimension)
# batch_ukf1(num_derivatives, ode_dimension)
# batch_uks1(num_derivatives, ode_dimension)
# ekf1(num_derivatives, ode_dimension)
# eks1(num_derivatives, ode_dimension)
# ukf1(num_derivatives, ode_dimension)
# uks1(num_derivatives, ode_dimension)
#
# dynamic_isotropic_ekf0(num_derivatives)
# dynamic_isotropic_eks0(num_derivatives)
# dynamic_batch_ekf0(num_derivatives, ode_dimension)
# dynamic_batch_eks0(num_derivatives, ode_dimension)
# dynamic_batch_ekf1(num_derivatives, ode_dimension)
# dynamic_batch_eks1(num_derivatives, ode_dimension)
# dynamic_batch_ukf1(num_derivatives, ode_dimension)
# dynamic_batch_uks1(num_derivatives, ode_dimension)
# dynamic_ekf1(num_derivatives, ode_dimension)
# dynamic_eks1(num_derivatives, ode_dimension)
# dynamic_ukf1(num_derivatives, ode_dimension)
# dynamic_uks1(num_derivatives, ode_dimension)


def case_isotropic_ekf0():
    return recipes.dynamic_isotropic_ekf0(num_derivatives=3, atol=1e-3, rtol=1e-6)


def case_isotropic_eks0():
    return recipes.dynamic_isotropic_eks0(num_derivatives=3, atol=1e-3, rtol=1e-6)


def case_dynamic_ekf1():
    irrelevant = 1
    return recipes.dynamic_ekf1(
        num_derivatives=3, ode_dimension=irrelevant, atol=1e-3, rtol=1e-6
    )


@pytest_cases.parametrize_with_cases("solver", cases=".")
def test_is_solver(solver):
    assert isinstance(solver, solvers.Adaptive)
