"""Test cases: Solver recipes.

All ODE test problems will be two-dimensional.
"""
from pytest_cases import case

from odefilter import recipes


@case(tags=["terminal_value", "checkpoint", "smoother"])
def solver_dynamic_isotropic_fixedpt_eks0():
    return recipes.dynamic_isotropic_fixedpt_eks0(num_derivatives=3)


@case(tags=["terminal_value", "solve", "smoother"])
def solver_dynamic_isotropic_eks0():
    return recipes.dynamic_isotropic_eks0(num_derivatives=3)


@case(tags=["terminal_value", "solve", "checkpoint", "filter"])
def solver_dynamic_isotropic_ekf0():
    return recipes.dynamic_isotropic_ekf0(num_derivatives=3)


@case(tags=["terminal_value", "solve", "checkpoint", "filter"])
def solver_isotropic_ekf0():
    return recipes.isotropic_ekf0(num_derivatives=3)


@case(tags=["terminal_value", "solve", "checkpoint", "filter"])
def solver_dynamic_ekf1():
    return recipes.dynamic_ekf1(num_derivatives=3, ode_dimension=2)


@case(tags=["terminal_value", "solve", "checkpoint", "filter"])
def solver_ekf1():
    return recipes.ekf1(num_derivatives=3, ode_dimension=2)
