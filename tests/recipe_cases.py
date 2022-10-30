"""Test cases: Solver recipes.

All ODE test problems will be two-dimensional.
"""
from pytest_cases import case

from odefilter import recipes


@case(tags=["terminal_value", "solve", "checkpoint", "filter"])
def solver_ekf0_isotropic():
    return recipes.ekf0_isotropic(num_derivatives=3)


@case(tags=["terminal_value", "solve", "checkpoint", "filter"])
def solver_ekf0_isotropic_dynamic():
    return recipes.ekf0_isotropic_dynamic(num_derivatives=3)


@case(tags=["terminal_value", "solve", "smoother"])
def solver_eks0_isotropic():
    return recipes.eks0_isotropic(num_derivatives=3)


@case(tags=["terminal_value", "solve", "smoother"])
def solver_eks0_isotropic_dynamic():
    return recipes.eks0_isotropic_dynamic(num_derivatives=3)


@case(tags=["terminal_value", "checkpoint", "smoother"])
def solver_eks0_isotropic_fixedpoint():
    return recipes.eks0_isotropic_fixedpoint(num_derivatives=3)


@case(tags=["terminal_value", "checkpoint", "smoother"])
def solver_eks0_isotropic_dynamic_fixedpoint():
    return recipes.eks0_isotropic_dynamic_fixedpoint(num_derivatives=3)


@case(tags=["terminal_value", "solve", "checkpoint", "filter"])
def solver_ekf0_batch():
    return recipes.ekf0_batch(num_derivatives=3, ode_dimension=2)


@case(tags=["terminal_value", "solve", "checkpoint", "filter"])
def solver_ekf0_batch_dynamic():
    return recipes.ekf0_batch_dynamic(ode_dimension=2, num_derivatives=3)


@case(tags=["terminal_value", "solve", "smoother"])
def solver_eks0_batch():
    return recipes.eks0_batch(ode_dimension=2, num_derivatives=3)


@case(tags=["terminal_value", "solve", "smoother"])
def solver_eks0_batch_dynamic():
    return recipes.eks0_batch_dynamic(ode_dimension=2, num_derivatives=3)


@case(tags=["terminal_value", "checkpoint", "smoother"])
def solver_eks0_batch_fixedpoint():
    return recipes.eks0_batch_fixedpoint(ode_dimension=2, num_derivatives=3)


@case(tags=["terminal_value", "checkpoint", "smoother"])
def solver_eks0_batch_dynamic_fixedpoint():
    return recipes.eks0_batch_dynamic_fixedpoint(num_derivatives=3, ode_dimension=2)


@case(tags=["terminal_value", "solve", "checkpoint", "filter"])
def solver_ekf1():
    return recipes.ekf1(num_derivatives=3, ode_dimension=2)


@case(tags=["terminal_value", "solve", "checkpoint", "filter"])
def solver_ckf1():
    return recipes.ckf1(num_derivatives=3, ode_dimension=2)


@case(tags=["terminal_value", "solve", "checkpoint", "filter"])
def solver_ukf1():
    return recipes.ukf1(num_derivatives=3, ode_dimension=2)


@case(tags=["terminal_value", "solve", "checkpoint", "filter"])
def solver_ekf1_dynamic():
    return recipes.ekf1_dynamic(num_derivatives=3, ode_dimension=2)


@case(tags=["terminal_value", "solve", "smoother"])
def solver_eks1():
    return recipes.eks1(num_derivatives=3, ode_dimension=2)


@case(tags=["terminal_value", "solve", "smoother"])
def solver_eks1_dynamic():
    return recipes.eks1_dynamic(num_derivatives=3, ode_dimension=2)


@case(tags=["terminal_value", "checkpoint", "smoother"])
def solver_eks1_fixedpoint():
    return recipes.eks1_fixedpoint(num_derivatives=3, ode_dimension=2)


@case(tags=["terminal_value", "checkpoint", "smoother"])
def solver_eks1_dynamic_fixedpoint():
    return recipes.eks1_dynamic_fixedpoint(num_derivatives=3, ode_dimension=2)
