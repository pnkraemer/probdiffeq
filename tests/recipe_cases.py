"""Test cases: Solver recipes.

All ODE test problems will be two-dimensional.
"""
from pytest_cases import case, parametrize

from odefilter import recipes


@case(tags=["terminal_value", "solve", "checkpoint", "filter"])
@parametrize("calibration", ["dynamic", "mle"])
@parametrize("n", [3])
def solver_ekf0_isotropic(calibration, n):
    return recipes.ekf0_isotropic(num_derivatives=n, calibration=calibration)


@case(tags=["terminal_value", "solve", "smoother"])
@parametrize("calibration", ["dynamic", "mle"])
@parametrize("n", [3])
def solver_eks0_isotropic(calibration, n):
    return recipes.eks0_isotropic(num_derivatives=n, calibration=calibration)


@case(tags=["terminal_value", "checkpoint", "smoother"])
@parametrize("calibration", ["dynamic", "mle"])
@parametrize("n", [3])
def solver_eks0_isotropic_fixedpoint(calibration, n):
    return recipes.eks0_isotropic_fixedpoint(num_derivatives=n, calibration=calibration)


@case(tags=["terminal_value", "solve", "checkpoint", "filter"])
@parametrize("calibration", ["dynamic", "mle"])
@parametrize("n", [3])
def solver_ekf0_batch(calibration, n):
    return recipes.ekf0_batch(
        num_derivatives=n, calibration=calibration, ode_dimension=2
    )


@case(tags=["terminal_value", "solve", "smoother"])
@parametrize("calibration", ["dynamic", "mle"])
@parametrize("n", [3])
def solver_eks0_batch(calibration, n):
    return recipes.eks0_batch(
        ode_dimension=2, num_derivatives=n, calibration=calibration
    )


@case(tags=["terminal_value", "checkpoint", "smoother"])
@parametrize("calibration", ["dynamic", "mle"])
@parametrize("n", [3])
def solver_eks0_batch_fixedpoint(calibration, n):
    return recipes.eks0_batch_fixedpoint(
        ode_dimension=2, num_derivatives=n, calibration=calibration
    )


@case(tags=["terminal_value", "solve", "checkpoint", "filter"])
@parametrize("calibration", ["dynamic", "mle"])
@parametrize("n", [3])
def solver_ekf1(calibration, n):
    return recipes.ekf1(ode_dimension=2, num_derivatives=n, calibration=calibration)


@case(tags=["terminal_value", "solve", "checkpoint", "filter"])
@parametrize("calibration", ["dynamic", "mle"])
@parametrize("n", [3])
def solver_ckf1(calibration, n):
    return recipes.ckf1(ode_dimension=2, num_derivatives=n, calibration=calibration)


@case(tags=["terminal_value", "solve", "checkpoint", "filter"])
@parametrize("calibration", ["dynamic", "mle"])
@parametrize("n", [3])
def solver_ukf1(calibration, n):
    return recipes.ukf1(ode_dimension=2, num_derivatives=n, calibration=calibration)


@case(tags=["terminal_value", "solve", "checkpoint", "filter"])
@parametrize("calibration", ["dynamic", "mle"])
@parametrize("n", [3])
@parametrize("degree", [5])
def solver_ghkf1(calibration, n, degree):
    return recipes.ghkf1(
        ode_dimension=2, num_derivatives=n, calibration=calibration, degree=degree
    )


@case(tags=["terminal_value", "solve", "smoother"])
@parametrize("calibration", ["dynamic", "mle"])
@parametrize("n", [3])
def solver_eks1(calibration, n):
    return recipes.eks1(ode_dimension=2, num_derivatives=n, calibration=calibration)


@case(tags=["terminal_value", "checkpoint", "smoother"])
@parametrize("calibration", ["dynamic", "mle"])
@parametrize("n", [3])
def solver_eks1_fixedpoint(calibration, n):
    return recipes.eks1_fixedpoint(
        ode_dimension=2, num_derivatives=n, calibration=calibration
    )
