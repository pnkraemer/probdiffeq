"""Test cases for IVP solver implementations."""

from probdiffeq import ivpsolvers
from probdiffeq.backend import testing


@testing.case(id="MLESolver")
def case_mle():
    return ivpsolvers.MLESolver


@testing.case(id="DynamicSolver")
def case_dynamic():
    return ivpsolvers.DynamicSolver


@testing.case(id="CalibrationFreeSolver")
def case_calibration_free():
    return ivpsolvers.CalibrationFreeSolver
