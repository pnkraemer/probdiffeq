"""Test cases for implementations."""

from probdiffeq import ivpsolvers
from probdiffeq.backend import testing


@testing.case(id="MLESolver")
def case_mle():
    return ivpsolvers.MLESolver
    # def factory(strategy, output_scale):
    #     return ivpsolvers.MLESolver(strategy)
    #
    # return factory


@testing.case(id="DynamicSolver")
def case_dynamic():
    return ivpsolvers.DynamicSolver
    # def factory(strategy, output_scale):
    #     return ivpsolvers.DynamicSolver(strategy)
    #
    # return factory


@testing.case(id="CalibrationFreeSolver")
def case_calibration_free():
    return ivpsolvers.CalibrationFreeSolver
