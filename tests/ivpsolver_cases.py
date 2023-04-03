"""Test cases for implementations."""

import pytest_cases

from probdiffeq import ivpsolvers


@pytest_cases.case(id="MLESolver")
def case_mle():
    def factory(strategy, output_scale_sqrtm):
        return ivpsolvers.MLESolver(strategy)

    return factory


@pytest_cases.case(id="DynamicSolver")
def case_dynamic():
    def factory(strategy, output_scale_sqrtm):
        return ivpsolvers.DynamicSolver(strategy)

    return factory


@pytest_cases.case(id="CalibrationFreeSolver")
def case_calibration_free():
    return ivpsolvers.CalibrationFreeSolver
