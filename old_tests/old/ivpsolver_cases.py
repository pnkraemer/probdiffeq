"""Test cases for IVP solver implementations."""

from probdiffeq import ivpsolvers
from probdiffeq.backend import testing


@testing.case(id="solver_mle")
def case_mle():
    return ivpsolvers.solver_mle


@testing.case(id="solver_dynamic")
def case_dynamic():
    return ivpsolvers.solver_dynamic


@testing.case(id="solver_calibrationfree")
def case_calibration_free():
    return ivpsolvers.solver_calibrationfree
