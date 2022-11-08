"""Solver test cases."""
import pytest_cases

from odefilter import cubature, solvers
from odefilter.implementations import implementations
from odefilter.strategies import filters, smoothers

# Note: All ODE test problems will be two-dimensional.


@pytest_cases.case(tags=["filter"])
def case_mle_filter_ts0_iso():
    implementation = implementations.IsoTS0.from_params()
    strategy = filters.Filter(implementation=implementation)
    return solvers.MLESolver(strategy=strategy)


@pytest_cases.case(tags=["smoother"])
def case_dynamic_smoother_ts0_iso():
    implementation = implementations.IsoTS0.from_params()
    strategy = smoothers.Smoother(implementation=implementation)
    return solvers.DynamicSolver(strategy=strategy)


@pytest_cases.case(tags=["fixedpoint"])
def case_fixedpoint_ts0_iso():
    implementation = implementations.IsoTS0.from_params()
    strategy = smoothers.FixedPointSmoother(implementation=implementation)
    return solvers.Solver(strategy=strategy, output_scale_sqrtm=100.0)


@pytest_cases.case(tags=["filter"])
def case_dynamic_filter_ts0_batch():
    implementation = implementations.BatchTS0.from_params(
        ode_dimension=2, num_derivatives=3
    )
    strategy = filters.Filter(implementation=implementation)
    return solvers.DynamicSolver(strategy=strategy)


@pytest_cases.case(tags=["smoother"])
def case_mle_smoother_ts0_batch():
    implementation = implementations.BatchTS0.from_params(ode_dimension=2)
    strategy = smoothers.Smoother(implementation=implementation)
    return solvers.MLESolver(strategy=strategy)


@pytest_cases.case(tags=["fixedpoint"])
def case_mle_fixedpoint_ts0_batch():
    implementation = implementations.BatchTS0.from_params(ode_dimension=2)
    strategy = smoothers.FixedPointSmoother(implementation=implementation)
    return solvers.MLESolver(strategy=strategy)


@pytest_cases.case(tags=["filter"])
def case_dynamic_filter_ts1():
    implementation = implementations.TS1.from_params(ode_dimension=2)
    strategy = filters.Filter(implementation=implementation)
    return solvers.DynamicSolver(strategy=strategy)


@pytest_cases.case(tags=["filter"])
def case_mle_filter_mm1_sci():
    cube = cubature.SphericalCubatureIntegration.from_params(ode_dimension=2)
    implementation = implementations.MM1.from_params(cubature=cube, ode_dimension=2)
    strategy = filters.Filter(implementation=implementation)
    return solvers.MLESolver(strategy=strategy)


@pytest_cases.case(tags=["filter"])
def case_dynamic_filter_mm1_ut():
    cube = cubature.UnscentedTransform.from_params(ode_dimension=2)
    implementation = implementations.MM1.from_params(cubature=cube, ode_dimension=2)
    strategy = filters.Filter(implementation=implementation)
    return solvers.DynamicSolver(strategy=strategy)


@pytest_cases.case(tags=["filter"])
def case_dynamic_filter_mm1_ut_batch():
    cube = cubature.UnscentedTransform.from_params(ode_dimension=2)
    implementation = implementations.BatchMM1.from_params(
        cubature=cube, ode_dimension=2
    )
    strategy = filters.Filter(implementation=implementation)
    return solvers.DynamicSolver(strategy=strategy)


@pytest_cases.case(tags=["filter"])
def case_dynamic_filter_mm1_gh():
    cube = cubature.GaussHermite.from_params(ode_dimension=2)
    implementation = implementations.MM1.from_params(cubature=cube, ode_dimension=2)
    strategy = filters.Filter(implementation=implementation)
    return solvers.DynamicSolver(strategy=strategy)


@pytest_cases.case(tags=["smoother"])
def case_mle_smoother_ts1():
    implementation = implementations.TS1.from_params(ode_dimension=2)
    strategy = smoothers.Smoother(implementation=implementation)
    return solvers.MLESolver(strategy=strategy)


@pytest_cases.case(tags=["fixedpoint"])
def case_mle_fixedpoint_ts1():
    implementation = implementations.TS1.from_params(ode_dimension=2)
    strategy = smoothers.FixedPointSmoother(implementation=implementation)
    return solvers.MLESolver(strategy=strategy)


@pytest_cases.case(tags=["fixedpoint"])
def case_mle_fixedpoint_ts0():
    implementation = implementations.TS0.from_params(ode_dimension=2)
    strategy = smoothers.FixedPointSmoother(implementation=implementation)
    return solvers.MLESolver(strategy=strategy)
