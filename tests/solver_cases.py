"""Solver test cases."""
import pytest_cases

from odefilter import cubature, solvers
from odefilter.implementations import recipes
from odefilter.strategies import filters, smoothers

# Note: All ODE test problems will be two-dimensional.


@pytest_cases.case(tags=["filter"])
def case_mle_filter_ts0_iso():
    implementation = recipes.IsoTS0.from_params()
    strategy = filters.Filter(implementation=implementation)
    return solvers.MLESolver(strategy=strategy)


@pytest_cases.case(tags=["smoother"])
def case_dynamic_smoother_ts0_iso():
    implementation = recipes.IsoTS0.from_params()
    strategy = smoothers.Smoother(implementation=implementation)
    return solvers.DynamicSolver(strategy=strategy)


@pytest_cases.case(tags=["fixedpoint"])
def case_fixedpoint_ts0_iso():
    implementation = recipes.IsoTS0.from_params()
    strategy = smoothers.FixedPointSmoother(implementation=implementation)
    return solvers.Solver(strategy=strategy, output_scale_sqrtm=100.0)


@pytest_cases.case(tags=["filter"])
def case_dynamic_filter_ts0_batch():
    implementation = recipes.BatchTS0.from_params(ode_dimension=2, num_derivatives=3)
    strategy = filters.Filter(implementation=implementation)
    return solvers.DynamicSolver(strategy=strategy)


@pytest_cases.case(tags=["smoother"])
def case_mle_smoother_ts0_batch():
    implementation = recipes.BatchTS0.from_params(ode_dimension=2)
    strategy = smoothers.Smoother(implementation=implementation)
    return solvers.MLESolver(strategy=strategy)


@pytest_cases.case(tags=["fixedpoint"])
def case_mle_fixedpoint_ts0_batch():
    implementation = recipes.BatchTS0.from_params(ode_dimension=2)
    strategy = smoothers.FixedPointSmoother(implementation=implementation)
    return solvers.MLESolver(strategy=strategy)


@pytest_cases.case(tags=["filter"])
def case_dynamic_filter_ts1_vect():
    implementation = recipes.VectTS1.from_params(ode_dimension=2)
    strategy = filters.Filter(implementation=implementation)
    return solvers.DynamicSolver(strategy=strategy)


@pytest_cases.case(tags=["filter"])
def case_mle_filter_mm1_vect():
    implementation = recipes.VectMM1.from_params(ode_dimension=2)
    strategy = filters.Filter(implementation=implementation)
    return solvers.MLESolver(strategy=strategy)


@pytest_cases.case(tags=["filter"])
def case_dynamic_filter_mm1_ut_vect():
    cube = cubature.UnscentedTransform.from_params(input_shape=(2,))
    implementation = recipes.VectMM1.from_params(cubature=cube, ode_dimension=2)
    strategy = filters.Filter(implementation=implementation)
    return solvers.DynamicSolver(strategy=strategy)


@pytest_cases.case(tags=["filter"])
def case_dynamic_filter_mm1_ut_batch():
    cube = cubature.UnscentedTransform.from_params_batch(input_shape=(2,))
    implementation = recipes.BatchMM1.from_params(cubature=cube, ode_dimension=2)
    strategy = filters.Filter(implementation=implementation)
    return solvers.DynamicSolver(strategy=strategy)


@pytest_cases.case(tags=["filter"])
def case_dynamic_filter_mm1_batch():
    implementation = recipes.BatchMM1.from_params(ode_dimension=2)
    strategy = filters.Filter(implementation=implementation)
    return solvers.DynamicSolver(strategy=strategy)


@pytest_cases.case(tags=["filter"])
def case_dynamic_filter_mm1_gh_vect():
    cube = cubature.GaussHermite.from_params(input_shape=(2,))
    implementation = recipes.VectMM1.from_params(cubature=cube, ode_dimension=2)
    strategy = filters.Filter(implementation=implementation)
    return solvers.DynamicSolver(strategy=strategy)


@pytest_cases.case(tags=["smoother"])
def case_mle_smoother_ts1_vect():
    implementation = recipes.VectTS1.from_params(ode_dimension=2)
    strategy = smoothers.Smoother(implementation=implementation)
    return solvers.MLESolver(strategy=strategy)


@pytest_cases.case(tags=["fixedpoint"])
def case_mle_fixedpoint_ts1_vect():
    implementation = recipes.VectTS1.from_params(ode_dimension=2)
    strategy = smoothers.FixedPointSmoother(implementation=implementation)
    return solvers.MLESolver(strategy=strategy)


@pytest_cases.case(tags=["fixedpoint"])
def case_mle_fixedpoint_ts0_vect():
    implementation = recipes.VectTS0.from_params(ode_dimension=2)
    strategy = smoothers.FixedPointSmoother(implementation=implementation)
    return solvers.MLESolver(strategy=strategy)
