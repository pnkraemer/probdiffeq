"""Solver test cases."""
from typing import Literal, NamedTuple, Tuple

import pytest_cases

from probdiffeq import cubature, solvers
from probdiffeq.implementations import recipes
from probdiffeq.strategies import filters, smoothers


class Tag(NamedTuple):
    strategy: Literal["filter", "smoother", "fixedpoint"]
    linearisation_order: Literal["zeroth", "first"]
    ode_shape: Tuple[int]
    ode_order: int  # todo: second-order problems


@pytest_cases.case(tags=Tag("filter", "zeroth", ode_shape=(2,), ode_order=1))
def case_mle_filter_ts0_iso():
    strategy = filters.Filter(recipes.IsoTS0.from_params())
    return solvers.MLESolver(strategy=strategy)


@pytest_cases.case(tags=Tag("smoother", "zeroth", ode_shape=(2,), ode_order=1))
def case_dynamic_smoother_ts0_iso():
    implementation = recipes.IsoTS0.from_params()
    strategy = smoothers.Smoother(implementation=implementation)
    return solvers.DynamicSolver(strategy)


@pytest_cases.case(tags=Tag("fixedpoint", "zeroth", ode_shape=(2,), ode_order=1))
def case_fixedpoint_ts0_iso():
    implementation = recipes.IsoTS0.from_params()
    strategy = smoothers.FixedPointSmoother(implementation=implementation)
    return solvers.Solver(strategy=strategy, output_scale_sqrtm=100.0)


@pytest_cases.case(tags=Tag("filter", "zeroth", ode_shape=(2,), ode_order=1))
def case_dynamic_filter_ts0_batch():
    implementation = recipes.BatchTS0.from_params(ode_shape=(2,), num_derivatives=3)
    strategy = filters.Filter(implementation=implementation)
    return solvers.DynamicSolver(strategy=strategy)


@pytest_cases.case(tags=Tag("smoother", "zeroth", ode_shape=(2,), ode_order=1))
def case_mle_smoother_ts0_batch():
    implementation = recipes.BatchTS0.from_params(ode_shape=(2,))
    strategy = smoothers.Smoother(implementation=implementation)
    return solvers.MLESolver(strategy=strategy)


@pytest_cases.case(tags=Tag("fixedpoint", "zeroth", ode_shape=(2,), ode_order=1))
def case_mle_fixedpoint_ts0_batch():
    implementation = recipes.BatchTS0.from_params(ode_shape=(2,))
    strategy = smoothers.FixedPointSmoother(implementation=implementation)
    return solvers.MLESolver(strategy=strategy)


@pytest_cases.case(tags=Tag("filter", "first", ode_shape=(2,), ode_order=1))
def case_dynamic_filter_ts1_vect():
    implementation = recipes.VectTS1.from_params(ode_shape=(2,))
    strategy = filters.Filter(implementation=implementation)
    return solvers.DynamicSolver(strategy=strategy)


@pytest_cases.case(tags=Tag("filter", "first", ode_shape=(2,), ode_order=1))
def case_mle_filter_slr1_vect():
    implementation = recipes.VectSLR1.from_params(ode_shape=(2,))
    strategy = filters.Filter(implementation=implementation)
    return solvers.MLESolver(strategy=strategy)


@pytest_cases.case(tags=Tag("filter", "first", ode_shape=(2,), ode_order=1))
def case_dynamic_filter_slr1_ut_vect():
    cube = cubature.UnscentedTransform.from_params(input_shape=(2,))
    implementation = recipes.VectSLR1.from_params(cubature=cube, ode_shape=(2,))
    strategy = filters.Filter(implementation=implementation)
    return solvers.DynamicSolver(strategy=strategy)


@pytest_cases.case(tags=Tag("filter", "first", ode_shape=(2,), ode_order=1))
def case_dynamic_filter_slr1_ut_batch():
    cube = cubature.UnscentedTransform.from_params_batch(input_shape=(2,))
    implementation = recipes.BatchSLR1.from_params(cubature=cube, ode_shape=(2,))
    strategy = filters.Filter(implementation=implementation)
    return solvers.DynamicSolver(strategy=strategy)


@pytest_cases.case(tags=Tag("filter", "first", ode_shape=(2,), ode_order=1))
def case_dynamic_filter_slr1_batch():
    implementation = recipes.BatchSLR1.from_params(ode_shape=(2,))
    strategy = filters.Filter(implementation=implementation)
    return solvers.DynamicSolver(strategy=strategy)


@pytest_cases.case(tags=Tag("filter", "first", ode_shape=(2,), ode_order=1))
def case_dynamic_filter_slr1_gh_vect():
    cube = cubature.GaussHermite.from_params(input_shape=(2,))
    implementation = recipes.VectSLR1.from_params(cubature=cube, ode_shape=(2,))
    strategy = filters.Filter(implementation=implementation)
    return solvers.DynamicSolver(strategy=strategy)


@pytest_cases.case(tags=Tag("smoother", "first", ode_shape=(2,), ode_order=1))
def case_mle_smoother_ts1_vect():
    implementation = recipes.VectTS1.from_params(ode_shape=(2,))
    strategy = smoothers.Smoother(implementation=implementation)
    return solvers.MLESolver(strategy=strategy)


@pytest_cases.case(tags=Tag("fixedpoint", "first", ode_shape=(2,), ode_order=1))
def case_mle_fixedpoint_ts1_vect():
    implementation = recipes.VectTS1.from_params(ode_shape=(2,))
    strategy = smoothers.FixedPointSmoother(implementation=implementation)
    return solvers.MLESolver(strategy=strategy)


@pytest_cases.case(tags=Tag("fixedpoint", "zeroth", ode_shape=(2,), ode_order=1))
def case_mle_fixedpoint_ts0_vect():
    implementation = recipes.VectTS0.from_params(ode_shape=(2,))
    strategy = smoothers.FixedPointSmoother(implementation=implementation)
    return solvers.MLESolver(strategy=strategy)
