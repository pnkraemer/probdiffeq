"""Solver test cases."""
import dataclasses
from typing import Literal

import pytest_cases

from probdiffeq import cubature, solvers
from probdiffeq.implementations import recipes
from probdiffeq.strategies import filters, smoothers


@dataclasses.dataclass
class Tag:
    """Tags for IVP solvers.

    These tags are used to match compatible solvers and ODEs.
    ODEs have a similar set of tags.
    """

    strategy: Literal["filter", "smoother", "fixedpoint"]
    linearisation_order: Literal["zeroth", "first"]
    ode_shape: Literal[(2,)]  # todo: scalar problems
    ode_order: Literal[1]  # todo: second-order problems


@pytest_cases.case(tags=(Tag("filter", "zeroth", ode_shape=(2,), ode_order=1),))
def case_mle_filter_ts0_iso():
    strategy = filters.Filter(recipes.IsoTS0.from_params())
    return solvers.MLESolver(strategy=strategy)


@pytest_cases.case(tags=(Tag("smoother", "zeroth", ode_shape=(2,), ode_order=1),))
def case_dynamic_smoother_ts0_iso():
    implementation = recipes.IsoTS0.from_params()
    strategy = smoothers.Smoother(implementation=implementation)
    return solvers.DynamicSolver(strategy)


@pytest_cases.case(tags=(Tag("fixedpoint", "zeroth", ode_shape=(2,), ode_order=1),))
def case_fixedpoint_ts0_iso():
    implementation = recipes.IsoTS0.from_params()
    strategy = smoothers.FixedPointSmoother(implementation=implementation)
    return solvers.CalibrationFreeSolver(strategy=strategy, output_scale_sqrtm=100.0)


@pytest_cases.case(tags=(Tag("filter", "zeroth", ode_shape=(2,), ode_order=1),))
def case_dynamic_filter_ts0_blockdiag():
    implementation = recipes.BlockDiagTS0.from_params(ode_shape=(2,), num_derivatives=3)
    strategy = filters.Filter(implementation=implementation)
    return solvers.DynamicSolver(strategy=strategy)


@pytest_cases.case(tags=(Tag("smoother", "zeroth", ode_shape=(2,), ode_order=1),))
def case_mle_smoother_ts0_blockdiag():
    implementation = recipes.BlockDiagTS0.from_params(ode_shape=(2,))
    strategy = smoothers.Smoother(implementation=implementation)
    return solvers.MLESolver(strategy=strategy)


@pytest_cases.case(tags=(Tag("fixedpoint", "zeroth", ode_shape=(2,), ode_order=1),))
def case_mle_fixedpoint_ts0_blockdiag():
    implementation = recipes.BlockDiagTS0.from_params(ode_shape=(2,))
    strategy = smoothers.FixedPointSmoother(implementation=implementation)
    return solvers.MLESolver(strategy=strategy)


@pytest_cases.case(tags=(Tag("filter", "first", ode_shape=(2,), ode_order=1),))
def case_dynamic_filter_ts1_dense():
    implementation = recipes.DenseTS1.from_params(ode_shape=(2,))
    strategy = filters.Filter(implementation=implementation)
    return solvers.DynamicSolver(strategy=strategy)


@pytest_cases.case(tags=(Tag("filter", "first", ode_shape=(2,), ode_order=1),))
def case_mle_filter_slr1_dense():
    implementation = recipes.DenseSLR1.from_params(ode_shape=(2,))
    strategy = filters.Filter(implementation=implementation)
    return solvers.MLESolver(strategy=strategy)


@pytest_cases.case(tags=(Tag("filter", "first", ode_shape=(2,), ode_order=1),))
def case_dynamic_filter_slr1_dense_ut():
    cube = cubature.UnscentedTransform.from_params(input_shape=(2,))
    implementation = recipes.DenseSLR1.from_params(cubature=cube, ode_shape=(2,))
    strategy = filters.Filter(implementation=implementation)
    return solvers.DynamicSolver(strategy=strategy)


@pytest_cases.case(tags=(Tag("filter", "first", ode_shape=(2,), ode_order=1),))
def case_dynamic_filter_slr1_ut_blockdiag():
    cube = cubature.UnscentedTransform.from_params_blockdiag(input_shape=(2,))
    implementation = recipes.BlockDiagSLR1.from_params(cubature=cube, ode_shape=(2,))
    strategy = filters.Filter(implementation=implementation)
    return solvers.DynamicSolver(strategy=strategy)


@pytest_cases.case(tags=(Tag("filter", "first", ode_shape=(2,), ode_order=1),))
def case_dynamic_filter_slr1_blockdiag():
    implementation = recipes.BlockDiagSLR1.from_params(ode_shape=(2,))
    strategy = filters.Filter(implementation=implementation)
    return solvers.DynamicSolver(strategy=strategy)


@pytest_cases.case(tags=(Tag("filter", "first", ode_shape=(2,), ode_order=1),))
def case_dynamic_filter_slr1_dense_gh():
    cube = cubature.GaussHermite.from_params(input_shape=(2,))
    implementation = recipes.DenseSLR1.from_params(cubature=cube, ode_shape=(2,))
    strategy = filters.Filter(implementation=implementation)
    return solvers.DynamicSolver(strategy=strategy)


@pytest_cases.case(tags=(Tag("filter", "zeroth", ode_shape=(2,), ode_order=1),))
def case_dynamic_filter_slr0_dense():
    implementation = recipes.DenseSLR0.from_params(ode_shape=(2,))
    strategy = filters.Filter(implementation=implementation)
    return solvers.DynamicSolver(strategy=strategy)


@pytest_cases.case(tags=(Tag("filter", "zeroth", ode_shape=(2,), ode_order=1),))
def case_dynamic_filter_slr0_dense_gh():
    cube = cubature.GaussHermite.from_params(input_shape=(2,))
    implementation = recipes.DenseSLR0.from_params(cubature=cube, ode_shape=(2,))
    strategy = filters.Filter(implementation=implementation)
    return solvers.DynamicSolver(strategy=strategy)


@pytest_cases.case(tags=(Tag("smoother", "first", ode_shape=(2,), ode_order=1),))
def case_mle_smoother_ts1_dense():
    implementation = recipes.DenseTS1.from_params(ode_shape=(2,))
    strategy = smoothers.Smoother(implementation=implementation)
    return solvers.MLESolver(strategy=strategy)


@pytest_cases.case(tags=(Tag("fixedpoint", "first", ode_shape=(2,), ode_order=1),))
def case_mle_fixedpoint_ts1_dense():
    implementation = recipes.DenseTS1.from_params(ode_shape=(2,))
    strategy = smoothers.FixedPointSmoother(implementation=implementation)
    return solvers.MLESolver(strategy=strategy)


@pytest_cases.case(tags=(Tag("fixedpoint", "zeroth", ode_shape=(2,), ode_order=1),))
def case_mle_fixedpoint_ts0_dense():
    implementation = recipes.DenseTS0.from_params(ode_shape=(2,))
    strategy = smoothers.FixedPointSmoother(implementation=implementation)
    return solvers.MLESolver(strategy=strategy)
