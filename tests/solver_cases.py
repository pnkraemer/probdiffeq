"""Solver test cases."""
import pytest_cases

from odefilter import cubature, solvers
from odefilter.implementations import batch, dense, isotropic
from odefilter.strategies import filters, smoothers

# Note: All ODE test problems will be two-dimensional.


@pytest_cases.case(tags=["filter"])
def case_ekf0_isotropic():
    implementation = isotropic.IsoTS0.from_params()
    strategy = filters.Filter(implementation=implementation)
    return solvers.MLESolver(strategy=strategy)


@pytest_cases.case(tags=["smoother"])
def case_eks0_isotropic():
    implementation = isotropic.IsoTS0.from_params()
    strategy = smoothers.Smoother(implementation=implementation)
    return solvers.DynamicSolver(strategy=strategy)


@pytest_cases.case(tags=["fixedpoint"])
def case_eks0_isotropic_fixedpoint():
    implementation = isotropic.IsoTS0.from_params()
    strategy = smoothers.FixedPointSmoother(implementation=implementation)
    return solvers.Solver(strategy=strategy, output_scale_sqrtm=100.0)


@pytest_cases.case(tags=["filter"])
def case_ekf0_batch():
    correction = batch.BatchTaylorZerothOrder()
    extrapolation = batch.BatchIBM.from_params(ode_dimension=2, num_derivatives=3)
    strategy = filters.Filter(correction=correction, extrapolation=extrapolation)
    return solvers.DynamicSolver(strategy=strategy)


@pytest_cases.case(tags=["smoother"])
def case_eks0_batch():
    correction = batch.BatchTaylorZerothOrder()
    extrapolation = batch.BatchIBM.from_params(ode_dimension=2)
    strategy = smoothers.Smoother(correction=correction, extrapolation=extrapolation)
    return solvers.MLESolver(strategy=strategy)


@pytest_cases.case(tags=["fixedpoint"])
def case_eks0_batch_fixedpoint():
    correction = batch.BatchTaylorZerothOrder()
    extrapolation = batch.BatchIBM.from_params(ode_dimension=2)
    strategy = smoothers.FixedPointSmoother(
        correction=correction, extrapolation=extrapolation
    )
    return solvers.MLESolver(strategy=strategy)


@pytest_cases.case(tags=["filter"])
def case_ekf1():
    correction = dense.TaylorFirstOrder(ode_dimension=2)
    extrapolation = dense.IBM.from_params(ode_dimension=2)
    strategy = filters.Filter(correction=correction, extrapolation=extrapolation)
    return solvers.DynamicSolver(strategy=strategy)


@pytest_cases.case(tags=["filter"])
def case_ckf1():
    cube = cubature.SphericalCubatureIntegration.from_params(ode_dimension=2)
    correction = dense.MomentMatching(cubature=cube, ode_dimension=2)
    extrapolation = dense.IBM.from_params(ode_dimension=2)
    strategy = filters.Filter(correction=correction, extrapolation=extrapolation)
    return solvers.MLESolver(strategy=strategy)


@pytest_cases.case(tags=["filter"])
def case_ukf1():
    cube = cubature.UnscentedTransform.from_params(ode_dimension=2)
    correction = dense.MomentMatching(cubature=cube, ode_dimension=2)
    extrapolation = dense.IBM.from_params(ode_dimension=2)
    strategy = filters.Filter(correction=correction, extrapolation=extrapolation)
    return solvers.DynamicSolver(strategy=strategy)


@pytest_cases.case(tags=["filter"])
def case_ghkf1():
    cube = cubature.GaussHermite.from_params(ode_dimension=2)
    correction = dense.MomentMatching(cubature=cube, ode_dimension=2)
    extrapolation = dense.IBM.from_params(ode_dimension=2)
    strategy = filters.Filter(correction=correction, extrapolation=extrapolation)
    return solvers.DynamicSolver(strategy=strategy)


@pytest_cases.case(tags=["smoother"])
def case_eks1():
    correction = dense.TaylorFirstOrder(ode_dimension=2)
    extrapolation = dense.IBM.from_params(ode_dimension=2)
    strategy = smoothers.Smoother(correction=correction, extrapolation=extrapolation)
    return solvers.MLESolver(strategy=strategy)


@pytest_cases.case(tags=["fixedpoint"])
def case_eks1_fixedpoint():
    correction = dense.TaylorFirstOrder(ode_dimension=2)
    extrapolation = dense.IBM.from_params(ode_dimension=2)
    strategy = smoothers.FixedPointSmoother(
        correction=correction, extrapolation=extrapolation
    )
    return solvers.MLESolver(strategy=strategy)


@pytest_cases.case(tags=["fixedpoint"])
def case_eks0_fixedpoint():
    correction = dense.TaylorZerothOrder(ode_dimension=2)
    extrapolation = dense.IBM.from_params(ode_dimension=2)
    strategy = smoothers.FixedPointSmoother(
        correction=correction, extrapolation=extrapolation
    )
    return solvers.MLESolver(strategy=strategy)
