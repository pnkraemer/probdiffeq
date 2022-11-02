"""Test cases: Solver recipes.

All ODE test problems will be two-dimensional.
"""
from pytest_cases import case

from odefilter import cubature, solvers
from odefilter.implementations import batch, dense, isotropic
from odefilter.strategies import filters, smoothers


@case(tags=["terminal_value", "solve", "checkpoint", "filter"])
def solver_ekf0_isotropic():
    correction = isotropic.IsoTaylorZerothOrder()
    extrapolation = isotropic.IsoIBM.from_params()
    strategy = filters.Filter(correction=correction, extrapolation=extrapolation)
    return solvers.MLESolver(strategy=strategy)


@case(tags=["terminal_value", "solve", "smoother"])
def solver_eks0_isotropic():
    correction = isotropic.IsoTaylorZerothOrder()
    extrapolation = isotropic.IsoIBM.from_params()
    strategy = smoothers.Smoother(correction=correction, extrapolation=extrapolation)
    return solvers.DynamicSolver(strategy=strategy)


@case(tags=["terminal_value", "checkpoint", "smoother"])
def solver_eks0_isotropic_fixedpoint():
    correction = isotropic.IsoTaylorZerothOrder()
    extrapolation = isotropic.IsoIBM.from_params()
    strategy = smoothers.FixedPointSmoother(
        correction=correction, extrapolation=extrapolation
    )
    return solvers.Solver(strategy=strategy, output_scale_sqrtm=100.0)


@case(tags=["terminal_value", "solve", "checkpoint", "filter"])
def solver_ekf0_batch():
    correction = batch.BatchTaylorZerothOrder()
    extrapolation = batch.BatchIBM.from_params(ode_dimension=2, num_derivatives=3)
    strategy = filters.Filter(correction=correction, extrapolation=extrapolation)
    return solvers.DynamicSolver(strategy=strategy)


@case(tags=["terminal_value", "solve", "smoother"])
def solver_eks0_batch():
    correction = batch.BatchTaylorZerothOrder()
    extrapolation = batch.BatchIBM.from_params(ode_dimension=2)
    strategy = smoothers.Smoother(correction=correction, extrapolation=extrapolation)
    return solvers.MLESolver(strategy=strategy)


@case(tags=["terminal_value", "checkpoint", "smoother"])
def solver_eks0_batch_fixedpoint():
    correction = batch.BatchTaylorZerothOrder()
    extrapolation = batch.BatchIBM.from_params(ode_dimension=2)
    strategy = smoothers.FixedPointSmoother(
        correction=correction, extrapolation=extrapolation
    )
    return solvers.MLESolver(strategy=strategy)


@case(tags=["terminal_value", "solve", "checkpoint", "filter"])
def solver_ekf1():
    correction = dense.TaylorFirstOrder(ode_dimension=2)
    extrapolation = dense.IBM.from_params(ode_dimension=2)
    strategy = filters.Filter(correction=correction, extrapolation=extrapolation)
    return solvers.DynamicSolver(strategy=strategy)


@case(tags=["terminal_value", "solve", "checkpoint", "filter"])
def solver_ckf1():
    cube = cubature.SphericalCubatureIntegration.from_params(ode_dimension=2)
    correction = dense.MomentMatching(cubature=cube, ode_dimension=2)
    extrapolation = dense.IBM.from_params(ode_dimension=2)
    strategy = filters.Filter(correction=correction, extrapolation=extrapolation)
    return solvers.MLESolver(strategy=strategy)


@case(tags=["terminal_value", "solve", "checkpoint", "filter"])
def solver_ukf1():
    cube = cubature.UnscentedTransform.from_params(ode_dimension=2)
    correction = dense.MomentMatching(cubature=cube, ode_dimension=2)
    extrapolation = dense.IBM.from_params(ode_dimension=2)
    strategy = filters.Filter(correction=correction, extrapolation=extrapolation)
    return solvers.DynamicSolver(strategy=strategy)


@case(tags=["terminal_value", "solve", "checkpoint", "filter"])
def solver_ghkf1():
    cube = cubature.GaussHermite.from_params(ode_dimension=2)
    correction = dense.MomentMatching(cubature=cube, ode_dimension=2)
    extrapolation = dense.IBM.from_params(ode_dimension=2)
    strategy = filters.Filter(correction=correction, extrapolation=extrapolation)
    return solvers.DynamicSolver(strategy=strategy)


@case(tags=["terminal_value", "solve", "smoother"])
def solver_eks1():
    correction = dense.TaylorFirstOrder(ode_dimension=2)
    extrapolation = dense.IBM.from_params(ode_dimension=2)
    strategy = smoothers.Smoother(correction=correction, extrapolation=extrapolation)
    return solvers.MLESolver(strategy=strategy)


@case(tags=["terminal_value", "checkpoint", "smoother"])
def solver_eks1_fixedpoint():
    correction = dense.TaylorFirstOrder(ode_dimension=2)
    extrapolation = dense.IBM.from_params(ode_dimension=2)
    strategy = smoothers.FixedPointSmoother(
        correction=correction, extrapolation=extrapolation
    )
    return solvers.MLESolver(strategy=strategy)


@case(tags=["terminal_value", "checkpoint", "smoother"])
def solver_eks0_fixedpoint():
    correction = dense.TaylorZerothOrder(ode_dimension=2)
    extrapolation = dense.IBM.from_params(ode_dimension=2)
    strategy = smoothers.FixedPointSmoother(
        correction=correction, extrapolation=extrapolation
    )
    return solvers.MLESolver(strategy=strategy)
