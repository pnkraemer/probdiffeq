"""Recipes for ODE filters.

Learning about the inner workings of an ODE filter is a little too much?
We hear ya -- tt can indeed get quite complicated.
Therefore, here we provide some recipes that create our favourite,
time-tested ODE filter versions.
We still recommend to build an ODE filter yourself,
but until you do so, use one of ours.

"""
import jax.tree_util

from odefilter import _cubature, solvers
from odefilter.implementations import batch, dense, isotropic
from odefilter.strategies import filters, smoothers


def ekf0_batch(*, ode_dimension, num_derivatives=4, ode_order=1):
    """Construct the equivalent of an explicit solver with a block-diagonal covariance \
    structure, and optimised for terminal-value simulation.

    Suitable for high-dimensional, non-stiff problems.
    """
    _assert_num_derivatives_sufficiently_large(
        num_derivatives=num_derivatives, ode_order=ode_order
    )
    implementation = batch.BatchIBM.from_num_derivatives(
        num_derivatives=num_derivatives, ode_dimension=ode_dimension
    )
    strategy = filters.Filter(implementation=implementation)
    solver = solvers.Solver(strategy=strategy)
    information_op = jax.tree_util.Partial(batch.EK0, ode_order=ode_order)
    return solver, information_op


def ekf0_batch_dynamic(*, ode_dimension, num_derivatives=4, ode_order=1):
    """Construct the equivalent of an explicit solver with a block-diagonal covariance \
    structure, dynamic calibration, and optimised for terminal-value simulation.

    Suitable for high-dimensional, non-stiff problems.
    """
    _assert_num_derivatives_sufficiently_large(
        num_derivatives=num_derivatives, ode_order=ode_order
    )
    implementation = batch.BatchIBM.from_num_derivatives(
        num_derivatives=num_derivatives, ode_dimension=ode_dimension
    )
    strategy = filters.Filter(implementation=implementation)
    solver = solvers.DynamicSolver(strategy=strategy)
    information_op = jax.tree_util.Partial(batch.EK0, ode_order=ode_order)
    return solver, information_op


def eks0_batch(*, ode_dimension, num_derivatives=4, ode_order=1):
    """Construct the equivalent of an explicit solver with a block-diagonal covariance \
    structure.

    Suitable for high-dimensional, non-stiff problems.
    """
    _assert_num_derivatives_sufficiently_large(
        num_derivatives=num_derivatives, ode_order=ode_order
    )
    implementation = batch.BatchIBM.from_num_derivatives(
        num_derivatives=num_derivatives, ode_dimension=ode_dimension
    )
    strategy = smoothers.Smoother(implementation=implementation)
    solver = solvers.Solver(strategy=strategy)
    information_op = jax.tree_util.Partial(batch.EK0, ode_order=ode_order)
    return solver, information_op


def eks0_batch_dynamic(*, ode_dimension, num_derivatives=4, ode_order=1):
    """Construct the equivalent of an explicit solver with a block-diagonal covariance \
    structure and dynamic calibration.

    Suitable for high-dimensional, non-stiff problems.
    """
    _assert_num_derivatives_sufficiently_large(
        num_derivatives=num_derivatives, ode_order=ode_order
    )
    implementation = batch.BatchIBM.from_num_derivatives(
        num_derivatives=num_derivatives, ode_dimension=ode_dimension
    )
    strategy = smoothers.Smoother(implementation=implementation)
    solver = solvers.DynamicSolver(strategy=strategy)
    information_op = jax.tree_util.Partial(batch.EK0, ode_order=ode_order)
    return solver, information_op


def eks0_batch_fixedpoint(*, ode_dimension, num_derivatives=4, ode_order=1):
    """Construct the equivalent of an explicit solver with a block-diagonal covariance \
    structure.

    Suitable for high-dimensional, non-stiff problems.
    """
    _assert_num_derivatives_sufficiently_large(
        num_derivatives=num_derivatives, ode_order=ode_order
    )
    implementation = batch.BatchIBM.from_num_derivatives(
        num_derivatives=num_derivatives, ode_dimension=ode_dimension
    )
    strategy = smoothers.FixedPointSmoother(implementation=implementation)
    solver = solvers.Solver(strategy=strategy)
    information_op = jax.tree_util.Partial(batch.EK0, ode_order=ode_order)
    return solver, information_op


def eks0_batch_dynamic_fixedpoint(*, ode_dimension, num_derivatives=4, ode_order=1):
    """Construct the equivalent of an explicit solver with a block-diagonal covariance \
    structure and dynamic calibration.

    Suitable for high-dimensional, non-stiff problems.
    """
    _assert_num_derivatives_sufficiently_large(
        num_derivatives=num_derivatives, ode_order=ode_order
    )
    implementation = batch.BatchIBM.from_num_derivatives(
        num_derivatives=num_derivatives, ode_dimension=ode_dimension
    )
    strategy = smoothers.FixedPointSmoother(implementation=implementation)
    solver = solvers.DynamicSolver(strategy=strategy)
    information_op = jax.tree_util.Partial(batch.EK0, ode_order=ode_order)
    return solver, information_op


def ekf0_isotropic(*, calibration="mle", num_derivatives=4, ode_order=1):
    """Construct the equivalent of an explicit solver with an isotropic covariance \
    structure, and optimised for terminal-value simulation.

    Suitable for high-dimensional, non-stiff problems.
    """
    _assert_num_derivatives_sufficiently_large(
        num_derivatives=num_derivatives, ode_order=ode_order
    )
    implementation = isotropic.IsotropicIBM.from_num_derivatives(
        num_derivatives=num_derivatives
    )
    strategy = filters.Filter(implementation=implementation)
    solver = _calibration_to_solver[calibration](strategy=strategy)
    information_op = jax.tree_util.Partial(isotropic.EK0, ode_order=ode_order)
    return solver, information_op


def eks0_isotropic(*, calibration="mle", num_derivatives=4, ode_order=1):
    """Construct the equivalent of an explicit solver with an isotropic covariance \
    structure.

    Suitable for high-dimensional, non-stiff problems.
    """
    _assert_num_derivatives_sufficiently_large(
        num_derivatives=num_derivatives, ode_order=ode_order
    )
    implementation = isotropic.IsotropicIBM.from_num_derivatives(
        num_derivatives=num_derivatives
    )
    strategy = smoothers.Smoother(implementation=implementation)
    solver = _calibration_to_solver[calibration](strategy=strategy)
    information_op = jax.tree_util.Partial(isotropic.EK0, ode_order=ode_order)
    return solver, information_op


def eks0_isotropic_fixedpoint(*, calibration="mle", num_derivatives=4, ode_order=1):
    """Construct the equivalent of an explicit solver with an isotropic covariance \
    structure.

    Suitable for high-dimensional, non-stiff problems.
    """
    _assert_num_derivatives_sufficiently_large(
        num_derivatives=num_derivatives, ode_order=ode_order
    )
    implementation = isotropic.IsotropicIBM.from_num_derivatives(
        num_derivatives=num_derivatives
    )
    strategy = smoothers.FixedPointSmoother(implementation=implementation)
    solver = _calibration_to_solver[calibration](strategy=strategy)
    information_op = jax.tree_util.Partial(isotropic.EK0, ode_order=ode_order)
    return solver, information_op


_calibration_to_solver = {"mle": solvers.Solver, "dynamic": solvers.DynamicSolver}


def ckf1(*, ode_dimension, num_derivatives=4, ode_order=1):
    """Construct the equivalent of a semi-implicit solver that does not use Jacobians.

    Suitable for low-dimensional, stiff problems.
    """
    _assert_num_derivatives_sufficiently_large(
        num_derivatives=num_derivatives, ode_order=ode_order
    )
    implementation = dense.IBM.from_num_derivatives(
        num_derivatives=num_derivatives, ode_dimension=ode_dimension
    )
    strategy = filters.Filter(implementation=implementation)
    solver = solvers.Solver(strategy=strategy)

    information_op = jax.tree_util.Partial(
        dense.CK1,
        cubature=_cubature.SCI.from_params(dim=ode_dimension),
        ode_dimension=ode_dimension,
        ode_order=ode_order,
    )
    return solver, information_op


def ukf1(*, ode_dimension, num_derivatives=4, ode_order=1, r=1.0):
    """Construct the equivalent of a semi-implicit solver that does not use Jacobians.

    Suitable for low-dimensional, stiff problems.
    """
    _assert_num_derivatives_sufficiently_large(
        num_derivatives=num_derivatives, ode_order=ode_order
    )
    implementation = dense.IBM.from_num_derivatives(
        num_derivatives=num_derivatives, ode_dimension=ode_dimension
    )
    strategy = filters.Filter(implementation=implementation)
    solver = solvers.Solver(strategy=strategy)

    information_op = jax.tree_util.Partial(
        dense.CK1,
        cubature=_cubature.UT.from_params(dim=ode_dimension, r=r),
        ode_dimension=ode_dimension,
        ode_order=ode_order,
    )
    return solver, information_op


def ekf1(*, ode_dimension, num_derivatives=4, ode_order=1):
    """Construct the equivalent of a semi-implicit solver.

    Suitable for low-dimensional, stiff problems.
    """
    _assert_num_derivatives_sufficiently_large(
        num_derivatives=num_derivatives, ode_order=ode_order
    )
    implementation = dense.IBM.from_num_derivatives(
        num_derivatives=num_derivatives, ode_dimension=ode_dimension
    )
    strategy = filters.Filter(implementation=implementation)
    solver = solvers.Solver(strategy=strategy)

    information_op = jax.tree_util.Partial(
        dense.EK1, ode_dimension=ode_dimension, ode_order=ode_order
    )
    return solver, information_op


def ekf1_dynamic(*, ode_dimension, num_derivatives=4, ode_order=1):
    """Construct the equivalent of a semi-implicit solver with \
     dynamic calibration, and optimised for terminal-value simulation.

    Suitable for low-dimensional, stiff problems.
    """
    _assert_num_derivatives_sufficiently_large(
        num_derivatives=num_derivatives, ode_order=ode_order
    )
    implementation = dense.IBM.from_num_derivatives(
        num_derivatives=num_derivatives, ode_dimension=ode_dimension
    )
    strategy = filters.Filter(implementation=implementation)
    solver = solvers.DynamicSolver(strategy=strategy)
    information_op = jax.tree_util.Partial(
        dense.EK1, ode_dimension=ode_dimension, ode_order=ode_order
    )
    return solver, information_op


def eks1(*, ode_dimension, num_derivatives=4, ode_order=1):
    """Construct the equivalent of a semi-implicit solver.

    Suitable for low-dimensional, stiff problems.
    """
    _assert_num_derivatives_sufficiently_large(
        num_derivatives=num_derivatives, ode_order=ode_order
    )
    implementation = dense.IBM.from_num_derivatives(
        num_derivatives=num_derivatives, ode_dimension=ode_dimension
    )
    strategy = smoothers.Smoother(implementation=implementation)
    solver = solvers.Solver(strategy=strategy)
    information_op = jax.tree_util.Partial(
        dense.EK1, ode_dimension=ode_dimension, ode_order=ode_order
    )
    return solver, information_op


def eks1_dynamic(*, ode_dimension, num_derivatives=4, ode_order=1):
    """Construct the equivalent of a semi-implicit solver with dynamic calibration.

    Suitable for low-dimensional, stiff problems.
    """
    _assert_num_derivatives_sufficiently_large(
        num_derivatives=num_derivatives, ode_order=ode_order
    )
    implementation = dense.IBM.from_num_derivatives(
        num_derivatives=num_derivatives, ode_dimension=ode_dimension
    )
    strategy = smoothers.Smoother(implementation=implementation)
    solver = solvers.DynamicSolver(strategy=strategy)
    information_op = jax.tree_util.Partial(
        dense.EK1, ode_dimension=ode_dimension, ode_order=ode_order
    )
    return solver, information_op


def eks1_fixedpoint(*, ode_dimension, num_derivatives=4, ode_order=1):
    """Construct the equivalent of a semi-implicit solver.

    Suitable for low-dimensional, stiff problems.
    """
    _assert_num_derivatives_sufficiently_large(
        num_derivatives=num_derivatives, ode_order=ode_order
    )
    implementation = dense.IBM.from_num_derivatives(
        num_derivatives=num_derivatives, ode_dimension=ode_dimension
    )
    strategy = smoothers.FixedPointSmoother(implementation=implementation)
    solver = solvers.Solver(strategy=strategy)
    information_op = jax.tree_util.Partial(
        dense.EK1, ode_dimension=ode_dimension, ode_order=ode_order
    )
    return solver, information_op


def eks1_dynamic_fixedpoint(*, ode_dimension, num_derivatives=4, ode_order=1):
    """Construct the equivalent of a semi-implicit solver with dynamic calibration.

    Suitable for low-dimensional, stiff problems.
    """
    _assert_num_derivatives_sufficiently_large(
        num_derivatives=num_derivatives, ode_order=ode_order
    )

    implementation = dense.IBM.from_num_derivatives(
        num_derivatives=num_derivatives, ode_dimension=ode_dimension
    )
    strategy = smoothers.FixedPointSmoother(implementation=implementation)
    solver = solvers.DynamicSolver(strategy=strategy)
    information_op = jax.tree_util.Partial(
        dense.EK1, ode_dimension=ode_dimension, ode_order=ode_order
    )
    return solver, information_op


def _assert_num_derivatives_sufficiently_large(*, num_derivatives, ode_order):
    assert num_derivatives >= ode_order
