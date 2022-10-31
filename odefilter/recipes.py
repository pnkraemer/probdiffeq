# """Recipes for ODE filters.
#
# Learning about the inner workings of an ODE filter is a little too much?
# We hear ya -- tt can indeed get quite complicated.
# Therefore, here we provide some recipes that create our favourite,
# time-tested ODE filter versions.
# We still recommend to build an ODE filter yourself,
# but until you do so, use one of ours.
#
# """
#
# from odefilter import _cubature, solvers
# from odefilter.implementations import batch, dense, isotropic
# from odefilter.strategies import filters, smoothers
#
#
# def ekf0_batch(*, ode_dimension, calibration="mle", num_derivatives=4, ode_order=1):
#     """Construct the equivalent of an explicit solver with a block-diagonal covariance \
#     structure, and optimised for terminal-value simulation.
#
#     Suitable for high-dimensional, non-stiff problems.
#     """
#     _assert_num_derivatives_sufficiently_large(
#         num_derivatives=num_derivatives, ode_order=ode_order
#     )
#
#     extrapolation = batch.BatchIBM.from_params(
#         num_derivatives=num_derivatives, ode_dimension=ode_dimension
#     )
#     correction = batch.EK0(ode_order=ode_order)
#     strategy = filters.Filter(extrapolation=extrapolation, correction=correction)
#     solver = _calibration_to_solver[calibration](strategy=strategy)
#     return solver
#
#
# def eks0_batch(*, ode_dimension, calibration="mle", num_derivatives=4, ode_order=1):
#     """Construct the equivalent of an explicit solver with a block-diagonal covariance \
#     structure.
#
#     Suitable for high-dimensional, non-stiff problems.
#     """
#     _assert_num_derivatives_sufficiently_large(
#         num_derivatives=num_derivatives, ode_order=ode_order
#     )
#     extrapolation = batch.BatchIBM.from_params(
#         num_derivatives=num_derivatives, ode_dimension=ode_dimension
#     )
#     correction = batch.EK0(ode_order=ode_order)
#     strategy = smoothers.Smoother(extrapolation=extrapolation, correction=correction)
#     solver = _calibration_to_solver[calibration](strategy=strategy)
#     return solver
#
#
# def eks0_batch_fixedpoint(
#     *, ode_dimension, calibration="mle", num_derivatives=4, ode_order=1
# ):
#     """Construct the equivalent of an explicit solver with a block-diagonal covariance \
#     structure.
#
#     Suitable for high-dimensional, non-stiff problems.
#     """
#     _assert_num_derivatives_sufficiently_large(
#         num_derivatives=num_derivatives, ode_order=ode_order
#     )
#     extrapolation = batch.BatchIBM.from_params(
#         num_derivatives=num_derivatives, ode_dimension=ode_dimension
#     )
#     correction = batch.EK0(ode_order=ode_order)
#     strategy = smoothers.FixedPointSmoother(
#         extrapolation=extrapolation, correction=correction
#     )
#     solver = _calibration_to_solver[calibration](strategy=strategy)
#     return solver
#
#
# def ekf0_isotropic(*, calibration="mle", num_derivatives=4, ode_order=1):
#     """Construct the equivalent of an explicit solver with an isotropic covariance \
#     structure, and optimised for terminal-value simulation.
#
#     Suitable for high-dimensional, non-stiff problems.
#     """
#     _assert_num_derivatives_sufficiently_large(
#         num_derivatives=num_derivatives, ode_order=ode_order
#     )
#     extrapolation = isotropic.IsotropicIBM.from_params(num_derivatives=num_derivatives)
#     correction = isotropic.EK0(ode_order=ode_order)
#     strategy = filters.Filter(extrapolation=extrapolation, correction=correction)
#     solver = _calibration_to_solver[calibration](strategy=strategy)
#     return solver
#
#
# def eks0_isotropic(*, calibration="mle", num_derivatives=4, ode_order=1):
#     """Construct the equivalent of an explicit solver with an isotropic covariance \
#     structure.
#
#     Suitable for high-dimensional, non-stiff problems.
#     """
#     _assert_num_derivatives_sufficiently_large(
#         num_derivatives=num_derivatives, ode_order=ode_order
#     )
#     extrapolation = isotropic.IsotropicIBM.from_params(num_derivatives=num_derivatives)
#     correction = isotropic.EK0(ode_order=ode_order)
#     strategy = smoothers.Smoother(extrapolation=extrapolation, correction=correction)
#     solver = _calibration_to_solver[calibration](strategy=strategy)
#     return solver
#
#
# def eks0_isotropic_fixedpoint(*, calibration="mle", num_derivatives=4, ode_order=1):
#     """Construct the equivalent of an explicit solver with an isotropic covariance \
#     structure.
#
#     Suitable for high-dimensional, non-stiff problems.
#     """
#     _assert_num_derivatives_sufficiently_large(
#         num_derivatives=num_derivatives, ode_order=ode_order
#     )
#     extrapolation = isotropic.IsotropicIBM.from_params(num_derivatives=num_derivatives)
#     correction = isotropic.EK0(ode_order=ode_order)
#     strategy = smoothers.FixedPointSmoother(
#         extrapolation=extrapolation, correction=correction
#     )
#     solver = _calibration_to_solver[calibration](strategy=strategy)
#     return solver
#
#
# def ckf1(*, ode_dimension, calibration="mle", num_derivatives=4, ode_order=1):
#     """Construct the equivalent of a semi-implicit solver that does not use Jacobians.
#
#     Suitable for low-dimensional, stiff problems.
#     """
#     _assert_num_derivatives_sufficiently_large(
#         num_derivatives=num_derivatives, ode_order=ode_order
#     )
#     extrapolation = dense.IBM.from_params(
#         num_derivatives=num_derivatives, ode_dimension=ode_dimension
#     )
#     correction = dense.CK1(
#         cubature=_cubature.SCI.from_params(dim=ode_dimension),
#         ode_dimension=ode_dimension,
#         ode_order=ode_order,
#     )
#     strategy = filters.Filter(extrapolation=extrapolation, correction=correction)
#     solver = _calibration_to_solver[calibration](strategy=strategy)
#     return solver
#
#
# def ukf1(*, ode_dimension, calibration, num_derivatives=4, ode_order=1, r=1.0):
#     """Construct the equivalent of a semi-implicit solver that does not use Jacobians.
#
#     Suitable for low-dimensional, stiff problems.
#     """
#     _assert_num_derivatives_sufficiently_large(
#         num_derivatives=num_derivatives, ode_order=ode_order
#     )
#     extrapolation = dense.IBM.from_params(
#         num_derivatives=num_derivatives, ode_dimension=ode_dimension
#     )
#     correction = dense.CK1(
#         cubature=_cubature.UT.from_params(dim=ode_dimension, r=r),
#         ode_dimension=ode_dimension,
#         ode_order=ode_order,
#     )
#     strategy = filters.Filter(extrapolation=extrapolation, correction=correction)
#     solver = _calibration_to_solver[calibration](strategy=strategy)
#     return solver
#
#
# def ghkf1(*, ode_dimension, calibration, num_derivatives=4, ode_order=1, degree=1):
#     """Construct the equivalent of a semi-implicit solver that does not use Jacobians.
#
#     Suitable for low-dimensional, stiff problems.
#     """
#     _assert_num_derivatives_sufficiently_large(
#         num_derivatives=num_derivatives, ode_order=ode_order
#     )
#     extrapolation = dense.IBM.from_params(
#         num_derivatives=num_derivatives, ode_dimension=ode_dimension
#     )
#     correction = dense.CK1(
#         cubature=_cubature.GaussHermite.from_params(degree=degree, dim=ode_dimension),
#         ode_dimension=ode_dimension,
#         ode_order=ode_order,
#     )
#     strategy = filters.Filter(extrapolation=extrapolation, correction=correction)
#     solver = _calibration_to_solver[calibration](strategy=strategy)
#     return solver
#
#
# def ekf1(*, ode_dimension, calibration="mle", num_derivatives=4, ode_order=1):
#     """Construct the equivalent of a semi-implicit solver.
#
#     Suitable for low-dimensional, stiff problems.
#     """
#     _assert_num_derivatives_sufficiently_large(
#         num_derivatives=num_derivatives, ode_order=ode_order
#     )
#     extrapolation = dense.IBM.from_params(
#         num_derivatives=num_derivatives, ode_dimension=ode_dimension
#     )
#     correction = dense.EK1(ode_dimension=ode_dimension, ode_order=ode_order)
#     strategy = filters.Filter(extrapolation=extrapolation, correction=correction)
#     solver = _calibration_to_solver[calibration](strategy=strategy)
#     return solver
#
#
# def eks1(*, ode_dimension, calibration="mle", num_derivatives=4, ode_order=1):
#     """Construct the equivalent of a semi-implicit solver.
#
#     Suitable for low-dimensional, stiff problems.
#     """
#     _assert_num_derivatives_sufficiently_large(
#         num_derivatives=num_derivatives, ode_order=ode_order
#     )
#     extrapolation = dense.IBM.from_params(
#         num_derivatives=num_derivatives, ode_dimension=ode_dimension
#     )
#     correction = dense.EK1(ode_dimension=ode_dimension, ode_order=ode_order)
#     strategy = smoothers.Smoother(extrapolation=extrapolation, correction=correction)
#     solver = _calibration_to_solver[calibration](strategy=strategy)
#     return solver
#
#
# def eks1_fixedpoint(
#     *, ode_dimension, calibration="mle", num_derivatives=4, ode_order=1
# ):
#     """Construct the equivalent of a semi-implicit solver.
#
#     Suitable for low-dimensional, stiff problems.
#     """
#     _assert_num_derivatives_sufficiently_large(
#         num_derivatives=num_derivatives, ode_order=ode_order
#     )
#     extrapolation = dense.IBM.from_params(
#         num_derivatives=num_derivatives, ode_dimension=ode_dimension
#     )
#     correction = dense.EK1(ode_dimension=ode_dimension, ode_order=ode_order)
#     strategy = smoothers.FixedPointSmoother(
#         extrapolation=extrapolation, correction=correction
#     )
#     solver = _calibration_to_solver[calibration](strategy=strategy)
#     return solver
#
#
# _calibration_to_solver = {"mle": solvers.MLESolver, "dynamic": solvers.DynamicSolver}
#
#
# def _assert_num_derivatives_sufficiently_large(*, num_derivatives, ode_order):
#     assert num_derivatives >= ode_order
