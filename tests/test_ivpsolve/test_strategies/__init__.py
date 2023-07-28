"""Tests for the strategy choices.

To ensure that all strategies work correctly, we do the following:

* the RMSE of the smoother should be (slightly) lower than the RMSE of the filter
using the same configuration (e.g. fixed grid solutions).
Both should yield a reasonable approximation of the ODE solution.
* the result of the fixed-point smoother in solve_and_save_at should be *identical*
to interpolating the smoother results (we can reuse the solution from earlier).
Both should yield a reasonable approximation of the ODE solution.

if these are true (and, again, assuming that interpolation works correctly),
the strategies must work correctly.

Since the strategies are closely tied to the extrapolation models,
we need to run these tests with one solver-recipe for each state-space factorisation.
That means that we run both tests three times each
(blockdiag, dense, iso; scalar is covered by the blockdiag tests).
Each test runs two simulations.
We solve 12 differential equations here.

"""
