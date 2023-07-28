"""Tests for calibration/solver styles.

To ensure that all calibration-/solver-styles work correctly, we do the following:

* Only the dynamic calibration can solve an exponentially increasing problem
(see the notebook). Solve such a problem and compare to the closed-form solution.
Use simulate_terminal_values for speed.
* The output of the calibration-free solver should be identical to the output of
the MLE solver if initialised with the MLE output scale.
Both should yield a reasonable approximation of the solution.
Use solve_and_save_at for increased difficulty
(i.e. we also check interpolation and close-to-endpoint behaviour, for example)

If these hold, the calibration-/solver-styles must all work correctly
(since we already know that the MLE solver is okay).
Since calibration depends on the state-space model factorisation,
we run each test with one of each state-space models.

We solve 12 differential equations here.

"""
