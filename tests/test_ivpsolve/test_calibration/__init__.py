"""Tests for calibration/solver styles.

To ensure that all calibration-/solver-styles work correctly, we do the following:

* Only the dynamic calibration can solve an exponentially increasing problem
(see the notebook). Solve such a problem and compare to the closed-form solution.
Use simulate_terminal_values for speed.


The MLE solver is heavily tested through all the other tests (it is the default).
The calibration-free solver is tested in the parameter-estimation context.
"""
