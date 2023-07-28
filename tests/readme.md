# Readme

Why do we test the IVP solvers the way we do?

## What needs testing?

- solution routines (simulate_terminal_values, solve_and_save_at, solve_with_python_while_loop, solve_fixed_grid)
- solvers (dynamic, mle, etc.)
- strategies (filter, smoother, fixedpoint-smoother)
- state-space model factorisations, including:
  - different extrapolation types
  - different linearisation/correction types

as well as

- controls
- taylor series estimators
- dense output, log-likelihood computations, etc. (in solution.py)

## How?


### Solution routines

There is a base-configuration: 

```python
solver = mle(filter(ts0_iso()))
solution = solve_with_python_while_loop(lotka_volterra, solver, pi_control(), taylor_mode())
```

If this approximation is accurate (measured in relative error comparing to e.g. diffrax), 
the solvers work (in principle). 

To guarantee that the other solution routines also work, we do the following (everything is a single test):

* simulate_terminal_values() with the same arguments should yield the same terminal value as the base case.
* solve_and_save_at() should be identical to interpolating the solve_with_python_while_loop results
* solve_fixed_grid() should be identical if the fixed grid is the solution grid of solve_with_python_while_loop

if these tests pass, and assuming that interpolation is correct, the solution routines must be correct.
As a result, we can run all remaining tests with either one of the solution routines without loss of generality.

We run three tests with one simulation each.
We solve three differential equations.


### Strategies & extrapolation models

To ensure that all strategies work correctly, we do the following (everything is a single test):

* the RMSE of the smoother should be (slightly) lower than the RMSE of the filter using the same configuration. Both should yield a reasonable approximation of the ODE solution.
* the result of the fixed-point smoother in solve_and_save_at should be *identical* to interpolating the smoother results (we can reuse the solution from earlier). Both should yield a reasonable approximation of the ODE solution.

if these are true (and, again, assuming that interpolation works correctly), the strategies must work correctly.

Since the strategies are closely tied to the extrapolation models, we need to run these tests with one solver-recipe for each state-space model factorisation.
That means that we run both tests four times each (blockdiag, dense, iso, scalar). Each test runs two simulations. 

We solve 16 differential equations here. 


### Calibration-/solver-styles

To ensure that all calibration-/solver-styles work correctly, we do the following:

* Only the dynamic calibration can solve an exponentially increasing problem (see the notebook). Solve such a problem and compare to the closed-form solution. Use simulate_terminal_values for speed.
* The output of the calibration-free solver should be identical to the output of the MLE solver if initialised with the MLE output scale. Both should yield a reasonable approximation of the solution. Use solve_and_save_at for increased difficulty (i.e. we also check interpolation and close-to-endpoint behaviour, for example)

If these hold, the calibration-/solver-styles must all work correctly (since we already know that the MLE solver is okay).
Since calibration depends on the state-space model factorisation, we run each test with one of each state-space models.

We solve 12 differential equations here.

### State-space model factorisations

To ensure that all state-space model factorisations work as expected, we need to consider extrapolation and correction separately.
We already know that extrapolation is correct (from the strategy tests).

To check that linearisation/correction are correct, we run the base-case once for each recipe.
Currently, there are 8 recipes. 

We solve 8 differential equations here.

### Other

Cubature and taylor-series are tested separately.



We know that the isotropic TS0 works correctly in all configurations.
This means that the isotropic IBM implementation must be correct.
To ensure that the dense IBM implementation is correct, we do the following:

* Ensure that the filter, smoother, and fixed-point smoother results of the dense TS0 are identical to that of the isotropic TS0. For example, use the base-configuration and solve_fixed_step. This requires 6 simulations.
