# Change log

## v0.2.0

This version overhauls large parts of the API. 
Much of the functionality now looks different.
Almost all implementations remain identical to before, though, and the API changes
reduce to renaming functions and functino parameters.
Change to the new API according to the instructions below.

Notable breaking changes:

* `solution_routines.py` has been renamed to `ivpsolve.py`. 
  The contents of both modules are identical.
  This is done to pave the way for boundary value problem solvers
  and to tighten the correspondence between solvers and solution routines.
* `solvers.py` has been renamed to `ivpsolvers.py`. 
  The contents of both modules are identical.
  This is done to pave the way for boundary value problem solvers
  and to tighten the correspondence between solvers and solution routines.
* `cubature.py` has been moved to `probdiffeq.implementations`.
  The contents of both modules are identical.
* `dense_output.py` has been renamed to `solution.py` and from now on also contains
  the `Solution` object (formerly in `solvers.py`). 
  This has been done to pave the way for boundary value problem solvers.
* `solution.negative_marginal_log_likelihood` has been renamed to
  `solution.log_marginal_likelihood` and its output is -1 times the output of the former function.
  The new term is mathematically more accurate, implements less logic and has a shorter name.
  The same applies to `solution.negative_marginal_log_likelihood_terminal_values`, which
  has become `solution.log_marginal_likelihood_terminal_values`.
* `norm_of_whitened_residual_sqrtm()` has been renamed to `mahalanobis_norm(x, /)` and is a function of one argument now.
  This is mathematically more accurate; the function should depend on an input.
* The recipes in implementation.recipes are not class-methods anymore but functions.
  For instance, instead of `recipes.IsoTS0.from_params(**kwargs)` users must call `recipes.ts0_iso(**kwargs)`.
  The advantages of this change are much less code to achieve the same logic, 
  more freedom to change background-implementations without worrying about API, 
  and improved ease of maintenance (no more classmethods, no more custom pytree node registration.)
* `extract_fn` and `extract_terminal_value_fn` in the solvers are now positional only. 
  They only depend on a single argument (`state`) and since the term `state` is so incredibly overloaded
  we make it positional only now, and will rename it afterwards. 
  To update your code, replace `solver.extract_*fn(state=x)` with `solver.extract_*fn(x)`.
* The `output_scale_sqrtm` parameter moved from being hidden in the SSM implementation (exception: CalibrationFreeSolver)
  to being an input argument to solve()-style methods. Concretely, this means that instead of
  `solve*(solver=CalibrationFreeSolver(..., output_scale_sqrtm=1.))`, users implement
  `solve(solver=CalibrationFreeSolver(...), output_scale_sqrtm=1.)` from now on.
  The advantages of this change are that all solvers are now created equal; that this opens the door to introducing more solver-hyper-parameters,
  and that it simplifies some lower-level codes. What about MLESolver and DynamicSolver? Those also accept the same argument,
  but since their outputs are independent of the prior scale (which can be shown),
  the value of output_scale_sqrtm is not important. It is set to a default value of 1.
* The `output_scale_sqrtm` parameter is now called `output_scale`. 
  This is mathematically more accurate: the parameter models $sigma$, and the `_sqrtm` 
  suffix was previously used to mark that ProbDiffEq estimates $sigma$ not $sigma^2$ (like other packages).
* The output_scale parameter is not part of the step_fn() API anymore. Instead, it is tracked in the solver state.
* `probdiffeq.implementations` is now called `probdiffeq.statespace`. The content remains the same.


Notable enhancements:

* Scalar solvers are now part of the public API. While all "other" methods are for IVPs of shape `(d,)`,
  scalar solvers target problems of shape `()` (i.e. if the initial values are floats, not arrays).
* The public API has been defined (see the developer docs). Notably, this document describes changes in which modules necessitate an entry in this changelog.
* `dt0` can now be provided to the solution routines. To do so, call `simulate_terminal_values(..., dt0=1.)` replacing `1.` with appropriate values.
  This change is completely backwards-compatible. The argument `dt0` is entirely optional, and its default value is set to the same as before.


Notable bug fixes:

* The log-pdf behaviour of Gaussian random variables has been corrected (previously, the returned values were incorrect).
  This means that the behaviour of, e.g., parameter estimation scripts will change slightly.
  A related bugfix in the whitened residuals implies that the DenseTS1 is not exactly equivalent 
  to tornadox.ReferenceEK1 anymore (because the latter still has the same error).
* The interpolation behaviour of the MLESolver when called in solve_and_save_at() had a small error, which amplified the output scale unnecessarily between steps.
  This has been fixed. As a result, the posterior-uncertainty notebook displays more realistic uncertainty estimates in high-order derivatives. Check it out!

## Prior to v0.2.0

This changelog has been started between v0.1.4 and 0.2.0.
