# Change log

## v0.2.0

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


Notable enhancements:

* None


## Prior to v0.2.0

This changelog has been started between v0.1.4 and 0.2.0.
