# Changelog

## v0.3.0

**New features:**

* A new function `taylor_mode_unroll` implements Taylor-series estimation without a `scan`.

**Breaking changes:**

* What was formerly `taylor_mode()`, is now `taylor_mode_scan()` and stands in contrast to the new `taylor_mode_unroll()`.
* What was formerly `forward_mode()`, is now `odejet_via_jvp()`.
* The entire `taylor` subpackage moved to top-level. Instead of `from probdiffeq.solvers.taylor import ...`, use `from probdiffeq.taylor import ...`. 


## v0.2.2

This release was due to issues in the publishing workflow.

## v0.2.1

**Breaking changes:**

* The input-argument to `taylor_mode_doubling` is `num_doublings` instead of `num`.
  This argument behaves differently to e.g., `taylor_mode_scan(..., num)`.


## v0.2.0

This version overhauls large parts of the API. 
Consider the quickstart for an introduction about the "new" way of doing things.
From now on, this change log will be used properly.

**Notable bug-fixes:**

* The log-pdf behaviour of Gaussian random variables has been corrected (previously, the returned values were slightly incorrect).
  This means that the behaviour of, e.g., parameter estimation scripts will change slightly.
  A related bugfix in computing the whitened residuals implies that the dynamic solver with a ts1() correction and a dense implementation is not exactly equivalent 
  to tornadox.ReferenceEK1 anymore (because the tornadox-version still has the same error).
* The interpolation behaviour of the MLESolver when called in solve_adaptive_save_at() had a small error, which amplified the output scale unnecessarily between steps.
  This has been fixed. As a result, the posterior-uncertainty notebook displays more realistic uncertainty estimates in high-order derivatives. Check it out!

## Prior to v0.2.0

This changelog has been started between v0.1.4 and 0.2.0.
