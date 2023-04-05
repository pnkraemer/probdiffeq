# Choosing a solver

Good solvers are problem-dependent. Nevertheless, some guidelines exist:


* If your problem is high-dimensional, use an implementation based on Kronecker-factorisation 
  or block-diagonal covariances. 
  Kronecker-factorisation underlies all `Iso*()` methods, e.g. `IsoTS0`.
  Block-diagonal covariances correspond to `BlockDiag*()` methods, e.g. `BlockDiagTS0()`.
  The complexity of those two scales as O(d) 
  for d-dimensional problems (per step). 
  `DenseTS1()` and `DenseSLR1()` cost O(d^3) per step.
* Use `IsoTS0()` or `BlockDiagTS0()` instead of `DenseTS0()`. They do the same, but are cheaper.
  At the moment, and in the way it is implemented, `DenseTS0()` is essentially a legacy algorithm.
* If your problem is stiff, use a solver with first-order linearisation (for instance `DenseTS1` or `DenseSLR1`)
  instead of one with zeroth-order linearisation. Try to avoid too extreme state-space model factorisations (e.g. `Iso*` or `BlockDiag*`)
* Almost always, use a `Filter` strategy for `simulate_terminal_values()`, 
  a smoother strategy for `solve_with_native_python_loop()`, 
  and a `FixedPointSmoother` strategy for `solve_and_save_at()`. 
  Counterexamples do exist, but are for experienced users.
* Use `DynamicSolver()` if you expect that the output scale of your IVP solution varies greatly. 
  Otherwise, choose an `MLESolver()`. Try a `CalibrationFreeSolver()` for parameter-estimation problems.
* If you are solving a scalar differential equation (e.g. the initial values have `shape=()`), choose `Scalar*()` solvers,
  e.g. `ScalarTS0()`. These methods are designed for this simple use-case and independent of any multi-dimensional state-space model concerns.
  If you wish to use the other solvers, transform the problem into one of `shape=(1,)`.


These guidelines are a work in progress and may change soon. If you have any input, let us know!
