# Choosing a solver

Good solvers are problem-dependent. Nevertheless, some guidelines exist:

## State-space model factorisation

* If your problem is scalar-valued (`shape=()`), use a `scalar` implementation. Of course, you are always welcome to transform your problem into one with shape `(1,)` and use a vector-valued solver (not all features are implemented for scalar models).
* If your problem is vector-valued, be aware that different implementation choices imply different modelling choices.

If you don't care about modelling choices:

* If your problem is high-dimensional, use a `blockdiag` or `isotropic` implementation.
* If your problem is medium-dimensional, use any implementations. 
  `isotropic` factorisations tend to be the fastest with the worst UQ and worst stability, 
  `dense` factorisations tend to be the slowest with the best UQ and best stability, 
  `blockdiag` factorisations are somewhere in between.


## Stiffness

If your problem is stiff, use a a `dense` implementation in combination with a
correction scheme that employs first-order linearisation; 
for instance, `ts1` or `slr1`.
Zeroth-order approximation and too-aggressive state-space model factorisation 
will likely fail.

If your problem is stiff and high-dimensional: try first-order linearisation with a block-diagonal factorisation. 
If that does not work: let me know what you come up with...

## Filters vs smoothers

Almost always, use a `filters.filter_adaptive` strategy for `simulate_terminal_values`, 
a `smoothers.smoother_adaptive` strategy for `solve_and_save_every_step`,
and a `fixedpoint.fixedpoint_adaptive` strategy for `solve_and_save_at`.
Use either a filter (if you must) or a smoother (recommended) for `solve_fixed_step`.
Other combinations are possible, but rather rare 
(and require some understanding of the underlying statistical concepts).

## Calibration
Use a `solvers.dynamic` solver if you expect that the output scale of your IVP solution varies greatly.
Otherwise, use an `solvers.mle` solver.
Try a `solvers.solver` for parameter-estimation.

## Miscellaneous
If you use a `ts0`, choose an `isotropic` factorisation instead of a `dense` factorisation.
They do the same, but the `isotropic` factorisation is cheaper.


These guidelines are a work in progress and may change soon. If you have any input, let me know!
