# Choosing a solver

Good solvers are problem-dependent. However, some guidelines exist:

## State-space model factorisation

* If your problem is scalar-valued (`shape=()`), use a dense factorisation. All factorisations have the same complexity for scalar models, but dense factorisations offer the most comprehensive solver suite.

* If your problem is vector-valued, be aware that different implementation choices imply different modelling choices.
However, if you don't care too much about modelling choices:

* If your problem is high-dimensional, use a `blockdiag` or `isotropic` implementation.
* If your problem is medium-dimensional, use any implementations. 
  `isotropic` factorisations tend to be the fastest with the worst UQ and worst stability, 
  `dense` factorisations tend to be the slowest with the best UQ and best stability, 
  `blockdiag` factorisations are somewhere in between.


## Stiffness

If your problem is stiff, use a a `dense` implementation in combination with a
correction scheme that employs first-order linearisation; 
for instance, `ts1` or `slr1`.
Zeroth-order approximation and isotropic/blockdiag factorisations often fail for stiff problems.

If your problem is stiff and high-dimensional: try first-order linearisation with a block-diagonal factorisation. 
If that does not work: good luck; probabilistic solvers for problems that are stiff 
*and* high-dimensional are a bit of an open problem as of writing this.

## Filters vs smoothers

As a rule of thumb, use a `ivpsolvers.strategy_filter` strategy for `simulate_terminal_values`, 
a `ivpsolvers.strategy_smoother_fixedpoint` strategy for `solve_adaptive_save_at`,
and a `ivpsolvers.strategy_smoother_fixedinterval` strategy for `solve_fixed_step`.
Other combinations are possible, but rare.


## Calibration
Use a `solvers.solver_dynamic` solver if you expect that the output scale of your differential equation
solution varies greatly (eg for first-order, linear ODEs; see the tutorials).
Otherwise, use an `solvers.solver_mle` solver for plain simulation problems, 
and a `solvers.solver` for parameter-estimation.

## Miscellaneous
If you use a `ts0`, choose an `isotropic` factorisation instead of a `dense` factorisation.
They are mathematically equivalent, but the `isotropic` factorisation is faster.


## Future guidelines
These guidelines are a work in progress and may change at any point. If you have any input, reach out.
