# Choose the right probabilistic solver

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
See also the output scale recommendations under "Prior distributions".


## Prior distributions
If you're uncertain which prior to choose, prefer an integrated Wiener process over more advanced priors.
The reason is that integrated Wiener processes have closed form transition parameters, which makes
simulations much faster. For other cases, use exponential priors according to the recommendations
in https://arxiv.org/abs/2305.14978.

Regarding output scales: 
if the ODE states carry different magnitudes (eg in the Robertson problem, where two states 
are O(1) and the third one is O($10^{-5}$)), a dedicated output scale when constructing the 
prior makes sense. Consult the DAE tutorials for specific information.

Regarding the number of Taylor coefficients: assuming the ODE solution is smooth, then
more Taylor coefficients increase the convergence *rate* but also increase the complexity
per step and the requirements on numerical robustness. When in doubt, use 4-5 Taylor coefficients,
or 7-8 Taylor coefficients if the goal is to achieve accuracy close to machine precision.
In single precision (eg on a GPU), track only 2-3 Taylor coefficients.


## Miscellaneous
If you use a `ts0`, choose an `isotropic` factorisation instead of a `dense` factorisation.
They are mathematically equivalent, but the `isotropic` factorisation is faster.

For parameter estimation problems with adaptive solvers, replace Probdiffeq's while-loops
with Equinox's while-loops; see the tutorials for how.

## Future guidelines
These guidelines are a work in progress and may change at any point. If you have any input, reach out.
Something missing? Reach out!
