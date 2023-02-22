# Transitioning from other packages

Here is how you get started with ``probdiffeq`` for solving initial value problems (IVPs) 
if you already have experience with other (ProbNum-)ODE solver packages in Python and Julia.

The most similar packages to ``probdiffeq`` are

* Tornadox
* ProbNumDiffEq.jl
* ProbNum

and we will explain differences below.


We will also cover transitioning from  

* Diffrax
* General non-probabilistic IVP solver libraries, e.g., SciPy. 


Before starting, credit is due:
``probdiffeq`` draws a lot of inspiration from those code bases; without them,
it wouldn't exist.

Let's dive in.


## Transitioning from Tornadox
[Tornadox](https://github.com/pnkraemer/tornadox) is a package that contains JAX implementations 
of probabilistic IVP solvers.
It has been used, for instance, to solve 
[million-dimensional](https://arxiv.org/abs/2110.11812) differential equations.

`probdiffeq` is (more or less) a successor of `tornadox`: 
it can do almost everything that `tornadox` can do, 
but is generally faster (compiling entire solver-loops instead of only single steps), 
offers more solvers, and provides more features built ``around'' IVP solutions: 
e.g. dense output or posterior sampling.
`probdiffeq` is also more thoroughly tested, and has a few example notebooks that are not
yet (at the time of writing this) implemented in `tornadox`.

Most of `tornadox`' implementations can be reproduced with `probdiffeq`:

| In `tornadox`:                        | In `probdiffeq`:                                    | Comments                                                                                                                                    |
|---------------------------------------|-----------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| `ek0.KroneckerEK0()`                  | `DynamicSolver(Filter(IsoTS0()))`                   | Try other solvers or strategies, depending on the problem.                                                                                  |
| `ek0.DiagonalEK0()`                   | `DynamicSolver(Filter(BlockDiagTS0()))`             | See above.                                                                                                                                  |
| `ek0.ReferenceEK0()`                  | `DynamicSolver(Filter(DenseTS0()))`                 | See above.                                                                                                                                  |
| `ek1.ReferenceEK1()`                  | `DynamicSolver(Filter(DenseTS1()))`                 | See above.                                                                                                                                  |
| `ek1.ReferenceEK1ConstantDiffusion()` | `MLESolver(Filter(DenseTS1()))`                     | See above.                                                                                                                                  |
| `ek1.DiagonalEK1()`                   | Work in progress.                                   |                                                                                                                                             |
| `solver.solve()`                      | `solve_with_native_python_loop(..., solver=solver)` | Try `solve_and_save_at()` instead.                                                                                                          |
| `solver.simulate_final_state()`       | `simulate_terminal_values(..., solver=solver)`      | `probdiffeq` compiles the whole loop (it uses `jax.lax.while_loop()`)                                                                       |
| `solver.solution_generator()`         | Work in progress.                                   |                                                                                                                                             |
| `init.TaylorMode()`                   | `taylor.taylor_mode_fn`                             | Consider `taylor.forward_mode_fn()` for low numbers of derivatives and `taylor_mode_doubling_fn()` for (absurdly) high numbers of derivatives |
| `init.RungeKutta()`                   | `taylor.make_runge_kutta_starter()`                 |                                                                                                                                             |


Precise equivalences are tested for in consult `test/test_equivalences/test_tornadox_equivalence.py`.


`tornadox` is not very active at the moment, so its API is fairly stable.
In contrast, APIs in `probdiffeq` may change.
Other than that, you may safely replace `tornadox` with `probdiffeq`, 
because it is faster, more thoroughly documented, and has more features.



## Transitioning from ProbNumDiffEq.jl

`ProbNumDiffEq.jl` has been around for a while now (and successfully so), embeds into the `SciML` ecosystem, 
and can do a few things we have not achieved yet.
It might be the most performant probabilistic-IVP-solver library to date.
But there are some neat little features that `probdiffeq` provides that are not available in `ProbNumDiffEq.jl`.


Ignoring that `ProbNumDiffEq.jl` is written in Julia and that `probdiffeq` bases on JAX, the two packages are very similar.
Both packages should be more or less equally efficient for small(ish) problems. 
For high-dimensional differential equations, the state-space model factorisations in `probdiffeq` 
are expected to yield big advantages. (Benchmarks incoming?).

The most obvious feature differences between `ProbNumDiffEq.jl` and `probdiffeq` are (at the time of this writing):

* `ProbNumDiffEq.jl` can solve mass-matrix problems, which `probdiffeq` does not yet do
* `ProbNumDiffEq.jl` allows callbacks, which `probdiffeq` does not yet do
* `probdiffeq` uses state-space model factorisations that are not yet implemented in ProbNumDiffEq.jl. 
  These factorisations are crucial for high-dimensional problems.
* `probdiffeq` has a few methods that are not implemented in ProbNumDiffEq, e.g., statistical linearisation solvers.

Both packages are still evolving, so this list may not remain up-to-date for long.

To translate between the two packages, consider the following:

* Everything termed `EK0` or `EK1` in`ProbNumDiffEq.jl`is `TS0` or `TS1` in `probdiffeq` 
  ("TS" stands for "Taylor series" linearisation and stands in contrast to "SLR", i.e. "statistical linear regression").
* `ProbNumDiffEq.jl` calibrates output scales via `DynamicDiffusion`, `FixedDiffusion`, `DynamicMVDiffusion`, or `FixedMVDiffusion`. 
  Their equivalents in `probdiffeq` are `DynamicSolver()` or `MLESolver()`. 
  Feed them with any strategies (Filters/Smoothers) and any state-space model implementations. 
  Use a block-diagonal implementation (e.g. `BlockDiagTS0()`) for multivariate output scales ("`MVDiffusion`"). 
  Try the `CalibrationFreeSolver()` with a manual (gradient-based?) calibration if the other routines are not satisfactory.
* `ProbNumDiffEq.jl` refers to `IBM(output_scale_sqrtm=x)` as `IWP(diffusion=x)`. 
  They are the same processes. 
* `ProbNumDiffEq.jl` switches between filtering and smoothing with a `smooth=true/false` flag. 
  We use different strategies to distinguish between those, because this way it becomes easier 
  to cache reusable quantities for the smoother. 
* Initialisation schemes from `ProbNumDiffEq` can be found in `probdiffeq/taylor.py`. 
  `probdiffeq` offers some rules for high-order differential equations and some quite niche methods 
  (e.g. doubling). But the feature lists are fairly similar.
* The features in [`Fenrir`](https://github.com/nathanaelbosch/Fenrir.jl), which bases on `ProbNumDiffEq.jl`, should be 
  more or less readily available via `probdiffeq/dense_output.py`. Check out the tutorial notebooks!


Should I replace `ProbNumDiffeq.jl` with `probdiffeq`?
Short answer: No. 
Long answer: Use `ProbNumDiffeq.jl` in Julia and `probdiffeq` in JAX. Use the Julia code for funky problems like mass-matrix IVPs, and use `probdiffeq` for high-dimensional problems (or if you need statistical linearisation).



## Transitioning from ProbNum
[ProbNum](https://probnum.readthedocs.io/en/latest/) is a probabilistic numerics library in Python, just like `probdiffeq`.
`ProbNum` collects probabilistic solvers for all kinds of problems, not just ordinary differential equations.
It's API and documentation are more mature than API and documentation in `probdiffeq`.
That said, `probdiffeq` specialises on JAX and on state-space-model based IVP solvers, which leads to some pretty large efficiency gains.


The features (ignoring the non-IVP-related features in ProbNum, such as linear solvers or numerical integration) 
differ between `probdiffeq` and `ProbNum` as follows:

* `ProbNum` implements IVP solvers, filters and smoothers. 
  At the moment, we only provide IVP solvers. 
  The ability to seamlessly switch between ODE problems and filtering problems in `ProbNum` is quite cool, 
  for example, to build [latent force models](https://arxiv.org/abs/2103.10153).
* The filtering and smoothing options in `ProbNum` are broader than what `probdiffeq` needs;
  e.g., `ProbNum` offers particle filtering or nonlinear, continuous-time state-space models.
* `ProbNum` offers perturbation-based solvers.
* `ProbNum` offers callbacks
* `probdiffeq` offers state-space model factorisations, differnet modes f linearisation, more output-scale calibration routines, more estimation strategies, and different modes of solving.
* `probdiffeq` is compatible with everything JAX.


At the time of writing, `ProbNum` is NumPy-based (but a [JAX-backend may be coming soon](https://github.com/probabilistic-numerics/probnum/pull/581)), and `probdiffeq` is pure JAX.
This implies that we can take a few specialisations (mostly vmap- or  PyTree-centric ones) that lead to drastic efficiency gains in efficiency.

In other words, the solvers in `ProbNum` are amazing for their breadth and for didactic purposes, but `probdiffeq` is more efficient in what it provides (check the benchmarks!).
Our solvers also naturally work with function transformations such as `vmap`, `jacfwd`, or `jit`, which is quite useful for combining `probdiffeq` with e.g. `optax` or `blackjax` (check the examples).



## Transitioning from Diffrax
[Diffrax](https://docs.kidger.site/diffrax/) is a JAX-based library offering numerical differential equation solvers.
It is especially interesting because it unifies implementations of solvers for SDEs, CDEs, and ODEs.
But `diffrax` does not provide probabilistic numerical algorithms.

In fact, both libraries (`diffrax` and `probdiffeq`) do not really serve the same purpose, and no one is force to use either or.
Nevertheless, they both solve differential equations and can (and will) be benchmarked against one another.


The major differences are the following:

* To build a solver in `diffrax`, it usually suffices to call e.g. `diffrax.Tsit5()`. 
  In `probdiffeq`, we wrap a solver around an estimation strategy around a state-space model. 
  This is more involved, but in return offers more fine-tuning regarding the solver implementations, and all other advantages of probabilistic solvers over non-probabilistic ones.
* The vector fields in `diffrax` are `diffrax.ODETerm()`s (presumably, because of the joint treatment of ODEs/SDEs); 
  in `probdiffeq`, we pass plain functions.
* `diffrax` offers multiple modes of differentiating the IVP solver. For probabilistic solvers, this is work in progress.


To roughly translate the `diffrax` IVP solvers to `probdiffeq` solvers, consider the following selection of solvers:

| In `diffrax`:                                             | In `probdiffeq`:                                                                              | Comments                                                                       | 
|-----------------------------------------------------------|-----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| `Heun()`, `Midpoint()`, `Ralston()`, `LeapfrogMidpoint()` | e.g. `IsoTS0(num_derivatives=1)`, `BlockDiagTS0(num_derivatives=1)`                           | Use block-diagonal covariances if the ODE dimensions have greatly different scales |
| `Bosh3()`                                                 | e.g. `IsoTS0(num_derivatives=2)`, `BlockDiagTS0(num_derivatives=2)`                           | See above.                                                                     |
| `Tsit5()`, `Dopri5()`                                     | e.g. `IsoTS0(num_derivatives=4)`, `BlockDiagTS0(num_derivatives=4)`                           | See above.                                                                     |
| `Dopri8()`                                                | e.g. `IsoTS0(num_derivatives=4)`. If this converges slowly, try `DenseTS1(num_derivatives=7)` | See above.                                                                     |
| `Kvaerno3()`                                              | e.g. `DenseTS1(num_derivatives=2)`, `DenseSLR1(num_derivatives=2)`                            | See above.                                                                     |
| `Kvaerno4()`                                              | e.g. `DenseTS1(num_derivatives=3)`, `DenseSLR1(num_derivatives=3)`                            | See above.                                                                     |
| `Kvaerno5()`                                              | e.g. `DenseTS1(num_derivatives=4)`, `DenseSLR1(num_derivatives=4)`                            | See above.                                                                     |
| Symplectic methods                                        | Work in progress.                                                                             |                                                                          |
| Reversible methods                                        | Work in progress.                                                                             |                                                                      |




## General divergences from other non-probabilistic solver libraries (e.g. jax.odeint or SciPy)
Most of the divergences from Diffrax apply. 
Additionally:

* Solution objects in `probdiffeq` are random processes (posterior distributions). 
  Arrays are usually replaced by random variables. This uncertainty is quite rich, but needs to be calibrated (e.g. by using an `MLESolver()` instead of `CalibrationFreeSolver()`)
* `probdiffeq` offers different solution methods: `simulate_terminal_values()`, `solve_with_native_python_loop()`, or `solve_and_save_at()`.
  Not only does this lead to simple code in each respective solution routine, it also allows matching the solver to the routine.
  For example, `simulate_terminal_values()` is best combined with a filter.


## Choosing a solver
Good solvers are problem-dependent. Nevertheless, some guidelines exist:

* If your problem is high-dimensional, use an implementation based on Kronecker-factorisation or block-diagonal covariances. 
  Only those scale as O(d), for d-dimensional problems. `DenseTS1()` and `DenseSLR1()` cost O(d^3) per step.
* Use `IsoTS0()` or `BlockDiagTS0()` instead of `DenseTS0()`
* If your problem is stiff, use a solver with first-order linearisation (DenseTS1, DenseSLR1, etc.) 
  instead of one with zeroth-order linearisation. Try to avoid too extreme state-space model factorisations (e.g. `Iso*` or `BlockDiag*`)
* Almost always, use a `Filter` strategy for `simulate_terminal_values()`, 
  a smoother strategy for `solve_with_native_python_loop()`, 
  and a `FixedPointSmoother` strategy for `solve_and_save_at()`.
* Use `DynamicSolver()` if you expect that the output scale of your IVP solution varies greatly. 
  Otherwise, choose a `MLESolver()`. Try a `CalibrationFreeSolver()` for parameter-inference problems.

These guidelines are a work in progress and may change soon. If you have input, let us know!