# Transitioning from other packages

Here is how you get started with ``probdiffeq`` for solving initial value problems (IVPs) if you already have experience with other (ProbNum-)ODE solver packages in Python and Julia.

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


## Transitioning from Tornadox
[Tornadox](https://github.com/pnkraemer/tornadox) is a package that contains JAX implementations of probabilistic IVP solvers.
It has been used to solve [million-dimensional](https://arxiv.org/abs/2110.11812) IVPs.


`probdiffeq` is (more or less) a successor of `tornadox`: it can do everything that `tornadox` can do, but has more efficient solution routines (compiling whole solver-loops instead of single steps), more solvers, and more features built ``around'' IVP solutions: e.g. dense output and posterior sampling.

To reproduce tornadox' implementations use the following cheat sheet:

| `Tornadox`                            | `probdiffeq`                                 | Comments                                                   |
|---------------------------------------|----------------------------------------------|------------------------------------------------------------|
| `ek0.KroneckerEK0()`                  | `DynamicSolver(Filter(IsoTS0()))`            | Try other solvers or strategies, depending on the problem. |
| `ek0.DiagonalEK0()`                   | `DynamicSolver(Filter(BlockDiagTS0()))`      | See above.                                                 |
| `ek0.ReferenceEK0()`                  | `DynamicSolver(Filter(DenseTS0()))`          | See above.                                                 |
| `ek1.ReferenceEK1()`                  | `DynamicSolver(Filter(DenseTS1()))`          | See above.                                                 |
| `ek1.ReferenceEK1ConstantDiffusion()` | `MLESolver(Filter(DenseTS1()))`              | See above.                                                 |
| `ek1.DiagonalEK1()`                   | Work in progress.                            |                                                            |
| `solver.solve()`                      | `solve_with_native_python_loop(..., solver)` | Try `solve_and_save_at` instead.                           |
| `solver.simulate_final_state()`       | `simulate_terminal_values(..., solver)`      | `probdiffeq` compiles the whole loop (it uses jax.lax)     |
| `solver.solution_generator()`         | Work in progress.                            |                                                            |
| `init.TaylorMode()`                   | `taylor.taylor_mode_fn`                      |                                                            |
| `init.RungeKutta()`                   | `taylor.make_runge_kutta_starter()`          |                                                            |


Precise equivalences are tested for in consult `test/test_equivalences/test_tornadox_equivalence`.

## Transitioning from ProbNumDiffEq.jl

ProbNumDiffEq.jl is amazing and can do a few things we have not achieved yet.
But there are also some features that `probdiffeq` provides, which are not available in the Julia package.

Ignoring that `ProbNumDiffEq.jl` is written in Julia and that `probdiffeq` bases on JAX, the two packages are very similar.
Both packages should be similarly efficient for small(ish) problems. For high-dimensional ODEs, the state-space model factorisations in `probdiffeq` are expected to yield big advantages.

The feature-differences are that (at the time of this writing):
* `ProbNumDiffEq.jl` can solve mass-matrix problems, which `probdiffeq` does not yet do
* `ProbNumDiffEq.jl` allows callbacks, which `probdiffeq` does not yet do
* `probdiffeq` uses state-space model factorisations that are not yet implemented in ProbNumDiffEq.jl. These factorisations are important for high-dimensional problems.
* `probdiffeq` has a few methods that are not implemented in ProbNumDiffEq, e.g., statistical linearisation solvers.

To translate between the two packages, consider the following:
* Everything termed `EK0` or `EK1` in ProbNumDiffEq.jl is `TS0` or `TS1` in `probdiffeq` ("TS" stands for "Taylor series" linearisation).
* ProbNumDiffEq uses `DynamicDiffusion`, `FixedDiffusion`, `DynamicMVDiffusion`, or `FixedMVDiffusion`. Our respective solvers are `DynamicSolver()` or `MLESolver()`. Feed those with any implementations. Use a block-diagonal implementation (e.g. `BlockDiagTS0()`) for multivariate output scales. Try the `CalibrationFreeSolver()` with a manual (gradient-based?) calibration if the other routines are not satisfactory.
* ProbNumDiffEq switches between filtering and smoothing with a `smooth=true` flag. We use different strategies entirely, to simplify switching between both implementations for 
* Initialisation schemes from `ProbNumDiffEq` can be found in `probdiffeq/taylor.py`.




## Transitioning from ProbNum
[ProbNum](https://probnum.readthedocs.io/en/latest/) is a probabilistic numerics library in Python, just like `probdiffeq`.
ProbNum collects probabilistic solvers for all kinds of problems, not just ordinary differential equations.
It's API and documentation are more mature than the ones in `probdiffeq`.

The features (ignoring the non-IVP-related ones in ProbNum) differ as follows:
* ProbNum implements ODE solvers, filters and smoothers. At the moment, we only provide IVP solvers. The ability to seamlessly switch between ODE problems and filtering problems in ProbNum is quite cool, for example for [latent force models](https://arxiv.org/abs/2103.10153)
* The filtering and smoothing options in ProbNum are way beyond what `probdiffeq` aims to provide, e.g., we do not plan on offering particle filtering or nonlinear, continuous-time state-space models anytime soon.
* ProbNum offers perturbation-based solvers and callbacks
* The feature list in `probdiffeq`, focussing on state-space-model-based IVP solvers, is much bigger than what probnum offers. For example, we have more calibration techniques, more seamless switching between filtering and smoothing, different solving-modes, state-space model factorisations (!), and more modes of linearisation.

At the time of writing, ProbNum is NumPy based (JAX-backends may be coming soon), and `probdiffeq` is pure JAX.
This implies that we can take a few specialisations (mostly vmap- or  PyTree-centric ones) that lead to quite drastic gains in efficiency.
The solvers in ProbNum are amazing for their breadth and for didactic purposes, but `probdiffeq` is more efficient (check the benchmarks!).
Our solvers also naturally work with function transformations such as `vmap`, `jacfwd`, or `jit`, which is quite useful for combining the library with e.g. `optax` or `blackjax` (again, check the examples).



## Transitioning from Diffrax
Diffrax is a general-purpose ODE solving library written in JAX.
It is especially amazing because it unifies implementations of SDE and ODE solvers and has a growing selection of solver options.

Diffrax does not provide probabilistic numerical algorithms.
`probdiffeq` does not (aim to) cover SDE solvers, and the challenges in writing a probabilistic-IVP-solver library are different to the challenges of writing a general (mostly Runge-Kutta-based) ODE solver library.
More specifically:
* To build a solver in diffrax, it usually suffices to call e.g. `diffrax.Tsit5()`. In probdiffeq, we wrap a solver/calibration around an estimation strategy around a state-space model implementation. This is more involved, but in return offers more fine-tuning regarding the solver implementations, and all other advantages of probabilistic solvers over non-probabilistic ones.
* The vector fields in diffrax are `diffrax.ODETerm()`s (presumably, because of the joint treatment of ODEs/SDEs; in `probdiffeq`, we pass simple callables because we only care about ODEs.
* Diffrax offers multiple modes of differentiating the IVP solver. For probabilistic algorithms, this is work in progress.


## General divergences from non-probabilistic solver libraries (e.g. jax.odeint, or SciPy)
* Building a solver in `probdiffeq` is more involved than in e.g. Scipy or Diffrax, because it consists of 
* `probdiffeq` offers different solution methods: `simulate_terminal_values()`, `solve_with_native_python_loop()`, or `solve_and_save_at()`.
  Not only does this lead to simple code in each respective solution routine, it also allows matching the solver to the routine.
  For example, `simulate_terminal_values()` should only ever be used with filters.
* 

## Choosing a solver
Good solvers are problem-dependent. Nevertheless, some guidelines exist:
* If your problem is high-dimensional, use an implementation based on Kronecker-factorisation or block-diagonal covariances. Only those scale as O(d), for d-dimensional problems.
* If your problem is stiff, use a first-order solver (DenseTS1, DenseSLR1, etc.) instead of zeroth-order solvers. Be careful with state-space model factorisations
* Use a `Filter` strategy for `simulate_terminal_values()`, a smoother strategy for `solve_with_native_python_loop()`, and a `FixedPointSmoother` for `solve_and_save_at()`.
* Use `DynamicSolver()` if you expect that the output scale of your IVP solution varies greatly. Otherwise, choose a `MLESolver()`. Try a `CalibrationFreeSolver()` for parameter-inference prblems.
* 