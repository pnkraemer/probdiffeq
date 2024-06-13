# Transitioning from other packages

Here is how you get started with ProbDiffEq for solving ordinary differential equations (ODEs) 
if you already have experience with other (probabilistic) ODE solver packages in Python and Julia.

The most similar packages to ProbDiffEq are

* [Tornadox](https://github.com/pnkraemer/tornadox)
* [ProbNumDiffEq.jl](https://nathanaelbosch.github.io/ProbNumDiffEq.jl/stable/)
* [ProbNum](https://probnum.readthedocs.io/en/latest/)

We will explain the differences below.

We will also cover differences to  

* [Diffrax](https://docs.kidger.site/diffrax/)
* Other non-probabilistic IVP solver libraries, e.g., [SciPy](https://scipy.org/). 

Before starting, credit is due: ProbDiffEq draws much inspiration from those code bases; without them, it wouldn't exist.

## Transitioning from Tornadox

[Tornadox](https://github.com/pnkraemer/tornadox) is a package that contains JAX implementations of probabilistic IVP solvers.
It has been used, for instance, to solve [million-dimensional](https://arxiv.org/abs/2110.11812) differential equations.

ProbDiffEq is (more or less) a successor of Tornadox: it can do almost everything that Tornadox can do, but is generally faster (compiling entire solver loops instead of only single steps), offers more solvers, and provides more features built ``around'' IVP solutions: e.g. dense output or posterior sampling.
ProbDiffEq is also more thoroughly tested and documented, and has a few features that are not yet (that is, at the time of writing this document) implemented in Tornadox.

ProbDiffEq can reproduce most of the implementations in Tornadox:

| In Tornadox:                        | In ProbDiffEq:                                    | Comments                                                                                                                                                         |
|---------------------------------------|---------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ek0.KroneckerEK0()`                  | `dynamic(strategy_filter(ibm_adaptive(), ts0()))` | Combine with `impl.select("isotropic", ...)`                                                                                                                     |
| `ek0.DiagonalEK0()`                   | `dynamic(strategy_filter(ibm_adaptive(), ts0()))` | Combine with `impl.select("blockdiag", ...)`                                                                                                                     |
| `ek0.ReferenceEK0()`                  | `dynamic(strategy_filter(ibm_adaptive(), ts0()))` | Combine with `impl.select("dense", ...)`                                                                                                                         |
| `ek1.ReferenceEK1()`                  | `dynamic(strategy_filter(ibm_adaptive(), ts1()))` | Combine with `impl.select("dense", ...)`                                                                                                                         |
| `ek1.ReferenceEK1ConstantDiffusion()` | `mle(strategy_filter(ibm_adaptive(), ts1()))`     | Combine with `impl.select("dense", ...)`.                                                                                                                        |
| `ek1.DiagonalEK1()`                   | Work in progress.                                 |                                                                                                                                                                  |
| `solver.solve()`                      | `solve_adaptive_save_every_step()`                     | Try `solve_adaptive_save_at()` instead.                                                                                                                               |
| `solver.simulate_final_state()`       | `simulate_terminal_values()`                      | ProbDiffEq compiles the whole loop; it will be much faster.                                                                                                      |
| `solver.solution_generator()`         | Work in progress.                                 |                                                                                                                                                                  |
| `init.TaylorMode()`                   | `taylor.taylor.taylor_mode`                     | Consider `taylor.taylor.odejet_via_jvp()` for low numbers of derivatives and `taylor.taylor.odejet_doubling_unroll()` for (absurdly) high numbers of derivatives |
| `init.RungeKutta()`                   | `taylor.taylor.runge_kutta_starter()`         |                                                                                                                                                                  |


Recently, the development of Tornadox has not been very active, so its API is relatively stable.
In contrast, APIs in ProbDiffEq may change.
Other than that, you may safely replace Tornadox with ProbDiffEq, 
because it is faster, more thoroughly documented, and has more features.

## Transitioning from ProbNumDiffEq.jl

ProbNumDiffEq.jl has been around for a while now (and successfully so), embeds into the `SciML` ecosystem, and can do a few things we have yet to achieve.
It might be the most performant probabilistic-IVP-solver library to date.
But there are some neat little features that ProbDiffEq provides that are unavailable in ProbNumDiffEq.jl.

Ignoring that ProbNumDiffEq.jl is written in Julia and that ProbDiffEq builds on JAX, the two packages are very similar.
Both packages should be more or less equally efficient for small(ish) problems. 
For high-dimensional differential equations, the state-space model factorisations in ProbDiffEq are expected to yield significant advantages. 
(Benchmarks incoming?).

The most apparent feature differences between ProbNumDiffEq.jl and ProbDiffEq are (at the time of this writing):

* ProbNumDiffEq.jl can solve mass-matrix problems, which ProbDiffEq does not yet do.
* ProbNumDiffEq.jl allows callbacks, which ProbDiffEq does not yet do.
* ProbDiffEq uses state-space model factorisations not yet implemented in ProbNumDiffEq.jl. These factorisations are crucial for high-dimensional problems.
* ProbDiffEq has a few methods that are not in ProbNumDiffEq.jl, e.g., statistical linearisation solvers.

Both packages are still evolving, so this list may not remain up-to-date. When in doubt, consult each package's API documentation.

To translate between the two packages, consider the following:

* Everything termed `EK0` or `EK1` in ProbNumDiffEq.jl is `ts0` or `ts1` in ProbDiffEq ("ts" stands for "Taylor series" linearisation and stands in contrast to "slr", i.e. "statistical linear regression").
* ProbNumDiffEq.jl calibrates output scales via `DynamicDiffusion`, `FixedDiffusion`, `DynamicMVDiffusion`, or `FixedMVDiffusion`. 
 Their equivalents in ProbDiffEq are `ivpsolvers.solver_dynamic()` or `ivpsolvers.solver_mle()`. Feed them with any strategies (Filters/Smoothers) and any state-space model implementations. Use a block-diagonal implementation (e.g. `impl.choose("blockdiag")`) for multivariate output scales ("`MVDiffusion`"). Try the `ivpsolvers.solver()` with a manual (gradient-based?) calibration if the other routines are unsatisfactory.
* ProbNumDiffEq.jl refers to `ibm_adaptive(output_scale=x)` as `IWP(diffusion=x^2)`. They are the same processes. 
* ProbNumDiffEq.jl switches between filtering and smoothing with a `smooth=true/false` flag. ProbDiffEq uses different strategies to distinguish these strategies and offers a third one (fixedpoint-smoothing). 
* Initialisation schemes like those in `ProbNumDiffEq` are in `probdiffeq/taylor/*.py`. ProbDiffEq offers some rules for high-order differential equations and some unique methods (e.g. doubling). But the feature lists are relatively similar.
* The features in [Fenrir.jl](https://github.com/nathanaelbosch/Fenrir.jl), which extends ProbNumDiffEq.jl, should be more or less readily available via `probdiffeq/solution.py`. Check out the tutorial notebooks!

Should I replace ProbNumDiffEq.jl with ProbDiffEq?
Short answer: No. 
Long answer: Use ProbNumDiffEq.jl in Julia and ProbDiffEq in JAX. Use the Julia code for funky problems like mass-matrix IVPs, and use ProbDiffEq for high-dimensional differential equations (or if you need statistical linearisation).



## Transitioning from ProbNum

[ProbNum](https://probnum.readthedocs.io/en/latest/) is a probabilistic numerics library in Python, just like ProbDiffEq.
ProbNum collects probabilistic solvers for many problems, not just ordinary differential equations.
Its API and documentation are more mature than the API and documentation in ProbDiffEq.
That said, ProbDiffEq specialises in pure JAX and state-space-model-based IVP solvers, which leads to significant efficiency gains.


The features (ignoring the non-IVP-related features in ProbNum, such as linear solvers or numerical integration) 
differ between ProbDiffEq and ProbNum as follows:

* ProbNum implements IVP solvers, and general continuous/discrete filters and smoothers. 
  At the moment, ProbDiffEq's support for discrete-time filtering and smoothing is very limited. 
  The ability to seamlessly switch between ODE problems and filtering problems in ProbNum is helpful, for example, to build [latent force models](https://arxiv.org/abs/2103.10153).
* The filtering and smoothing options in ProbNum are broader than what ProbDiffEq needs;
  e.g., ProbNum offers particle filtering or nonlinear, continuous-time state-space models.
* ProbNum offers perturbation-based solvers, which ProbDiffEq does not provide.
* ProbNum offers callbacks, which are not yet supported in ProbDiffEq

In contrast:
* ProbDiffEq offers state-space model factorisations, different modes of linearisation, 
  more output-scale calibration routines, more estimation strategies, 
  and different modes of solving.
* ProbDiffeq computes IVP solutions faster.
* ProbDiffEq is compatible with everything that builds on JAX.


At the time of writing, ProbNum is NumPy-based (but a [JAX-backend has been in development](https://github.com/probabilistic-numerics/probnum/pull/581)), and ProbDiffEq is pure JAX.
Using JAX implies we can take a few specialisations (mostly vmap- or  PyTree-centric ones) that lead to drastic efficiency gains, but ProbDiffEq will generally not be compatible with non-JAX based code.

In other words, the solvers in ProbNum are excellent for their breadth and for didactic purposes, but ProbDiffEq is more efficient in what it provides (check the benchmarks!).
Our solvers also naturally work with function transformations such as `vmap`, `jacfwd`, or `jit`, which is quite helpful for combining ProbDiffEq with, e.g. `optax` or `blackjax` (check the examples).



## Transitioning from Diffrax
[Diffrax](https://docs.kidger.site/diffrax/) is a JAX-based library offering numerical differential equation solvers.
One of its big selling points is that it unifies implementations of solvers for SDEs, CDEs, and ODEs.

The main difference between ProbDiffEq and Diffrax is that Diffrax provides non-probabilistic ODE solvers whereas ProbDiffEq provides probabilistic solvers.
Both solve differential equations, but they can only be compared to a certain extent:

Yes, both classes of algorithms solve differential equations and can (and will) be part of the same benchmarks.
But the sets of methods provided by each package are completely disjoint, and the choice between both toolboxes reduces to the choice between non-probabilistic and probabilistic algorithms.
(When to choose which one is a subject for another document; some selling points of probabilistic solvers are discussed in the example notebooks.)

The main API differences between the packages are the following:

* To build a solver in Diffrax, it usually suffices to call, e.g. `diffrax.Tsit5()`. 
  In ProbDiffEq, constructing a solver is more involved (which is not necessarily a drawback; check the quickstart). 
* The vector fields in Diffrax are `diffrax.ODETerm()`s (presumably, because of the joint treatment of ODEs/SDEs); 
  in ProbDiffEq, we pass plain functions with signature `(*ys, t)`.
* Diffrax offers multiple modes of differentiating the IVP solver. For probabilistic solvers, all but what JAX natively provides is a work in progress.


To roughly translate the Diffrax IVP solvers to ProbDiffEq solvers, consider the following selection of solvers:

| If you use the following solvers in Diffrax:              | You might like the following solvers in ProbDiffEq:                                                                     | Comments                                                                               | 
|-----------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| `Heun()`, `Midpoint()`, `Ralston()`, `LeapfrogMidpoint()` | e.g. `ibm_adaptive(num_derivatives=1)`, `ts0()` with an `isotropic` or `blockdiag` implementation                       | Use a block-diagonal factorisation if the ODE dimensions have greatly different scales |
| `Bosh3()`                                                 | increase `num_derivatives` to `num_derivatives=2` in the above                                                          | See above.                                                                             |
| `Tsit5()`, `Dopri5()`                                     | increase `num_derivatives` to `num_derivatives=4` in the above                                                          | See above.                                                                             |
| `Dopri8()`                                                | increase `num_derivatives` to `num_derivatives={5,6,7}` in the above. If this is inefficient, try a `ts1()` correction. | See above.                                                                             |
| `Kvaerno3()`                                              | use `num_derivatives=2` and a `ts1()` correction.                                                                       | See above.                                                                             |
| `Kvaerno4()`                                              | use `num_derivatives=3` and a `ts1()` correction.                                                                       | See above.                                                                             |
| `Kvaerno5()`                                              | use `num_derivatives=4` and a `ts1()` correction.                                                                       | See above.                                                                             |
| Symplectic methods                                        | Work in progress.                                                                                                       |                                                                                        |
| Reversible methods                                        | Work in progress.                                                                                                       |                                                                                        |




## General divergences from other non-probabilistic solver libraries (e.g. jax.odeint or SciPy)
Most of the divergences from Diffrax apply. 
Additionally:

* Solution objects in ProbDiffEq are random processes (posterior distributions). Random variable types replace most vectors and matrices. This statistical description is richer than a point estimate but needs to be calibrated and demands a non-trivial interaction with the solution (e.g. via sampling from it instead of simply plotting the point-estimate)
* ProbDiffEq offers different solution methods: `simulate_terminal_values()`, `solve_adaptive_save_every_step()`, or `solve_adaptive_save_at()`. Many conventional ODE solver suites expose this functionality through flags in a single `solve` function. 
Expressing different modes of solving differential equations in different functions almost exclusively affects the source-code simplicity; but it also allows matching the solver to the solving mode (e.g., terminal values vs save-at). For example, `simulate_terminal_values()` is best combined with a filter.
