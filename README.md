# Probabilistic solvers in JAX

[![CI](https://github.com/pnkraemer/probdiffeq/workflows/ci/badge.svg)](https://github.com/pnkraemer/probdiffeq/actions)
[![PyPI version](https://img.shields.io/pypi/v/probdiffeq.svg)](https://pypi.python.org/pypi/probdiffeq)
[![License](https://img.shields.io/pypi/l/probdiffeq.svg)](https://pypi.python.org/pypi/probdiffeq)
[![Python versions](https://img.shields.io/pypi/pyversions/probdiffeq.svg)](https://pypi.python.org/pypi/probdiffeq)




**Probdiffeq** implements adaptive probabilistic numerical solvers for differential equations (ODEs). It builds on [JAX](https://jax.readthedocs.io/en/latest/), thus inheriting **automatic differentiation**, **vectorisation**, and **GPU acceleration**.



> ⚠️ Probdiffeq is an active research project. Expect rough edges and sudden API changes.


**Features:**

- ⚡ Calibration and step-size adaptation  
- ⚡ Stable implementations of filtering, smoothing, and other estimation strategies  
- ⚡ Custom information operators, dense output, posterior sampling, and prior distributions.
- ⚡ State-space model factorisations  
- ⚡ Parameter estimation
- ⚡ Taylor-series estimation with and without jets  
- ⚡ Seamless interoperability with [Optax](https://optax.readthedocs.io/en/latest/index.html), [BlackJAX](https://blackjax-devs.github.io/blackjax/), and other JAX-based libraries  
- ⚡ Numerous examples (basic and advanced) -- see the [documentation](https://pnkraemer.github.io/probdiffeq/)  




**Contributing:** Contributions are very welcome!  

- Browse open issues (look for “good first issue”).  
- Check the developer documentation.  
- Open an issue for feature requests or ideas.  



**Related projects:**

- [Tornadox](https://github.com/pnkraemer/tornadox): One of Probdiffeq's precursors.
- [ProbNumDiffEq.jl](https://nathanaelbosch.github.io/ProbNumDiffEq.jl/stable/): A similar library in Julia
- [ProbNum](https://probnum.readthedocs.io/en/latest/): Probabilistic numerics in Numpy.

The docs include guidance on migrating from these packages. Missing something? Open an issue or pull request!



**You might also like:**

- [diffeqzoo](https://diffeqzoo.readthedocs.io/en/latest/): reference implementations of differential equations in NumPy and JAX  
- [probfindiff](https://probfindiff.readthedocs.io/en/latest/): probabilistic finite-difference methods in JAX  


## Installation

Install the latest release from PyPI:

```bash
pip install probdiffeq
```

This assumes [JAX](https://jax.readthedocs.io/en/latest/) is already installed.  

To install with JAX (CPU backend):  

```bash
pip install probdiffeq[cpu]
```



**Versioning**

Probdiffeq follows semantic versioning via **0.MINOR.PATCH**:

- **PATCH**: increase with bugfixes & new features  
- **MINOR**: increase with breaking changes  

See [semantic versioning](https://semver.org/).
Notably, Probdiffeq's API is not guaranteed to be stable, but we do our best to follow the versioning scheme so that downstream projects remain reproducible.




## Choose the right solver

Good solvers are problem-dependent. However, some guidelines exist:

### State-space model factorisation

* If your problem is scalar-valued (`shape=()`), use a dense factorisation. All factorisations have the same complexity for scalar models, but dense factorisations have the most solvers implemented.

* If your problem is vector-valued, be aware that different implementation choices imply different modelling choices.
However, if you don't care too much about modelling choices:

* If your problem is high-dimensional, use a `blockdiag` or `isotropic` implementation.
* If your problem is medium-dimensional, use any implementations. 
  `isotropic` factorisations tend to be the fastest with the worst UQ and worst stability, 
  `dense` factorisations tend to be the slowest with the best UQ and best stability, 
  `blockdiag` factorisations are somewhere in between.


### Stiffness

* If your problem is stiff, use a a `dense` implementation in combination with a
correction scheme that employs first-order linearisation; for instance, `ts1` or `slr1`.
These first-order methods, if used in conjunction with an integrated Wiener process prior,
are known to be $A$-stable:

> Tronarp, F., Kersting, H., Särkkä, S., & Hennig, P. (2019). 
Probabilistic solutions to ordinary differential equations as nonlinear Bayesian filtering: 
a new perspective. Statistics and Computing, 29(6), 1297-1315.

* If your problem is a discretised semilinear PDE, try an integrated Ornstein-Uhlenbeck prior.
The combination of IOUP + first-order linearisation is $L$-stable (thus also $A$-stable):

> Bosch, N., Hennig, P., & Tronarp, F. (2023). Probabilistic exponential integrators. 
Advances in Neural Information Processing Systems, 36, 40450-40467.

Often, IWP priors are still more effective than IOUP priors because the transition parameters
are considerably cheaper to compute. Still, good priors matter.

* Zeroth-order approximation (`ts0`, `slr0`) is neither $A$- nor $L$-stable, so do not use them for
stiff problems. Relatedly, solvers in isotropic and blockdiagonal factorisations are also
not $A$- nor $L$-stable, even if one uses IOUPs or first-order linearisation. Though for only 
mildly stiff problems, like the Brusselator perhaps, blockdiagonal factorisations in combination
with first-order linearisation may still deliver acceptable performance.

* If your problem is stiff **and** high-dimensional: 
Probabilistic solvers for problems that are stiff and high-dimensional are a bit of an open problem
as of writing this. Try combining blockdiagonal factorisations with first-order linearisation,
which may yield satisfactory results, but do not expect to outperform non-probabilistic solvers
for stiff high-dimensional equation (eg those that exploit sparsity in Jacobians).


### Filters vs smoothers
As a rule of thumb:

* Use a filter strategy for simulating terminal values
* Use a fixed-point smoother for solving via the `save_at` functionality 
* Use a fixed-interval smoother for fixed steps. 

Other combinations are possible, but typically rare.


### Calibration

* Use a dynamic solver if you expect that the output scale of your differential equation
solution varies greatly (eg for first-order, linear ODEs; see the examples)
* Otherwise, use an MLE-based solver solver for typical simulation problems 
* Use a solver without automatic calibration for parameter-estimation.

See also the output-scale recommendations under "Prior distributions".


### Prior distributions
If you're uncertain which prior to choose, prefer an integrated Wiener process over more advanced priors.
The reason is that integrated Wiener processes have closed form transition parameters, which makes
simulations much faster. For other cases, use exponential priors according to the recommendations
in https://arxiv.org/abs/2305.14978.

Regarding output scales: 
if the ODE states carry different magnitudes (eg in the Robertson problem, where two states 
are O(1) and the third one is O($10^{-5}$)), a dedicated output scale when constructing the 
prior makes sense. Consult the DAE examples for specific information.


## Number of Taylor coefficients (''order'')

Regarding the number of Taylor coefficients: assuming the ODE solution is smooth, then
more Taylor coefficients increase the convergence *rate*:

> Tronarp, F., Särkkä, S., & Hennig, P. (2021). Bayesian ODE solvers: the maximum a posteriori estimate. Statistics and Computing, 31(3), 23.

> Kersting, H., Sullivan, T. J., & Hennig, P. (2020). Convergence rates of Gaussian ODE filters. Statistics and computing, 30(6), 1791-1816.

However, more coefficients also increase the complexity per step 
and the requirements on numerical robustness:

> Kraemer, N., & Hennig, P. (2024). Stable implementation of probabilistic ODE solvers. Journal of Machine Learning Research, 25(111), 1-29.

When in doubt, use 4-5 Taylor coefficients for most problems.
If if the goal is to achieve accuracy close to machine precision, use 7-8 Taylor coefficients.
In low precision (e.g. on a GPU), use 2-3 Taylor coefficients.

For reference, the posterior mean of some probabilistic solvers coincides with non-probabilistic ODE simulators:

- 


### Miscellaneous
If you use a zero-th order method, choose an `isotropic` factorisation instead of a `dense` factorisation.
They are mathematically equivalent, but the `isotropic` factorisation is faster.

For parameter estimation problems with adaptive solvers, replace Probdiffeq's while-loops
with Equinox's while-loops; see the examples for how.


### Future guidelines
These guidelines are a work in progress and may change at any point. If you have any input, reach out.
Something missing? Reach out!


## Migrate from other libraries

This guide helps you get started with Probdiffeq for solving ordinary differential equations (ODEs), especially if you are familiar with other probabilistic or non-probabilistic ODE solvers in Python or Julia.

Probdiffeq is a JAX library that focuses on state-space-model-based formulations of probabilistic IVP solvers. For what this means, have a look at [this thesis](https://tobias-lib.ub.uni-tuebingen.de/xmlui/handle/10900/152754).



### Migrate from ProbNumDiffEq.jl (Julia)

[ProbNumDiffEq.jl](https://nathanaelbosch.github.io/ProbNumDiffEq.jl/stable/) is a library for probabilistic IVP solvers in Julia, similar to Probdiffeq. However, while the feature offerings are similar, the libraries are unrelated.
To translate ProbNumDiffEq.jl code to Probdiffeq code:

| ProbNumDiffEq.jl           | ProbDiffEq Equivalent                                      |
|-||
| `EK0` / `EK1`              | `constraint_ode_ts0()` / `constraint_ode_ts1()`                         |
| `DynamicDiffusion` / `FixedDiffusion` | `solver_dynamic()` or `solver_mle()` |
| `IWP(diffusion=x^2)` |  `prior_wiener_integrated(output_scale=x)`                                       |
| Filtering and smoothing via `smooth=true/false`      | `strategy_filter`, `strategy_smoother_fixedpoint`, `strategy_smoother_fixedinterval`    |


Both libraries are evolving, and these translation guides may not be up-to-date. 
Consult each libraries' latest API documentation when in doubt.



### Migrate from ProbNum (Python, Numpy)

[ProbNum](https://probnum.readthedocs.io/en/latest/) is a general probabilistic numerics library based on Numpy. Probdiffeq specializes in IVP solvers using pure JAX, offering:

* Greater efficiency for ODE problems because of JAX (e.g. jit)
* Probdiffeq implements more mature solvers. The algorithms are generally faster (eg state-space model factorisations, improved adaptive step-size selection)
* Probdiffeq offers more solvers and somewhat richer outputs (sampling, marginal likelihoods, etc.).



### Migrate from Diffrax

[Diffrax](https://docs.kidger.site/diffrax/) is a JAX-based library for differential equations. The key difference is that Diffrax's solvers are non-probabilistic; Probdiffeq solvers are probabilistic. Approximate solver mapping:

| Diffrax                     | ProbDiffEq Equivalent                                     |
|--|--|
| `Heun()`, `Midpoint()`      | Track $n=2$ Taylor coefficients and use `constraint_ode_ts0()`.  |
| `Tsit5()`, `Dopri5()`       | Track $n=4$ Taylor coefficients instead.                               |
| `Dopri8()`                   | Track $n=5, 6, 7$ Taylor coefficients instead; `constraint_ode_ts1()` and `solver_dynamic()` recommended but not required |
| `Kvaerno3()`, `Kvaerno5()`   | Track $n=2,3,4$ Taylor coefficients and use `constraint_ode_ts1()`         |
| Other methods (e.g. SDE solvers)                | Work in progress                                          |




### Migrate from other common ODE solvers (e.g., SciPy, jax.odeint)

* Probdiffeq's solutions are posterior distributions instead of point estimates, enabling uncertainty quantification and more sophisticated models (eg easy switch to second-order problems).
* Probdiffeq's solver modes are explicit: `simulate_terminal_values()`, and `solve_adaptive_save_at()` instead of a one-size-fits-all `solve()` method.


## Troubleshoot common issues

### General troubleshooting

If you encounter unexpected issues, please ensure you have the latest version of JAX installed. 
If you're not already using [virtual environments](https://docs.python.org/3/tutorial/venv.html), now might be a good time to start, as they can help manage dependencies more effectively.

With these points covered, try to execute some of the examples in Probdiffeq's documentation, for example [the quickstart](https://pnkraemer.github.io/probdiffeq/examples_quickstart/quickstart/).
If these examples work, great! If not, reach out. 


### Long compilation times

If a solution routine takes an unexpectedly long time to compile but runs quickly afterward, the issue might be related to how Taylor coefficients are computed. 
Some functions in `probdiffeq.taylor` unroll a small loop, which can slow down compilation.  
To avoid this, try using the padded scan, which replaces loop unrolling with a scan.  
If the problem persists, consider:  

- Reducing the number of derivatives (if appropriate for your problem).  
- Switching to a different Taylor-coefficient routine, such as a Runge-Kutta starter.

For $\nu < 5$, using a Runge-Kutta starter should maintain solver performance. However, for higher-order methods (e.g., \(\nu = 9\)), Taylor-mode ("jets") is the best choice.  


### Taylor-derivative routines yield NaNs

If you encounter unexpected NaNs while estimating Taylor derivative routines, the issue might come from the vector field itself.
For instance, in the Pleiades problem, there's a term like $\|x\|^2 / (\|x\|^2 + \|y\|^2)$, which can have differentiability issues near zero, depending on how it's implemented. 
See [this issue (external)](https://github.com/pnkraemer/diffeqzoo/issues/126) for more details.
In some cases, the fix is as simple as wrapping the quotient in `jax.numpy.nan_to_num`. 
You can also check out [Probdiffeq's Pleiades benchmark](https://github.com/pnkraemer/probdiffeq/blob/main/docs/benchmarks/work-precision-pleiades.py) for a concrete example.


### Other problems
Your problem is not discussed here? Feel free to reach out. Opening an issue is a great way to get help!


## Use the CI

This guide explains how to install dependencies, run linting and formatting checks, execute tests, and build documentation as part of the continuous integration (CI) process.

### Install Probdiffeq with all dev-related dependencies

After cloning the repository, in the root of the project, and assuming JAX is already installed, do the following:
To install all development dependencies, use one or more of the following commands:

```commandline
pip install .[test]  
pip install .[format-and-lint] 
pip install .[doc] 
```

To install everything required for development, you can install all extras at once:

```commandline
pip install .[test,format-and-lint,doc]
```

### Run all the checks

The project uses a `Makefile` to streamline common CI tasks. 
You can run the following commands to check code quality and correctness:

#### 1. Check/apply formatting and Linting

To check code formatting and linting rules, run:

```commandline
make format-and-lint
```

This will:
- Ensure code is properly formatted.
- Verify that imports are correctly ordered.
- Check for style violations and linting issues.
- Enforce documentation conventions.

#### 2. Run tests

To execute all tests, use:

```commandline
make test
```

This will execute all tests (in parallel, for efficiency).


#### 3. Build the documentation

To generate the documentation, use:

```commandline
make doc
```

This will:
- Sync content in docs/* with the rest of the repo.
- Execute the examples and benchmarks
- Build the documentation site.

To preview the docs, use:

```commandline 
make doc-serve
```

#### 4. Clean Up

To remove auxiliary files generated during testing or documentation builds, run:

```commandline
make clean
```

This removes unnecessary files (eg pytest or mypy caches) to keep the repository clean.

### Use pre-commit hooks

To ensure code quality before committing, the project uses `pre-commit` hooks. These automatically format, lint, and check files before they are committed to the repository.

#### Set up Pre-commit

Install `pre-commit` and set up the hooks by running:

```commandline
pip install pre-commit  # Included in `pip install -e .[format-and-lint]`
pre-commit install
```

#### Run pre-commit hooks manually

To check all files, not just the staged ones, run:

```commandline
pre-commit run --all-files
```

To check only the files staged for commit, run:

```commandline
pre-commit run
```

This ensures that only properly formatted and linted code is committed.


## Create new examples or benchmarks

Probdiffeq hosts numerous examples and benchmarks. The differences between examples and benchmarks are minimal: they are all Python scripts (which become `Jupyter notebook` files in the final docs) and each demonstrates one functionality. 

- Examples show *what* probdiffeq offers

- Benchmarks show *how well* it performs, often compared to other solver libraries. 

Each example or benchmark should run in under a minute, most run in a few seconds. New contributions are welcome!

### Steps

1. **Create the script:**  
  Create a new  notebook in the appropriate subdirectory of `examples/` or `benchmarks/`.
  Choose a meaningful name (e.g., `benchmarks/work-precision-hires.py`, `examples/demonstrate-calibration.py`). 
  The examples show up in the documentation according to the alphabetic order in the `examples/` and `benchmarks` directories.
    
2. **Fill the script:** 
  Write the benchmark/example code. Ensure the execution time stays well below one minute to keep CI manageable.
 
3. **Write documentation:**
  The module docstring will become the title and description of the notebook, so choose a good one. 
    
4. **Pull request:**  
  Commit the new file (the pre-commit hook will handle formatting and linting). Open a pull request and you're done.

## Citation

Here are some references for citing Probdiffeq and its algorithms in your research.

### The library 

If you use **Probdiffeq** in your research, please cite:

```bibtex
@phdthesis{kramer2024implementing,
  title={Implementing probabilistic numerical solvers for differential equations},
  author={Kr{\"a}mer, Peter Nicholas},
  year={2024},
  school={Universit{"a}t T{"u}bingen}
}
```
The [PDF](https://tobias-lib.ub.uni-tuebingen.de/xmlui/handle/10900/152754) explains the mathematics and algorithms behind this library.  

### The libraries' time-stepping 

For the *solve-and-save-at* functionality, cite:

```bibtex
@InProceedings{kramer2024adaptive,
  title     = {Adaptive Probabilistic ODE Solvers Without Adaptive Memory Requirements},
  author    = {Kr{\"a}mer, Nicholas},
  booktitle = {Proceedings of the First International Conference on Probabilistic Numerics},
  pages     = {12--24},
  year      = {2025},
  editor    = {Kanagawa, Motonobu and Cockayne, Jon and Gessner, Alexandra and Hennig, Philipp},
  volume    = {271},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v271/kramer25a.html}
}
```
Link to the paper: [PDF](https://arxiv.org/abs/2410.10530).

Link to the experiments: 
[Code for experiments](https://github.com/pnkraemer/code-adaptive-prob-ode-solvers).  


### Specific algorithms

Algorithms in **Probdiffeq** are based on multiple research papers. If you’re unsure which to cite, feel free to reach out. 
A (subjective, probdiffeq-centric) list of relevant work includes the following articles.


#### Numerically robustness and state-space model factorisations

- Nicholas Krämer & Philipp Hennig (2024). Stable implementation of probabilistic ODE solvers. Journal of Machine Learning Research, 25(111), 1–29.    All suggestions made in this work are critical to Probdiffeq (and other libraries). They are rarely discussed though, and almost taken for granted by now.

- Nicholas Krämer, Nathanael Bosch, Jonathan Schmidt & Philipp Hennig (2022). Probabilistic ODE solutions in millions of dimensions.  In ICML 2022, 11634–11649. PMLR. Every time Probdiffeq uses state-space model factorisations, it follows the recommendations in this work. 

#### Adaptive step-size selection
  
- Michael Schober, Simo Särkkä & Philipp Hennig (2019). A probabilistic model for the numerical solution of initial value problems. Statistics and Computing, 29(1), 99–122.  
  
- Nathanael Bosch, Philipp Hennig & Filip Tronarp (2021). Calibrated adaptive probabilistic ODE solvers. In AISTATS 2021, 3466–3474. PMLR.  
  
- Nicholas Krämer (2025). Adaptive Probabilistic ODE Solvers Without Adaptive Memory Requirements. In Kanagawa, M., Cockayne, J., Gessner, A., & Hennig, P. (Eds.), Proceedings of the First International Conference on Probabilistic Numerics, 12–24. PMLR.
  
#### Constraints, linearisation, and information operators
  
- Bosch, Nathanael, Filip Tronarp, and Philipp Hennig. "Pick-and-mix information operators for probabilistic ODE solvers." International Conference on Artificial Intelligence and Statistics. PMLR, 2022.

- Tronarp, Filip, et al. "Probabilistic solutions to ordinary differential equations as nonlinear Bayesian filtering: a new perspective." Statistics and Computing 29.6 (2019): 1297-1315.

- See also the Linearisation-chapter in: Krämer, Nicholas. Implementing probabilistic numerical solvers for differential equations. Diss. Dissertation, Tübingen, Universität Tübingen, 2024.


#### Parameter estimation

- Kersting, H., Krämer, N., Schiegg, M., Daniel, C., Tiemann, M., & Hennig, P. (2020, November). Differentiable likelihoods for fast inversion of’likelihood-free’dynamical systems. In International Conference on Machine Learning (pp. 5198-5208). PMLR.

-  Tronarp, Filip, Nathanael Bosch, and Philipp Hennig. "Fenrir: Physics-enhanced regression for initial value problems." International Conference on Machine Learning. PMLR, 2022.

-  Beck, J., Bosch, N., Deistler, M., Kadhim, K. L., Macke, J. H., Hennig, P., & Berens, P. (2024, July). Diffusion Tempering Improves Parameter Estimation with Probabilistic Integrators for Ordinary Differential Equations. In International Conference on Machine Learning (pp. 3305-3326). PMLR.


#### Prior distributions

- Bosch, Nathanael, Philipp Hennig, and Filip Tronarp. "Probabilistic exponential integrators." Advances in Neural Information Processing Systems 36 (2023): 40450-40467.




Anything missing? Reach out! 
