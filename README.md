# Probabilistic solvers in JAX

[![CI](https://github.com/pnkraemer/probdiffeq/workflows/ci/badge.svg)](https://github.com/pnkraemer/probdiffeq/actions)
[![PyPI version](https://img.shields.io/pypi/v/probdiffeq.svg)](https://pypi.python.org/pypi/probdiffeq)
[![License](https://img.shields.io/pypi/l/probdiffeq.svg)](https://pypi.python.org/pypi/probdiffeq)
[![Python versions](https://img.shields.io/pypi/pyversions/probdiffeq.svg)](https://pypi.python.org/pypi/probdiffeq)


**Probdiffeq** implements adaptive probabilistic numerical solvers for ordinary differential equations (ODEs). It builds on [JAX](https://jax.readthedocs.io/en/latest/), thus inheriting **automatic differentiation**, **vectorisation**, and **GPU acceleration**.

**Features:**

- ⚡ Calibration and step-size adaptation  
- ⚡ Stable implementations of filtering, smoothing, and other estimation strategies  
- ⚡ Custom information operators, dense output, and posterior sampling  
- ⚡ State-space model factorisations  
- ⚡ Custom prior processes, ODE-, mass-matrix-ODE-, and DAE-solvers
- ⚡ Parameter estimation
- ⚡ Taylor-series estimation with and without Jets  
- ⚡ Seamless interoperability with [Optax](https://optax.readthedocs.io/en/latest/index.html), [BlackJAX](https://blackjax-devs.github.io/blackjax/), and other JAX-based libraries  
- ⚡ Numerous tutorials (basic and advanced) -- see the [documentation](https://pnkraemer.github.io/probdiffeq/)  



**Contributing**

Contributions are very welcome!  
- Browse open issues (look for “good first issue”).  
- Check the developer documentation.  
- Open an issue for feature requests or ideas.  



**Related projects**

- [Tornadox](https://github.com/pnkraemer/tornadox)  
- [ProbNumDiffEq.jl](https://nathanaelbosch.github.io/ProbNumDiffEq.jl/stable/)  
- [ProbNum](https://probnum.readthedocs.io/en/latest/)  

The docs include guidance on migrating from these packages. Missing something? Open an issue or pull request!



**You might also like**

- [diffeqzoo](https://diffeqzoo.readthedocs.io/en/latest/) — reference implementations of differential equations in NumPy and JAX  
- [probfindiff](https://probfindiff.readthedocs.io/en/latest/) — probabilistic finite-difference methods in JAX  


## Installation

Install the latest release from PyPI:

```bash
pip install probdiffeq
```

> This assumes [JAX](https://jax.readthedocs.io/en/latest/) is already installed.  

To install with JAX (CPU backend):  

```bash
pip install probdiffeq[cpu]
```

⚠️ **Note:** This is an active research project. Expect rough edges and breaking API changes.




**Versioning**

Probdiffeq follows **0.MINOR.PATCH** until its first stable release:  
- **PATCH** → bugfixes & new features  
- **MINOR** → breaking changes  

See [semantic versioning](https://semver.org/).
Notably, Probdiffeq's API is not guaranteed to be stable, but we do our best to follow the versioning scheme so that downstream projects remain reproducible.



## Citing

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


Algorithms in **Probdiffeq** are based on multiple research papers. If you’re unsure which to cite, feel free to reach out. 

A (subjective, probdiffeq-centric) list of relevant work includes:


- Numerically robust implementations of probabilistic solvers:

  > Nicholas Krämer & Philipp Hennig (2024). Stable implementation of probabilistic ODE solvers. Journal of Machine Learning Research, 25(111), 1–29.  
  
  All suggestions made in this work are critical to Probdiffeq (and other libraries). They are rarely discussed though, and almost taken for granted by now.


- State-space model factorisations:

  > Nicholas Krämer, Nathanael Bosch, Jonathan Schmidt & Philipp Hennig (2022). Probabilistic ODE solutions in millions of dimensions.  In ICML 2022, 11634–11649. PMLR.  

  Every time Probdiffeq uses state-space model factorisations, it follows the recommendations in this work. 

- Adaptive step-size selection:
  
  > Michael Schober, Simo Särkkä & Philipp Hennig (2019). A probabilistic model for the numerical solution of initial value problems. Statistics and Computing, 29(1), 99–122.  
  
  > Nathanael Bosch, Philipp Hennig & Filip Tronarp (2021). Calibrated adaptive probabilistic ODE solvers. In AISTATS 2021, 3466–3474. PMLR.  
  
  > Nicholas Krämer (2025). Adaptive Probabilistic ODE Solvers Without Adaptive Memory Requirements. In Kanagawa, M., Cockayne, J., Gessner, A., & Hennig, P. (Eds.), Proceedings of the First International Conference on Probabilistic Numerics, 12–24. PMLR.
  
- Constraints, linearisation, and information operators:
  
  > Bosch, Nathanael, Filip Tronarp, and Philipp Hennig. "Pick-and-mix information operators for probabilistic ODE solvers." International Conference on Artificial Intelligence and Statistics. PMLR, 2022.

  >Tronarp, Filip, et al. "Probabilistic solutions to ordinary differential equations as nonlinear Bayesian filtering: a new perspective." Statistics and Computing 29.6 (2019): 1297-1315.

  See also the Linearisation-chapter in:
  
  > Krämer, Nicholas. Implementing probabilistic numerical solvers for differential equations. Diss. Dissertation, Tübingen, Universität Tübingen, 2024.

  which describes some methods not mentioned anywhere else.

- Parameter estimation:

  > Kersting, H., Krämer, N., Schiegg, M., Daniel, C., Tiemann, M., & Hennig, P. (2020, November). Differentiable likelihoods for fast inversion of’likelihood-free’dynamical systems. In International Conference on Machine Learning (pp. 5198-5208). PMLR.

  > Tronarp, Filip, Nathanael Bosch, and Philipp Hennig. "Fenrir: Physics-enhanced regression for initial value problems." International Conference on Machine Learning. PMLR, 2022.

  > Beck, J., Bosch, N., Deistler, M., Kadhim, K. L., Macke, J. H., Hennig, P., & Berens, P. (2024, July). Diffusion Tempering Improves Parameter Estimation with Probabilistic Integrators for Ordinary Differential Equations. In International Conference on Machine Learning (pp. 3305-3326). PMLR.


Anything missing? Reach out! 


## Choose the right probabilistic solver

Good solvers are problem-dependent. However, some guidelines exist:

### State-space model factorisation

* If your problem is scalar-valued (`shape=()`), use a dense factorisation. All factorisations have the same complexity for scalar models, but dense factorisations offer the most comprehensive solver suite.

* If your problem is vector-valued, be aware that different implementation choices imply different modelling choices.
However, if you don't care too much about modelling choices:

* If your problem is high-dimensional, use a `blockdiag` or `isotropic` implementation.
* If your problem is medium-dimensional, use any implementations. 
  `isotropic` factorisations tend to be the fastest with the worst UQ and worst stability, 
  `dense` factorisations tend to be the slowest with the best UQ and best stability, 
  `blockdiag` factorisations are somewhere in between.


### Stiffness

If your problem is stiff, use a a `dense` implementation in combination with a
correction scheme that employs first-order linearisation; 
for instance, `ts1` or `slr1`.
Zeroth-order approximation and isotropic/blockdiag factorisations often fail for stiff problems.

If your problem is stiff and high-dimensional: try first-order linearisation with a block-diagonal factorisation. 
If that does not work: good luck; probabilistic solvers for problems that are stiff 
*and* high-dimensional are a bit of an open problem as of writing this.

### Filters vs smoothers
As a rule of thumb, use a `ivpsolvers.strategy_filter` strategy for `simulate_terminal_values`, 
a `ivpsolvers.strategy_smoother_fixedpoint` strategy for `solve_adaptive_save_at`,
and a `ivpsolvers.strategy_smoother_fixedinterval` strategy for `solve_fixed_step`.
Other combinations are possible, but rare.


### Calibration
Use a `solvers.solver_dynamic` solver if you expect that the output scale of your differential equation
solution varies greatly (eg for first-order, linear ODEs; see the tutorials).
Otherwise, use an `solvers.solver_mle` solver for plain simulation problems, 
and a `solvers.solver` for parameter-estimation.
See also the output scale recommendations under "Prior distributions".


### Prior distributions
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


### Miscellaneous
If you use a `ts0`, choose an `isotropic` factorisation instead of a `dense` factorisation.
They are mathematically equivalent, but the `isotropic` factorisation is faster.

For parameter estimation problems with adaptive solvers, replace Probdiffeq's while-loops
with Equinox's while-loops; see the tutorials for how.

### Future guidelines
These guidelines are a work in progress and may change at any point. If you have any input, reach out.
Something missing? Reach out!


## Migrate from other libraries

This guide helps you get started with Probdiffeq for solving ordinary differential equations (ODEs), especially if you are familiar with other probabilistic or non-probabilistic ODE solvers in Python or Julia.

Probdiffeq is a JAX library that focuses on state-space-model-based formulations of probabilistic IVP solvers. For what this means, have a look at [this thesis](https://tobias-lib.ub.uni-tuebingen.de/xmlui/handle/10900/152754).



### Transition from ProbNumDiffEq.jl (Julia)

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



### Transition from ProbNum (Python, Numpy)

[ProbNum](https://probnum.readthedocs.io/en/latest/) is a general probabilistic numerics library based on Numpy. Probdiffeq specializes in IVP solvers using pure JAX, offering:

* Greater efficiency for ODE problems because of JAX (e.g. jit)
* Probdiffeq implements more mature solvers. The algorithms are generally faster (eg state-space model factorisations, improved adaptive step-size selection)
* Probdiffeq offers more solvers and somewhat richer outputs (sampling, marginal likelihoods, etc.).



### Transition from Diffrax

[Diffrax](https://docs.kidger.site/diffrax/) is a JAX-based library for differential equations. The key difference is that Diffrax's solvers are non-probabilistic; Probdiffeq solvers are probabilistic. Approximate solver mapping:

| Diffrax                     | ProbDiffEq Equivalent                                     |
|--|--|
| `Heun()`, `Midpoint()`      | Track $n=2$ Taylor coefficients and use `constraint_ode_ts0()`.  |
| `Tsit5()`, `Dopri5()`       | Track $n=4$ Taylor coefficients instead.                               |
| `Dopri8()`                   | Track $n=5, 6, 7$ Taylor coefficients instead; `constraint_ode_ts1()` and `solver_dynamic()` recommended but not required |
| `Kvaerno3()`, `Kvaerno5()`   | Track $n=2,3,4$ Taylor coefficients and use `constraint_ode_ts1()`         |
| Other methods (e.g. SDE solvers)                | Work in progress                                          |




### General differences from other common ODE solvers (e.g., SciPy, jax.odeint)

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


## Use Probdiffeq's continuous integration

This guide explains how to install dependencies, run linting and formatting checks, execute tests, and build documentation as part of the continuous integration (CI) process.

### Full Installation

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

### Running Checks

The project uses a `Makefile` to streamline common CI tasks. 
You can run the following commands to check code quality and correctness:

#### 1. Formatting and Linting

To check code formatting and linting rules, run:

```commandline
make format-and-lint
```

This will:
- Ensure code is properly formatted.
- Verify that imports are correctly ordered.
- Check for style violations and linting issues.
- Enforce documentation conventions.

#### 2. Running Tests

To execute all tests, use:

```commandline
make test
```

This will:
- Run all tests.
- Execute tests in parallel for efficiency.

#### 3. Running Benchmarks


We maintain benchmarks comparing **Probdiffeq** against other solvers and libraries, including [SciPy](https://scipy.org/), [JAX](https://jax.readthedocs.io/en/latest/), and [Diffrax](https://docs.kidger.site/diffrax/).


To run the full benchmark suite, use:

```commandline
make benchmarks-run
make benchmarks-plot-results
```

This will:
- Execute benchmarking scripts to assess performance.
- Plot the results so that the next documentation build displays the results.

Benchmarking parameters and configurations can be adjusted in the relevant benchmark scripts, located in the `doc/benchmarks/` directory.

If the goal is not a full benchmark run, but simply a check whether the benchmark scripts execute correctly, use:
```commandline
make benchmarks-run-dry-run
```
This is helpful to verify that API changes are reflected in the benchmark code.


#### 4. Building Documentation

To generate the documentation, use:

```commandline
make doc
```

This will:
- Sync content in docs/* with the rest of the repo.
- Process Jupyter notebooks and Markdown files.
- Build the documentation site.

#### 5. Cleaning Up

To remove auxiliary files generated during testing or documentation builds, run:

```commandline
make clean
```

This removes unnecessary files (eg pytest or mypy caches) to keep the repository clean.

### Pre-commit Hooks

To ensure code quality before committing, the project uses `pre-commit` hooks. These automatically format, lint, and check files before they are committed to the repository.

#### Setting Up Pre-commit

Install `pre-commit` and set up the hooks by running:

```commandline
pip install pre-commit  # Included in `pip install -e .[format-and-lint]`
pre-commit install
```

#### Running Pre-commit Manually

To check all files, not just the staged ones, run:

```commandline
pre-commit run --all-files
```

To check only the files staged for commit, run:

```commandline
pre-commit run
```

This ensures that only properly formatted and linted code is committed.


## Create a new example or benchmark

Probdiffeq hosts numerous tutorials and benchmarks that demonstrate the library. The differences between examples and benchmarks are minimal: they are all Jupyter notebooks (paired to `py:light` files via jupytext for version control) and each demonstrates one functionality. Examples show *what* probdiffeq offers, while benchmarks show *how well* it performs, often compared to other solver libraries. Each tutorial or benchmark should run in under a minute. New contributions are welcome!

### Steps

1. **Create the script:**  
   Create a new Jupyter notebook in the appropriate subdirectory of `docs/`. Example paths include:
   - `docs/examples_benchmarks/benchmark-name.ipynb`
   - `docs/examples_advanced/example-name.ipynb`  
   Choose a meaningful name (e.g., `work-precision-hires`, `demonstrate-calibration`). The notebook should run the full example/benchmark and produce its plots. Ensure execution time stays well below one minute to keep CI manageable.

   If your example requires external dependencies (e.g., sampling or optimization libraries), place it in `examples_advanced`. If it is a benchmark, place it in `examples_benchmarks`. Otherwise, place it in
   `examples_basic`.

2. **Sync to py:light:**  
   Install documentation dependencies and pre-commit hooks if you haven't already:
   ```
   pip install .[doc,format-and-lint]
   pre-commit install
   ```
   Link the notebook to a py:light script using jupytext (preferred for version control and formatting):
   ```
   jupytext --set-formats ipynb,py:light <new-notebook.ipynb>
   ```
   Or link all notebooks at once:
   ```
   jupytext --set-formats ipynb,py:light docs/examples_*/*.ipynb
   ```
   Notebooks placed correctly according to the directory structure will be included by the previous command.
   
3. **Docs:**  
   Add the new `.ipynb` file to the documentation navigation in `mkdocs.yml` under `nav:`.  
   Ensure the corresponding script is excluded under `mkdocs.yml -> exclude:`; if needed, add it there.

4. **Makefile:**  
   Check whether the new example or benchmark needs to be added to the appropriate Makefile target (e.g., `examples-and-benchmarks`). Generally, new files are detected automatically, but check nevertheless.

5. **Pyproject.toml:**  
   If your example requires external dependencies, list them under the `doc` optional dependencies in `pyproject.toml`.

6. **Pull request:**  
   Commit the new notebook (the pre-commit hook will handle formatting and linting). Open a pull request and you're done.
