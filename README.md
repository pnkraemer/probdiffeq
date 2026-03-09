# Probabilistic solvers in JAX

[![CI](https://github.com/pnkraemer/probdiffeq/workflows/ci/badge.svg)](https://github.com/pnkraemer/probdiffeq/actions)
[![PyPI version](https://img.shields.io/pypi/v/probdiffeq.svg)](https://pypi.python.org/pypi/probdiffeq)
[![License](https://img.shields.io/pypi/l/probdiffeq.svg)](https://pypi.python.org/pypi/probdiffeq)
[![Python versions](https://img.shields.io/pypi/pyversions/probdiffeq.svg)](https://pypi.python.org/pypi/probdiffeq)




**Probdiffeq** implements adaptive probabilistic numerical solvers for differential equations (ODEs). It builds on [JAX](https://jax.readthedocs.io/en/latest/), thus inheriting **automatic differentiation**, **vectorisation**, and **GPU acceleration**.



> ⚠️ Probdiffeq is an active research project. Expect rough edges and sudden API changes.


**Features:**

- ⚡ Automatic calibration and step-size adaptation  
- ⚡ Stable implementations of filtering, smoothing, and other estimation strategies  
- ⚡ Custom information operators, dense output, posterior sampling, and prior distributions.
- ⚡ Efficient handling of high-dimensional problems through state-space model factorisations  
- ⚡ Parameter estimation
- ⚡ Taylor-series estimation with and without automatic differentiation  
- ⚡ Seamless interoperability with [Optax](https://optax.readthedocs.io/en/latest/index.html), [BlackJAX](https://blackjax-devs.github.io/blackjax/), and other JAX-based libraries  
- ⚡ Numerous examples (basic and advanced) -- see the [documentation](https://pnkraemer.github.io/probdiffeq/)  


**Quickstart:** See [here](https://pnkraemer.github.io/probdiffeq/Examples/A0_get_started/) for a minimal example to get you started.


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

To install the library with JAX (using the CPU backend):  

```bash
pip install probdiffeq[cpu]
```

**Compatibility note:** For GPU support, install JAX with CUDA following [JAX installation instructions](https://jax.readthedocs.io/en/latest/installation.html).



**Versioning:** Probdiffeq follows semantic versioning via **0.MINOR.PATCH**:

- **PATCH**: increase with bugfixes & new features  
- **MINOR**: increase with breaking changes  

See [semantic versioning](https://semver.org/).
Notably, Probdiffeq's API is not guaranteed to be stable, but we do our best to follow the versioning scheme so that downstream projects remain reproducible.




## Coming from other ODE solver libraries?

This guide helps you get started with Probdiffeq for solving ordinary differential equations (ODEs), especially if you are familiar with other probabilistic or non-probabilistic ODE solvers in Python or Julia.

Probdiffeq is a JAX library that focuses on state-space-model-based formulations of probabilistic IVP solvers. For what this means, have a look at [this thesis](https://tobias-lib.ub.uni-tuebingen.de/xmlui/handle/10900/152754).

**Probabilistic ODE solvers in a nutshell:** Unlike traditional solvers that return a single point estimate of the solution, probabilistic solvers return a posterior distribution. This built-in uncertainty quantification reflects the numerical error (and other modelling choices), and helps you make better decisions during the simulation and in downstream tasks, for example, during adaptive time-stepping, parameter estimation, or in physics-informed machine learning applications.


### From traditional (non-probabilistic) ODE solvers

If you're coming from traditional ODE solvers like SciPy's `integrate.solve_ivp`, JAX's `jax.experimental.odeint`, or [Diffrax](https://docs.kidger.site/diffrax/), you'll notice some fundamental differences:

**Key differences:**

* **Solutions as distributions:** Probdiffeq returns posterior distributions instead of point estimates. You automatically get uncertainty quantification, which you can use for sensitivity analysis, model selection, or downstream decision-making.
* **Fine-grained control:** Probdiffeq lets you customise the probabilistic model (prior distribution, calibration method, linearization order), giving you more control over solver behaviour.
Since the modelling matters, everyone *has* to build their own custom solvers, and default behaviour is rare.
* **Explicit solver modes:** Instead of a single `solve()` function, Probdiffeq offers specialised functions for targeting terminal values, checkpoints, or fixed grids. This is not just easier to maintain, but also enables better performance by easier code optimisation and specialised default parameters (e.g. whether or not timesteps should be clipped before checkpoints).

**Mapping from Diffrax methods:** If you're switching from Diffrax, here's how to achieve similar accuracy levels by adjusting Taylor coefficients and linearization order:

| Diffrax method               | ProbDiffEq approach                                     |
|--|--|
| `Heun()`, `Midpoint()` | Use 2 Taylor coefficients with zeroth-order linearization |
| `Tsit5()`, `Dopri5()` | Use 5 Taylor coefficients with zeroth-order linearization |
| `Dopri8()` | Use 8 Taylor coefficients with zeroth-order linearization |
| `Kvaerno3()`, `Kvaerno5()` | Use 2 to 5 Taylor coefficients with first-order linearization |

**Tidbit:** Probabilistic solvers based on the once-integrated Wiener/OU processes are closely related to (different versions of) the trapezoidal rule (Schober et al., 2019; Bosch et al., 2023). Higher-order methods connect to more general linear multistep methods (Schober et al., 2019).

- > Michael Schober, Simo Särkkä & Philipp Hennig (2019). A probabilistic model for the numerical solution of initial value problems. Statistics and Computing, 29(1), 99–122.

- > Bosch, Nathanael, Philipp Hennig, and Filip Tronarp. "Probabilistic exponential integrators." Advances in Neural Information Processing Systems 36 (2023): 40450-40467.



**Note:** Probdiffeq is not a drop-in replacement for these solvers; the probabilistic approach is fundamentally different. However, you can match performance and accuracy levels by tuning the solver configuration (see the examples in the documentation).

### From other probabilistic ODE solvers

If you're familiar with other probabilistic solver libraries, here are the comparisons:

**From ProbNum (Python, Numpy):** [ProbNum](https://probnum.readthedocs.io/en/latest/) is a general-purpose probabilistic numerics library, while Probdiffeq specialises in ODE solving with pure JAX. Advantages of Probdiffeq:

* Greater efficiency due to JAX's JIT compilation and autodiff
* More mature ODE algorithms (state-space factorisations, improved adaptive time-stepping)
* Richer outputs (sampling, marginal likelihoods, marginal-likelihood losses, etc.)

**From ProbNumDiffEq.jl (Julia):** [ProbNumDiffEq.jl](https://nathanaelbosch.github.io/ProbNumDiffEq.jl/stable/) is a Julia equivalent of Probdiffeq (though the libraries are unrelated), with similar features but slightly different APIs. Here's how to translate:

| ProbNumDiffEq.jl concept     | ProbDiffEq concept                                     |
|----------------------------- | --------------------|
| `EK0` / `EK1` | `constraint_ode_ts0()` / `constraint_ode_ts1()` |
| `DynamicDiffusion` / `FixedDiffusion` | `solver_dynamic()` / `solver_mle()` |
| `IWP(diffusion=x^2)` | `prior_wiener_integrated(output_scale=x)` |
| `smooth=true/false` | `strategy_filter()` / `strategy_smoother_fixedpoint()` / `strategy_smoother_fixedinterval()` |

Both libraries are actively evolving; consult their latest API documentation if you're unsure about equivalences.


## Choose the right solver

Good solvers are problem-dependent. However, some guidelines exist:

### Problem characteristics

Choosing the right approach matters because problem size and behaviour directly impact solver efficiency, stability, and the accuracy of the uncertainty quantification.

**Dimensionality:** 
For low-dimensional problems, use dense covariances, which track full correlations between state variables and offer the best stability and uncertainty quantification. For larger problems, use blockdiagonal or isotropic state-space models, which are more efficient by tracking only partial uncertainty correlations. However, their uncertainty quantification is typically worse. The general trade-off is between accuracy and speed: dense models scale cubically in the dimension but provide the best accuracy; the other two models scale linearly in the dimension.

**Stiffness:** Stiff problems have rapid changes or very different timescales. For these, use dense state-space models with first-order linearization. See also the prior recommendations below. Avoid zeroth-order methods and isotropic state-space models for stiff problems. Block-diagonal state-space models with first-order linearization may suffice for moderately stiff cases, but expect that all solvers except first-order linearisation in dense state-space models have worse stability than, for example, implicit Runge-Kutta methods.


### Filters vs smoothers

Choosing between filters and smoothers matters because it balances computational cost with the accuracy of uncertainty estimates across the trajectory. 
Use fixed-point smoothing for adaptive timestepping and fixed-interval smoothing for fixed timestepping. When only computing the terminal value of a differential equation, choose a filter.


### Calibration

Calibration matters not just because it ensures uncertainty estimates reflect the real error, but also because it can considerably affect adaptive time-stepping. Use dynamic calibration when output scales vary significantly, for example, in $u'(t) = 10u(t)$, $0 \leq t \leq 10$. Use maximum-likelihood calibration for other cases. Remove automatic calibration for parameter estimation.
When solving multidimensional problems where each dimension has a different magnitude, adjust the output scale of the prior manually before the simulation.


### Prior distributions

Prior distributions encode assumptions about solution dynamics. 
The default prior is the integrated Wiener process; however, integrated Ornstein-Uhlenbeck processes work well for discretised semilinear partial differential equations (and other semilinear problems), especially in fixed-step simulations.
For other needs, consult:

> Bosch, Nathanael, Philipp Hennig, and Filip Tronarp. "Probabilistic exponential integrators." Advances in Neural Information Processing Systems 36 (2023): 40450-40467.

Adjust the base-output-scales of the Wiener process if state variables have vastly different magnitudes, like in the Robertson problem, where one dimension is $10^{5}$ times smaller than the other.


### Number of Taylor coefficients (''order'')

The number of Taylor coefficients trades off accuracy against computational cost. Use 4-5 for most problems, 7-8 when simulating with tolerances close to machine precision, and 2-3 in low-precision arithmetic (for instance, on a GPU).


### Error estimation

In adaptive time-stepping, there exist different error estimates. The default for solving (explicit) ODEs is the residual-based one, because it has proven effective over many years. When solving implicit differential equations (like DAEs), use the state-based estimate instead because the constraints may live on arbitrary scales, which the residual-based method struggles with.
For error-normalisation, use scale-then-RMS when applicable, and RMS-then-scale only if necessary.

For controllers, the choice does not matter much. Integral controllers seem to be slightly more effective for most problems except for stiff ODEs, where proportional-integral controllers work better. Try all and report back with any insights.


### Summary: Choosing a solver

**For beginners:** Start with integrated Wiener processes and four Taylor coefficients, fixed-point smoothing, first-order linearization, and dense state-space models. For high-dimensional problems, use zeroth-order linearization and block-diagonal state-space models. For parameter estimation, use fixed steps with a fixed-interval smoother.

**For advanced users:** Use the guidelines above based on your problem's dimensionality, stiffness, and requirements. Consult the examples in the documentation, and reach out with any questions.



## Troubleshoot common issues 

If you encounter unexpected issues, please ensure you have the latest version of JAX installed. 
If you're not already using [virtual environments](https://docs.python.org/3/tutorial/venv.html), now might be a good time to start, as they can help manage dependencies more effectively.
With these points covered, try to execute some of the examples in Probdiffeq's documentation, for example [the quickstart](https://pnkraemer.github.io/probdiffeq/Examples/A0_get_started/).
If these examples work, great! If not, reach out. 


### Long compilation times

If a solution routine takes an unexpectedly long time to compile but runs quickly afterwards, the issue might be related to how Taylor coefficients are computed. 
Some functions in `probdiffeq.taylor` unroll a small loop, which can slow down compilation.  
To avoid this, try using the padded scan, which replaces loop unrolling with a scan.  
If the problem persists, consider reducing the number of derivatives (if appropriate for your problem). 

### Taylor-derivative routines yield NaNs

If you encounter unexpected NaNs while estimating Taylor derivative routines, the issue might come from the vector field itself.
For instance, in the Pleiades problem, there's a term like $\|x\|^2 / (\|x\|^2 + \|y\|^2)$, which can have differentiability issues near zero, depending on how it's implemented. 
See [this issue (external)](https://github.com/pnkraemer/diffeqzoo/issues/126) for more details.
In some cases, the fix is as simple as wrapping the quotient in `jax.numpy.nan_to_num`. 
You can also check out [Probdiffeq's Pleiades benchmark](https://github.com/pnkraemer/probdiffeq/Extended_Benchmarks/A1_walltime_|_pleiades/) for a concrete example.


### Other problems
Is your problem not discussed here? Feel free to reach out. Opening an issue is a great way to get help!


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

This removes unnecessary files (e.g., pytest or mypy caches) to keep the repository clean.

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

Probdiffeq hosts numerous examples and benchmarks. The differences between examples and benchmarks are minimal: they are all Python scripts (which become `Jupyter notebook` files in the final docs), and each demonstrates one functionality. 

- Examples show *what* probdiffeq offers

- Benchmarks show *how well* it performs, often compared to other solver libraries. 

Each example or benchmark should run in under a minute; most run in a few seconds. New contributions are welcome!

### Steps

1. **Create the script:** Create a new notebook in the appropriate subdirectory of `examples/` or `benchmarks/`.
 Choose a meaningful name (e.g., `benchmarks/work-precision-hires.py`, `examples/demonstrate-calibration.py`). 
 The examples show up in the documentation according to alphabetical order in the `examples/` and `benchmarks` directories.
    
2. **Fill the script:** 
 Write the benchmark/example code. Ensure the execution time stays well below one minute to keep CI manageable.
 
3. **Write documentation:** The module docstring will become the title and description of the notebook, so choose a good one. 
    
4. **Make a pull request:** Commit the new file (the pre-commit hook will handle formatting and linting). Open a pull request, and you're done.

## Citation

Please consider citing Probdiffeq and its algorithms if it helps you in your research.
Here are some concrete suggestions for how.

### Essential citations

If you use **Probdiffeq** in your research, please cite:

> Krämer, N. (2023). Implementing probabilistic numerical solvers for differential equations (Doctoral dissertation, Dissertation, Tübingen, Universität Tübingen, 2024).

Here is a BibTeX:

```bibtex
@phdthesis{kramer2024implementing,
  title={Implementing probabilistic numerical solvers for differential equations},
  author={Kr{\"a}mer, Peter Nicholas},
  year={2024},
  school={Universit{"a}t T{"u}bingen}
}
```
The [PDF](https://tobias-lib.ub.uni-tuebingen.de/xmlui/handle/10900/152754) explains the mathematics and algorithms behind this library.  
If there is one text to reference when acknowledging Probdiffeq, it is the PhD thesis above.

However, there are some additional references that are critical to this library:


**Adaptive time-stepping:**
When using adaptive time-stepping, also cite the adaptive step-sizing paper:

> Nicholas Krämer (2025). Adaptive Probabilistic ODE Solvers Without Adaptive Memory Requirements. In Kanagawa, M., Cockayne, J., Gessner, A., & Hennig, P. (Eds.), Proceedings of the First International Conference on Probabilistic Numerics, 12–24. PMLR.

Here is a BibTeX:

```bibtex
@InProceedings{kramer2024adaptive,
  title = {Adaptive Probabilistic ODE Solvers Without Adaptive Memory Requirements},
  author = {Kr{\"a}mer, Nicholas},
  booktitle = {Proceedings of the First International Conference on Probabilistic Numerics},
  pages = {12--24},
  year = {2025},
  editor = {Kanagawa, Motonobu and Cockayne, Jon and Gessner, Alexandra and Hennig, Philipp},
  volume = {271},
  series = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url = {https://proceedings.mlr.press/v271/kramer25a.html}
}
```
Link to the paper: [PDF](https://arxiv.org/abs/2410.10530).

Link to the experiments: 
[Code for experiments](https://github.com/pnkraemer/code-adaptive-prob-ode-solvers).  


**Numerical implementations:**
If you use more than one or two Taylor coefficients in the state-space model, you're benefiting from numerically robust implementations of probabilistic solvers:

> Nicholas Krämer & Philipp Hennig (2024). Stable implementation of probabilistic ODE solvers. Journal of Machine Learning Research, 25(111), 1–29.

Here is a BibTeX:

```bibtex
@article{kraemer2024stable,
  title={Stable implementation of probabilistic ODE solvers},
  author={Kraemer, Nicholas and Hennig, Philipp},
  journal={Journal of Machine Learning Research},
  volume={25},
  number={111},
  pages={1--29},
  year={2024}
}
```


### Specific algorithms

Algorithms in **Probdiffeq** are based on multiple research papers. If you’re unsure which to cite, feel free to reach out. 
A (subjective, probdiffeq-centric) list of relevant work includes the following articles.


#### Numerical robustness and state-space model factorisations

- Nicholas Krämer & Philipp Hennig (2024). Stable implementation of probabilistic ODE solvers. Journal of Machine Learning Research, 25(111), 1–29.  

    **Key insights:** All suggestions made in this work are critical to numerical implementations of probabilistic solvers. They are implemented by Probdiffeq (and other libraries).

- Nicholas Krämer, Nathanael Bosch, Jonathan Schmidt & Philipp Hennig (2022). Probabilistic ODE solutions in millions of dimensions.  In ICML 2022, 11634–11649. PMLR. 

    **Key insights:** Every time Probdiffeq uses state-space model factorisations, it follows the recommendations in this work. 

#### Adaptive step-size selection (and calibration)
  
- Michael Schober, Simo Särkkä & Philipp Hennig (2019). A probabilistic model for the numerical solution of initial value problems. Statistics and Computing, 29(1), 99–122.  

    **Key insights:** This work is the first on calibration and adaptive step-size selection in state-space-model-based ODE solvers.
  
- Nathanael Bosch, Philipp Hennig & Filip Tronarp (2021). Calibrated adaptive probabilistic ODE solvers. In AISTATS 2021, 3466–3474. PMLR.  

    **Key insights:** This work describes calibration and adaptive step-size selection as we use it now.

- Nicholas Krämer, Nathanael Bosch, Jonathan Schmidt & Philipp Hennig (2022). Probabilistic ODE solutions in millions of dimensions.  In ICML 2022, 11634–11649. PMLR. 

    **Key insights:** This work is a small extension of Bosch et al. (2021)'s calibration and error estimates to factorised state-space models. 

- Nicholas Krämer (2025). Adaptive Probabilistic ODE Solvers Without Adaptive Memory Requirements. In Kanagawa, M., Cockayne, J., Gessner, A., & Hennig, P. (Eds.), Proceedings of the First International Conference on Probabilistic Numerics, 12–24. PMLR. 

    **Key insights:** Adaptive time-stepping with fixed-point smoothers makes memory requirements constant. Probdiffeq's time-stepping loop implements this paper.
  
#### Constraints, linearisation, and information operators

- Tronarp, Filip, et al. "Probabilistic solutions to ordinary differential equations as nonlinear Bayesian filtering: a new perspective." Statistics and Computing 29.6 (2019): 1297-1315. 

    **Key insight:** As one of the foundational works on probabilistic solvers, it links ODE solvers to zeroth- and first-order linearisation in Gaussian filters.


- Bosch, Nathanael, Filip Tronarp, and Philipp Hennig. "Pick-and-mix information operators for probabilistic ODE solvers." International Conference on Artificial Intelligence and Statistics. PMLR, 2022. 

    **Key insights:** Encode, e.g. second-order dynamics, Hamiltonian preservation, or implicit differential equations directly in the constraints without transforming the problem into a first-order explicit ODE.


#### Parameter estimation

- Kersting, H., Krämer, N., Schiegg, M., Daniel, C., Tiemann, M., & Hennig, P. (2020, November). Differentiable likelihoods for fast inversion of `likelihood-free` dynamical systems. In International Conference on Machine Learning (pp. 5198-5208). PMLR. 

    **Key insight:** The first work on using the likelihood of observational data under a posterior distribution given by the probabilistic ODE solution. 

- Tronarp, Filip, Nathanael Bosch, and Philipp Hennig. "Fenrir: Physics-enhanced regression for initial value problems." International Conference on Machine Learning. PMLR, 2022. 

    **Key insight:** The formulation of the likelihood of the observational data as we use it now.

- Beck, J., Bosch, N., Deistler, M., Kadhim, K. L., Macke, J. H., Hennig, P., & Berens, P. (2024, July). Diffusion Tempering Improves Parameter Estimation with Probabilistic Integrators for Ordinary Differential Equations. In International Conference on Machine Learning (pp. 3305-3326). PMLR. 

    **Key insight:** An improved algorithm for parameter estimation using the above likelihood formulation based on diffusion tempering (see the tutorial).


#### Prior distributions

- Schober, M., Duvenaud, D., & Hennig, P. (2014). Probabilistic ODE solvers with Runge-Kutta means. Advances in neural information processing systems, 27. 

    **Key insights:** Use Gauss--Markov processes, specifically, high-order integrated Wiener processes, to replicate the efficiency of non-probabilistic ODE solvers. 

- Kersting, H., Sullivan, T. J., & Hennig, P. (2020). Convergence rates of Gaussian ODE filters. Statistics and computing, 30(6), 1791-1816. 

    **Key insights:** One of the first works that mentions integrated Ornstein-Uhlenbeck priors in the context of ODE solvers.

- Bosch, Nathanael, Philipp Hennig, and Filip Tronarp. "Probabilistic exponential integrators." Advances in Neural Information Processing Systems 36 (2023): 40450-40467. 

    **Key insights:** Replicate the behaviour of exponential integrators by choosing priors different to integrated Wiener processes.




Anything missing? Reach out! 
