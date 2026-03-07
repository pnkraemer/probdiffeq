# Use the CI

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

This will:
- Run all tests.
- Execute tests in parallel for efficiency.

#### 3. Execute benchmarks


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


#### 4. Build the documentation

To generate the documentation, use:

```commandline
make doc
```

This will:
- Sync content in docs/* with the rest of the repo.
- Process Jupyter notebooks and Markdown files.
- Build the documentation site.

To preview the docs, use:

```commandline 
make doc-serve
```

#### 5. Clean Up

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

#### Rune pre-commit hooks manually

To check all files, not just the staged ones, run:

```commandline
pre-commit run --all-files
```

To check only the files staged for commit, run:

```commandline
pre-commit run
```

This ensures that only properly formatted and linted code is committed.
