# Continuous integration

## Installation
To install all development-relevant dependencies, install either of
```
pip install probdiffeq[test]  # pytest, ...
pip install probdiffeq[format]  # black, isort, ...
pip install probdiffeq[lint]  # flake8, ...
pip install probdiffeq[doc]  # tueplots, diffrax, mkdocs, ...
```

Run the checks with a makefile, use either of the below
```
make format-and-lint
make test
make doc
```
Remove auxiliary files with 
```
make clean
```

## Pre-commit hook
To ensure that all commits satisfy most of the linters, no big files are addedd accidentally, and so on, use a pre-commit hook
```
pip install pre-commit  # included in `pip install -e .[full]`
pre-commit install
```
You may verify the installation by running
```commandline
pre-commit run
```
which is equivalent to 
```commandline
make lint
```
