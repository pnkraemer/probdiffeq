
.PHONY: format lint test pre-commit doc clean

format:
	isort .
	black .
	nbqa black docs/examples/
	nbqa black docs/advanced_examples/
	nbqa black docs/benchmarks/
	nbqa isort docs/examples/
	nbqa isort docs/advanced_examples/
	nbqa isort docs/benchmarks/
	jupytext --sync docs/examples/*
	jupytext --sync docs/advanced_examples/*
	jupytext --sync docs/benchmarks/*

lint:
	# The fail-fast linters
	isort --check --diff .
	black --check --diff .
	flake8
	# Apply the basics to the notebooks
	nbqa isort --check --diff docs/examples/
	nbqa isort --check --diff docs/advanced_examples/
	nbqa isort --check --diff docs/benchmarks/
	nbqa black --check --diff docs/examples/
	nbqa black --check --diff docs/advanced_examples/
	nbqa black --check --diff docs/benchmarks/
	nbqa flake8 docs/examples/
	nbqa flake8 docs/advanced_examples/
	nbqa flake8 docs/benchmarks/
	# Opt-in for specific pylint checks that flake8 can't detect
	pylint odefilter/ --disable=all --enable=arguments-differ,unused-variable,unnecessary-comprehension,redefined-builtin
	pylint tests/ --disable=all --enable=arguments-differ,unused-variable,unnecessary-comprehension,redefined-builtin
	# A very soft mypy check to detect obvious problems
	mypy odefilter --disable-error-code=var-annotated
test:
	pytest -n auto -x -v -s  # parallelise, fail early, verbose output, show all 'stdout's
	python -m doctest odefilter/*.py

example:
	jupytext --sync docs/examples/*
	jupytext --execute docs/examples/*
	jupytext --sync docs/examples/*
	jupytext --sync docs/advanced_examples/*
	jupytext --execute docs/advanced_examples/*
	jupytext --sync docs/advanced_examples/*
	# No --execute for advanced examples and benchmarks (takes too long)
	jupytext --sync docs/benchmarks/*

pre-commit:
	pre-commit autoupdate
	pre-commit run --all-files

clean:
	rm -rf .pytest_cache
	rm -rf *.egg-info
	rm -rf dist site
	rm -rf *.ipynb_checkpoints
	rm -rf docs/benchmarks/__pycache__

doc:
	mkdocs build
