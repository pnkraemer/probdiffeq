
.PHONY: format lint test pre-commit doc clean

format:
	isort .
	black .
	nbqa black docs/
	nbqa isort docs/
	jupytext --sync docs/examples/*
	jupytext --sync docs/benchmarks/*

lint:
	isort --check --diff .
	black --check --diff .
	flake8
	pylint odefilter/ --disable=all --enable=arguments-differ,unused-variable
	pylint tests/ --disable=all --enable=arguments-differ,unused-variable
	nbqa isort --check --diff .
	nbqa black --check --diff .
	nbqa flake8 docs/

test:
	pytest -n auto -x -v
	python -m doctest odefilter/*.py

example:
	jupytext --sync docs/examples/*
	jupytext --execute docs/examples/*
	jupytext --sync docs/examples/*
    # No --execute for benchmarks (takes too long)
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
