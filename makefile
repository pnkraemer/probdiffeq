
.PHONY: format lint test pre-commit doc clean

format:
	isort .
	black .
	nbqa black docs/
	nbqa isort docs/
	jupytext --sync docs/examples/*

lint:
	isort --check --diff .
	black --check --diff .
	flake8
	nbqa isort --check --diff .
	nbqa black --check --diff .
	nbqa flake8 docs/

test:
	pytest -x -v
	python -m doctest odefilter/*.py

example:
	jupytext --sync docs/examples/*
	jupytext --execute docs/examples/*

pre-commit:
	pre-commit autoupdate
	pre-commit run --all-files

clean:
	rm -rf .pytest_cache
	rm -rf *.egg-info
	rm -rf dist site

doc:
	mkdocs build
