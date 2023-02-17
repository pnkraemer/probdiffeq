
.PHONY: format lint test doc clean

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
	pre-commit run --all-files

test:
	pytest -n auto -x -v -s  # parallelise, fail early, verbose output, show all 'stdout's
	python -m doctest probdiffeq/*.py

example:
	jupytext --sync docs/examples/*
	jupytext --execute docs/examples/*
	jupytext --sync docs/examples/*
	jupytext --sync docs/advanced_examples/*
	jupytext --execute docs/advanced_examples/*
	jupytext --sync docs/advanced_examples/*
	# No --execute for advanced examples and benchmarks (takes too long)
	jupytext --sync docs/benchmarks/*

clean:
	pre-commit clean
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf *.egg-info
	rm -rf dist site build
	rm -rf *.ipynb_checkpoints
	rm -rf docs/benchmarks/__pycache__
	rm -rf docs/benchmarks/.ipynb_checkpoints

doc:
	mkdocs build
