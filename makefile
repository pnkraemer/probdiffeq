
.PHONY: format lint test doc example run-benchmarks clean

format:
	isort --quiet .
	black --quiet .
	nbqa black --quiet docs/quickstart/ docs/examples_benchmarks/
	nbqa isort --quiet docs/quickstart/ docs/examples_benchmarks/
	jupytext --quiet --sync docs/quickstart/*.ipynb
	jupytext --quiet --sync docs/examples_benchmarks/solvers_solutions/*.ipynb
	jupytext --quiet --sync docs/examples_benchmarks/parameter_estimation/*.ipynb
	jupytext --quiet --sync docs/examples_benchmarks/benchmarks/lotka_volterra/*.ipynb
	jupytext --quiet --sync docs/examples_benchmarks/benchmarks/pleiades/*.ipynb
	jupytext --quiet --sync docs/examples_benchmarks/benchmarks/stiff_van_der_pol/*.ipynb
	jupytext --quiet --sync docs/examples_benchmarks/benchmarks/hires/*.ipynb

lint:
	pre-commit run --all-files

test:
	pytest -n auto -v -s  # parallelise, verbose output, show all 'stdout's
	python -m doctest probdiffeq/*.py

example:
	jupytext --quiet --sync docs/quickstart/*.ipynb
	jupytext --quiet --execute docs/quickstart/*.ipynb
	jupytext --quiet --sync docs/examples_benchmarks/solvers_solutions/*
	jupytext --quiet --execute docs/examples_benchmarks/solvers_solutions/*
	jupytext --quiet --sync docs/examples_benchmarks/solvers_solutions/*
	jupytext --quiet --sync docs/examples_benchmarks/parameter_estimation/*
	jupytext --quiet --execute docs/examples_benchmarks/parameter_estimation/*
	jupytext --quiet --sync docs/examples_benchmarks/parameter_estimation/*
	# No --execute for advanced examples and benchmarks (takes too long)
	jupytext --quiet --sync docs/examples_benchmarks/benchmarks/lotka_volterra/*
	jupytext --quiet --sync docs/examples_benchmarks/benchmarks/pleiades/*
	jupytext --quiet --sync docs/examples_benchmarks/benchmarks/stiff_van_der_pol/*
	jupytext --quiet --sync docs/examples_benchmarks/benchmarks/hires/*

run-benchmarks:
	jupytext --quiet --sync docs/examples_benchmarks/benchmarks/lotka_volterra/*
	jupytext --quiet --sync docs/examples_benchmarks/benchmarks/pleiades/*
	jupytext --quiet --sync docs/examples_benchmarks/benchmarks/stiff_van_der_pol/*
	jupytext --quiet --sync docs/examples_benchmarks/benchmarks/hires/*
	jupytext --quiet --execute docs/examples_benchmarks/benchmarks/lotka_volterra/internal.ipynb
	jupytext --quiet --execute docs/examples_benchmarks/benchmarks/lotka_volterra/external.ipynb
	jupytext --quiet --execute docs/examples_benchmarks/benchmarks/pleiades/external.ipynb
	jupytext --quiet --execute docs/examples_benchmarks/benchmarks/stiff_van_der_pol/external.ipynb
	jupytext --quiet --execute docs/examples_benchmarks/benchmarks/hires/external.ipynb
	jupytext --quiet --sync docs/examples_benchmarks/benchmarks/lotka_volterra/*
	jupytext --quiet --sync docs/examples_benchmarks/benchmarks/pleiades/*
	jupytext --quiet --sync docs/examples_benchmarks/benchmarks/stiff_van_der_pol/*
	jupytext --quiet --sync docs/examples_benchmarks/benchmarks/hires/*

clean:
	pre-commit clean
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf *.egg-info
	rm -rf dist site build
	rm -rf *.ipynb_checkpoints
	rm -rf docs/examples_benchmarks/benchmarks/lotka_volterra/__pycache__
	rm -rf docs/examples_benchmarks/benchmarks/lotka_volterra/.ipynb_checkpoints
	rm -rf docs/examples_benchmarks/benchmarks/pleiades/__pycache__
	rm -rf docs/examples_benchmarks/benchmarks/pleiades/.ipynb_checkpoints
	rm -rf docs/examples_benchmarks/benchmarks/stiff_van_der_pol/__pycache__
	rm -rf docs/examples_benchmarks/benchmarks/stiff_van_der_pol/.ipynb_checkpoints
	rm -rf docs/examples_benchmarks/benchmarks/hires/__pycache__
	rm -rf docs/examples_benchmarks/benchmarks/hires/.ipynb_checkpoints

doc:
	mkdocs build
