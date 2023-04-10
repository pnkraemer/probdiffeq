
.PHONY: format lint test doc example run-benchmarks clean

format:
	isort .
	black .
	nbqa black docs/quickstart/
	nbqa black docs/examples_benchmarks/solvers_solutions/
	nbqa black docs/examples_benchmarks/parameter_estimation/
	nbqa black docs/examples_benchmarks/benchmarks/lotka_volterra/
	nbqa black docs/examples_benchmarks/benchmarks/pleiades/
	nbqa black docs/examples_benchmarks/benchmarks/stiff_van_der_pol/
	nbqa black docs/examples_benchmarks/benchmarks/hires/
	nbqa isort docs/quickstart/
	nbqa isort docs/examples_benchmarks/solvers_solutions/
	nbqa isort docs/examples_benchmarks/parameter_estimation/
	nbqa isort docs/examples_benchmarks/benchmarks/lotka_volterra/
	nbqa isort docs/examples_benchmarks/benchmarks/pleiades/
	nbqa isort docs/examples_benchmarks/benchmarks/stiff_van_der_pol/
	nbqa isort docs/examples_benchmarks/benchmarks/hires/
	jupytext --sync docs/quickstart/*
	jupytext --sync docs/examples_benchmarks/solvers_solutions/*
	jupytext --sync docs/examples_benchmarks/parameter_estimation/*
	jupytext --sync docs/examples_benchmarks/benchmarks/lotka_volterra/*
	jupytext --sync docs/examples_benchmarks/benchmarks/pleiades/*
	jupytext --sync docs/examples_benchmarks/benchmarks/stiff_van_der_pol/*
	jupytext --sync docs/examples_benchmarks/benchmarks/hires/*

lint:
	pre-commit run --all-files

test:
	pytest -n auto -v -s  # parallelise, verbose output, show all 'stdout's
	python -m doctest probdiffeq/*.py

example:
	jupytext --sync docs/quickstart/*.ipynb
	jupytext --execute docs/quickstart/*.ipynb
	jupytext --sync examples_benchmarks/solvers_solutions/*
	jupytext --execute examples_benchmarks/solvers_solutions/*
	jupytext --sync examples_benchmarks/solvers_solutions/*
	jupytext --sync examples_benchmarks/parameter_estimation/*
	jupytext --execute examples_benchmarks/parameter_estimation/*
	jupytext --sync examples_benchmarks/parameter_estimation/*
	# No --execute for advanced examples and benchmarks (takes too long)
	jupytext --sync examples_benchmarks/benchmarks/lotka_volterra/*
	jupytext --sync examples_benchmarks/benchmarks/pleiades/*
	jupytext --sync examples_benchmarks/benchmarks/stiff_van_der_pol/*
	jupytext --sync examples_benchmarks/benchmarks/hires/*

run-benchmarks:
	jupytext --sync examples_benchmarks/benchmarks/lotka_volterra/*
	jupytext --sync examples_benchmarks/benchmarks/pleiades/*
	jupytext --sync examples_benchmarks/benchmarks/stiff_van_der_pol/*
	jupytext --sync examples_benchmarks/benchmarks/hires/*
	jupytext --execute examples_benchmarks/benchmarks/lotka_volterra/internal.ipynb
	jupytext --execute examples_benchmarks/benchmarks/lotka_volterra/external.ipynb
	jupytext --execute examples_benchmarks/benchmarks/pleiades/external.ipynb
	jupytext --execute examples_benchmarks/benchmarks/stiff_van_der_pol/external.ipynb
	jupytext --execute examples_benchmarks/benchmarks/hires/external.ipynb
	jupytext --sync examples_benchmarks/benchmarks/lotka_volterra/*
	jupytext --sync examples_benchmarks/benchmarks/pleiades/*
	jupytext --sync examples_benchmarks/benchmarks/stiff_van_der_pol/*
	jupytext --sync examples_benchmarks/benchmarks/hires/*

clean:
	pre-commit clean
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf *.egg-info
	rm -rf dist site build
	rm -rf *.ipynb_checkpoints
	rm -rf examples_benchmarks/benchmarks/lotka_volterra/__pycache__
	rm -rf examples_benchmarks/benchmarks/lotka_volterra/.ipynb_checkpoints
	rm -rf examples_benchmarks/benchmarks/pleiades/__pycache__
	rm -rf examples_benchmarks/benchmarks/pleiades/.ipynb_checkpoints
	rm -rf examples_benchmarks/benchmarks/stiff_van_der_pol/__pycache__
	rm -rf examples_benchmarks/benchmarks/stiff_van_der_pol/.ipynb_checkpoints
	rm -rf examples_benchmarks/benchmarks/hires/__pycache__
	rm -rf examples_benchmarks/benchmarks/hires/.ipynb_checkpoints

doc:
	mkdocs build
