
.PHONY: format lint test doc example run-benchmarks clean

format:
	black --quiet .
	jupytext --quiet --sync docs/quickstart/*.ipynb
	jupytext --quiet --sync docs/examples_solver_config/*.ipynb
	jupytext --quiet --sync docs/examples_parameter_estimation/*.ipynb
	jupytext --quiet --sync docs/benchmarks/hires/*.ipynb
	jupytext --quiet --sync docs/benchmarks/pleiades/*.ipynb
	jupytext --quiet --sync docs/benchmarks/vanderpol/*.ipynb
	jupytext --quiet --sync docs/benchmarks/lotkavolterra/*.ipynb

lint:
	pre-commit run --all-files

test:
	IMPL=dense pytest -n auto -v # parallelise, verbose output
	IMPL=isotropic pytest -n auto -v # parallelise, verbose output
	IMPL=blockdiag pytest -n auto -v # parallelise, verbose output
	IMPL=scalar pytest -n auto -v # parallelise, verbose output

example:
	jupytext --quiet --sync docs/quickstart/*.ipynb
	jupytext --quiet --execute docs/quickstart/*.ipynb
	jupytext --quiet --sync docs/examples_solver_config/*
	jupytext --quiet --execute docs/examples_solver_config/*
	jupytext --quiet --sync docs/examples_solver_config/*
	jupytext --quiet --sync docs/examples_parameter_estimation/*
	jupytext --quiet --execute docs/examples_parameter_estimation/*
	jupytext --quiet --sync docs/examples_parameter_estimation/*

run-benchmarks:
	time python docs/benchmarks/lotkavolterra/run_lotkavolterra.py --start 3 --stop 12 --repeats 20  --save True
	jupytext --quiet --sync docs/benchmarks/lotkavolterra/*.ipynb
	jupytext --quiet --execute docs/benchmarks/lotkavolterra/*.ipynb
	time python docs/benchmarks/vanderpol/run_vanderpol.py --start 1 --stop 9 --repeats 3  --save True
	jupytext --quiet --sync docs/benchmarks/vanderpol/*.ipynb
	jupytext --quiet --execute docs/benchmarks/vanderpol/*.ipynb
	time python docs/benchmarks/pleiades/run_pleiades.py --start 3 --stop 11 --repeats 3  --save True
	jupytext --quiet --sync docs/benchmarks/pleiades/*.ipynb
	jupytext --quiet --execute docs/benchmarks/pleiades/*.ipynb
	time python docs/benchmarks/hires/run_hires.py --start 1 --stop 9 --repeats 10  --save True
	jupytext --quiet --sync docs/benchmarks/hires/*.ipynb
	jupytext --quiet --execute docs/benchmarks/hires/*.ipynb

dry-run-benchmarks:
	time python docs/benchmarks/lotkavolterra/run_lotkavolterra.py --start 1 --stop 3 --repeats 2  --save False
	time python docs/benchmarks/vanderpol/run_vanderpol.py --start 1 --stop 3 --repeats 2  --save False
	time python docs/benchmarks/pleiades/run_pleiades.py --start 3 --stop 5 --repeats 2  --save False
	time python docs/benchmarks/hires/run_hires.py --start 1 --stop 3 --repeats 2  --save False

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
	rm -rf docs/benchmarks/hires/__pycache__
	rm -rf docs/benchmarks/hires/.ipynb_checkpoints
	rm docs/benchmarks/hires/*.npy

doc:
	mkdocs build
