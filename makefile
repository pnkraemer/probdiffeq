
.PHONY: format lint test doc example run-benchmarks clean

format:
	ruff format --quiet .
	jupytext --quiet --sync docs/getting_started/*.ipynb
	jupytext --quiet --sync docs/examples_solver_config/*.ipynb
	jupytext --quiet --sync docs/examples_parameter_estimation/*.ipynb
	jupytext --quiet --sync docs/benchmarks/hires/*.ipynb
	jupytext --quiet --sync docs/benchmarks/pleiades/*.ipynb
	jupytext --quiet --sync docs/benchmarks/vanderpol/*.ipynb
	jupytext --quiet --sync docs/benchmarks/lotkavolterra/*.ipynb
	jupytext --quiet --sync docs/benchmarks/taylor_pleiades/*.ipynb
	jupytext --quiet --sync docs/benchmarks/taylor_fitzhughnagumo/*.ipynb
	jupytext --quiet --sync docs/benchmarks/taylor_node/*.ipynb

lint:
	pre-commit run --all-files

test:
	IMPL=dense pytest -n auto -v # parallelise, verbose output
	IMPL=isotropic pytest -n auto -v # parallelise, verbose output
	IMPL=blockdiag pytest -n auto -v # parallelise, verbose output
	IMPL=scalar pytest -n auto -v # parallelise, verbose output

example:
	jupytext --quiet --sync docs/getting_started/*.ipynb
	jupytext --quiet --execute docs/getting_started/*.ipynb
	jupytext --quiet --sync docs/examples_solver_config/*
	jupytext --quiet --execute docs/examples_solver_config/*
	jupytext --quiet --sync docs/examples_solver_config/*
	jupytext --quiet --sync docs/examples_parameter_estimation/*
	jupytext --quiet --execute docs/examples_parameter_estimation/*
	jupytext --quiet --sync docs/examples_parameter_estimation/*

run-benchmarks:
	time python docs/benchmarks/taylor_node/run_taylor_node.py --max_time 12 --repeats 3 --save
	jupytext --quiet --sync docs/benchmarks/taylor_node/*.ipynb
	jupytext --quiet --execute docs/benchmarks/taylor_node/*.ipynb
	time python docs/benchmarks/taylor_pleiades/run_taylor_pleiades.py --max_time 15 --repeats 5 --save
	jupytext --quiet --sync docs/benchmarks/taylor_pleiades/*.ipynb
	jupytext --quiet --execute docs/benchmarks/taylor_pleiades/*.ipynb
	time python docs/benchmarks/taylor_fitzhughnagumo/run_taylor_fitzhughnagumo.py --max_time 15 --repeats 15 --save
	jupytext --quiet --sync docs/benchmarks/taylor_fitzhughnagumo/*.ipynb
	jupytext --quiet --execute docs/benchmarks/taylor_fitzhughnagumo/*.ipynb
	time python docs/benchmarks/lotkavolterra/run_lotkavolterra.py --start 3 --stop 12 --repeats 20  --save
	jupytext --quiet --sync docs/benchmarks/lotkavolterra/*.ipynb
	jupytext --quiet --execute docs/benchmarks/lotkavolterra/*.ipynb
	time python docs/benchmarks/vanderpol/run_vanderpol.py --start 1 --stop 9 --repeats 3  --save
	jupytext --quiet --sync docs/benchmarks/vanderpol/*.ipynb
	jupytext --quiet --execute docs/benchmarks/vanderpol/*.ipynb
	time python docs/benchmarks/pleiades/run_pleiades.py --start 3 --stop 11 --repeats 3  --save
	jupytext --quiet --sync docs/benchmarks/pleiades/*.ipynb
	jupytext --quiet --execute docs/benchmarks/pleiades/*.ipynb
	time python docs/benchmarks/hires/run_hires.py --start 1 --stop 9 --repeats 10  --save
	jupytext --quiet --sync docs/benchmarks/hires/*.ipynb
	jupytext --quiet --execute docs/benchmarks/hires/*.ipynb

dry-run-benchmarks:
	time python docs/benchmarks/taylor_node/run_taylor_node.py --max_time 0.5 --repeats 2 --no-save
	time python docs/benchmarks/taylor_fitzhughnagumo/run_taylor_fitzhughnagumo.py --max_time 0.5 --repeats 2 --no-save
	time python docs/benchmarks/taylor_pleiades/run_taylor_pleiades.py --max_time 0.5 --repeats 2 --no-save
	time python docs/benchmarks/taylor_fitzhughnagumo/run_taylor_fitzhughnagumo.py --max_time 0.5 --repeats 2 --no-save
	time python docs/benchmarks/lotkavolterra/run_lotkavolterra.py --start 3 --stop 5 --repeats 2 --no-save
	time python docs/benchmarks/vanderpol/run_vanderpol.py --start 1 --stop 3 --repeats 2  --no-save
	time python docs/benchmarks/pleiades/run_pleiades.py --start 3 --stop 5 --repeats 2  --no-save
	time python docs/benchmarks/hires/run_hires.py --start 1 --stop 3 --repeats 2  --no-save

clean:
	pre-commit clean
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf *.egg-info
	rm -rf dist site build
	rm -rf *.ipynb_checkpoints
	rm -rf docs/benchmarks/hires/__pycache__
	rm -rf docs/benchmarks/hires/.ipynb_checkpoints
	rm -rf docs/benchmarks/pleiades/__pycache__
	rm -rf docs/benchmarks/pleiades/.ipynb_checkpoints
	rm -rf docs/benchmarks/lotkavolterra/__pycache__
	rm -rf docs/benchmarks/lotkavolterra/.ipynb_checkpoints
	rm -rf docs/benchmarks/vanderpol/__pycache__
	rm -rf docs/benchmarks/vanderpol/.ipynb_checkpoints
	rm -rf docs/benchmarks/taylor_pleiades/__pycache__
	rm -rf docs/benchmarks/taylor_pleiades/.ipynb_checkpoints
	rm -rf docs/benchmarks/taylor_fitzhughnagumo/__pycache__
	rm -rf docs/benchmarks/taylor_fitzhughnagumo/.ipynb_checkpoints
	rm -rf docs/benchmarks/taylor_node/__pycache__
	rm -rf docs/benchmarks/taylor_node/.ipynb_checkpoints

doc:
	mkdocs build
