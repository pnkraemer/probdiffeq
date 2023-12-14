format-and-lint:
	pre-commit run --all-files

test:
	IMPL=dense pytest -n auto -v 		# parallelise, verbose output
	IMPL=isotropic pytest -n auto -v 	# parallelise, verbose output
	IMPL=blockdiag pytest -n auto -v 	# parallelise, verbose output
	IMPL=scalar pytest -n auto -v 		# parallelise, verbose output

example:
	jupytext --quiet --to ipynb --update docs/examples*/*.md
	jupytext --execute docs/examples*/*.ipynb


plot-benchmarks:
	jupytext --quiet --to ipynb --update docs/benchmarks/**/*.md
	jupytext --execute docs/benchmarks/**/*.ipynb

run-benchmarks:
	time python docs/benchmarks/taylor_node/run_taylor_node.py --max_time 12 --repeats 3 --save
	time python docs/benchmarks/taylor_pleiades/run_taylor_pleiades.py --max_time 15 --repeats 5 --save
	time python docs/benchmarks/taylor_fitzhughnagumo/run_taylor_fitzhughnagumo.py --max_time 15 --repeats 15 --save
	time python docs/benchmarks/lotkavolterra/run_lotkavolterra.py --start 3 --stop 12 --repeats 20  --save
	time python docs/benchmarks/vanderpol/run_vanderpol.py --start 1 --stop 9 --repeats 3  --save
	time python docs/benchmarks/pleiades/run_pleiades.py --start 3 --stop 11 --repeats 3  --save
	time python docs/benchmarks/hires/run_hires.py --start 1 --stop 9 --repeats 10  --save
	make plot-benchmarks

dry-run-benchmarks:
	time python docs/benchmarks/taylor_node/run_taylor_node.py --max_time 0.5 --repeats 2 --no-save
	time python docs/benchmarks/taylor_fitzhughnagumo/run_taylor_fitzhughnagumo.py --max_time 0.5 --repeats 2 --no-save
	time python docs/benchmarks/taylor_pleiades/run_taylor_pleiades.py --max_time 0.5 --repeats 2 --no-save
	time python docs/benchmarks/taylor_fitzhughnagumo/run_taylor_fitzhughnagumo.py --max_time 0.5 --repeats 2 --no-save
	time python docs/benchmarks/lotkavolterra/run_lotkavolterra.py --start 3 --stop 5 --repeats 2 --no-save
	time python docs/benchmarks/vanderpol/run_vanderpol.py --start 1 --stop 3 --repeats 2  --no-save
	time python docs/benchmarks/pleiades/run_pleiades.py --start 3 --stop 5 --repeats 2  --no-save
	time python docs/benchmarks/hires/run_hires.py --start 1 --stop 3 --repeats 2  --no-save
	make plot-benchmarks


clean:
	pre-commit clean
	git clean -xdf

doc:
	# Execute the examples manually and not via mkdocs-jupyter
	# to gain clear error messages.
	make example
	make plot-benchmarks
	mkdocs build
