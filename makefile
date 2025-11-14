format-and-lint:
	pre-commit run --all-files

test:
	pytest -n auto -v -Werror		# parallelise, verbose output, warnings as errors

quickstart:
	# Run some code without installing any of the optional dependencies
	# Otherwise, it's unclear whether the listed main dependencies
	# are specified correctly. This avoids issues like
	# https://github.com/pnkraemer/probdiffeq/issues/810
	python docs/examples_quickstart/*.py

example-and-benchmark:
	jupytext --quiet --to ipynb --update docs/examples*/*.py
	jupytext --execute docs/examples*/*.ipynb


clean:
	pre-commit clean
	git clean -xdf

doc:
	# The readme is the landing page of the docs:
	cp README.md docs/index.md
	# Execute the examples manually and not via mkdocs-jupyter
	# to gain clear error messages.
	make example
	make benchmarks-plot-results
	JUPYTER_PLATFORM_DIRS=1 mkdocs build

doc-serve:
	# The readme is the landing page of the docs:
	cp README.md docs/index.md
	# Execute the examples manually and not via mkdocs-jupyter
	# to gain clear error messages.
	make example
	make benchmarks-plot-results
	JUPYTER_PLATFORM_DIRS=1 mkdocs serve

find-dead-code:
	vulture . --ignore-names case*,fixture*,*jvp --exclude probdiffeq/_version.py
