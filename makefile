format-and-lint:
	pre-commit run --all-files

test:
	python -m doctest probdiffeq/_probdiffeq/problem_types.py
	python -m doctest probdiffeq/_probdiffeq/linearization_points.py
	pytest -n auto -v -Werror		# parallelise, verbose output, warnings as errors

quickstart:
	# Run some code without installing any of the optional dependencies
	# Otherwise, it's unclear whether the listed main dependencies
	# are specified correctly. This avoids issues like
	# https://github.com/pnkraemer/probdiffeq/issues/810
	python scripts/examples_and_benchmarks_to_py_light.py
	python docs/Examples/A0*.py
	python docs/Examples/A1*.py


clean:
	pre-commit clean
	git clean -xdf

doc:
	python scripts/generate_api_docs.py
	python scripts/readme_to_dev_docs.py
	python scripts/examples_and_benchmarks_to_py_light.py
	# The following line executes the examples and benchmarks
	JUPYTER_PLATFORM_DIRS=1 mkdocs build

doc-serve:
	python scripts/generate_api_docs.py
	python scripts/readme_to_dev_docs.py
	python scripts/examples_and_benchmarks_to_py_light.py
	# The following line executes the examples and benchmarks
	JUPYTER_PLATFORM_DIRS=1 mkdocs serve

find-dead-code:
	vulture . --ignore-names case*,fixture*,*jvp --exclude probdiffeq/_version.py,build/**/*.py
