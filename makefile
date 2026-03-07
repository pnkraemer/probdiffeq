format-and-lint:
	pre-commit run --all-files

test:
	pytest -n auto -v -Werror		# parallelise, verbose output, warnings as errors

quickstart:
	# Run some code without installing any of the optional dependencies
	# Otherwise, it's unclear whether the listed main dependencies
	# are specified correctly. This avoids issues like
	# https://github.com/pnkraemer/probdiffeq/issues/810
	python scripts/tutorials_to_py_light.py 
	python docs/Tutorials/A*.py


clean:
	pre-commit clean
	git clean -xdf

doc:
	python scripts/generate_api_docs.py 
	python scripts/readme_to_dev_docs.py 
	python scripts/tutorials_to_py_light.py 
	python scripts/benchmarks_to_py_light.py 
	# Execute the examples and benchmarks manually and not 
	# via mkdocs-jupyter to gain clear error messages.
	JUPYTER_PLATFORM_DIRS=1 mkdocs build

doc-serve:
	python scripts/generate_api_docs.py 
	python scripts/readme_to_dev_docs.py 
	python scripts/tutorials_to_py_light.py 
	python scripts/benchmarks_to_py_light.py 
	# Execute the examples and benchmarks manually and not 
	# via mkdocs-jupyter to gain clear error messages.
	JUPYTER_PLATFORM_DIRS=1 mkdocs serve

find-dead-code:
	vulture . --ignore-names case*,fixture*,*jvp --exclude probdiffeq/_version.py
