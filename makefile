
.PHONY: format lint test pre-commit doc clean

format:
	isort .
	black .
	nbqa black docs/
	nbqa isort docs/

lint:
	isort --check --diff .
	black --check --diff .
	nbqa isort --check --diff .
	nbqa black --check --diff .
	nbqa flake8 docs/

test:
	pytest

example:

pre-commit:
	pre-commit autoupdate
	pre-commit run --all-files

clean:
	rm -rf .pytest_cache
	rm -rf *.egg-info
	rm -rf dist site
