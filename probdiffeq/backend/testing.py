"""Test-utilities.

In this file, we pick-and-mix functionality from pytest_cases and pytest.

Some of the functionality provided by both libraries overlaps,
and without bundling them here, choices between
(e.g.) pytest.fixture and pytest_cases.fixture
have been very inconsistent.
This is not good for extendability of the test suite.
"""

import pytest
import pytest_cases

case = pytest_cases.case
filterwarnings = pytest.mark.filterwarnings
fixture = pytest_cases.fixture
parametrize = pytest.mark.parametrize
parametrize_with_cases = pytest_cases.parametrize_with_cases
raises = pytest.raises
warns = pytest.warns
skip = pytest.skip
