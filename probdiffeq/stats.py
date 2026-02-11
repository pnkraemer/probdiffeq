"""Interact with IVP solutions.

For example, this module contains functionality to compute off-grid marginals,
or to evaluate marginal likelihoods of observations of the solutions.
"""

from probdiffeq.backend.typing import TypeVar

# TODO: the functions in here should only depend on posteriors / strategies!

T = TypeVar("T")
