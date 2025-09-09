"""Tests for the affine recursion."""

from probdiffeq import taylor
from probdiffeq.backend import numpy as np
from probdiffeq.backend import testing


@testing.parametrize("num", [1, 2, 4])
@testing.parametrize("num_derivatives_max", [5])
def test_affine_recursion(num, num_derivatives_max):
    """The approximation should coincide with the reference."""
    f, init, solution = _affine_problem(num_derivatives_max)
    derivatives = taylor.odejet_affine(f, init, num=num)
    assert len(derivatives) == num + 1
    assert testing.tree_all_allclose(derivatives, solution[: len(derivatives)])


def _affine_problem(n):
    A = np.eye(2) * np.arange(1.0, 5.0).reshape((2, 2))
    b = np.arange(6.0, 8.0)

    def vf(x: list, /):
        [[[x]]] = x
        return [A @ x + b]

    # Assert Pytree compatibility
    init = ([[[np.arange(9.0, 11.0)]]],)

    solution = taylor.odejet_padded_scan(vf, init, num=n)
    return vf, init, solution
