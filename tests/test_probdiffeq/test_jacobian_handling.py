"""Tests for Jacobian handlers."""

from probdiffeq import probdiffeq
from probdiffeq.backend import np, testing
from probdiffeq.backend.typing import Array

_JACOBIANS = [
    probdiffeq.jacobian_materialize(),
    probdiffeq.jacobian_monte_carlo_fwd(),
    probdiffeq.jacobian_monte_carlo_rev(),
]


@testing.fixture(name="jacobian_problem")
@testing.parametrize("jacobian", _JACOBIANS)
@testing.parametrize("n_out", [2])
@testing.parametrize("n_in", [3])
@testing.parametrize("d", [5])
def fixture_jacobian_problem(jacobian, n_out, n_in, d):
    """Set up a test function and array input for Jacobian handler tests."""

    def fun(s: Array) -> Array:
        s = np.asarray(s)
        mean = np.mean(s, axis=0)
        return np.stack([mean for _ in range(n_out)])

    x = np.ones((n_in, d))
    state = jacobian.init_jacobian_handler()
    return jacobian, fun, x, state, n_out, n_in, d


def test_materialize_dense(jacobian_problem) -> None:
    """Assert that materialize_dense returns correctly shaped outputs and rejects wrong-shaped inputs."""
    jacobian, fun, x, state, n_out, n_in, d = jacobian_problem

    msg = "Received: "
    with testing.raises(TypeError, match=msg):
        _ = jacobian.materialize_dense(fun, [*x], state)
    with testing.raises(ValueError, match=msg):
        _ = jacobian.materialize_dense(fun, x.reshape((-1,)), state)

    fx, J, _ = jacobian.materialize_dense(fun, x, state)
    assert fx.shape == (n_out, d)
    assert J.shape == (n_out, d, n_in, d)


def test_calculate_trace_along_d(jacobian_problem) -> None:
    """Assert that calculate_trace_along_d returns a correctly shaped trace and rejects wrong-shaped inputs."""
    jacobian, fun, x, state, n_out, n_in, d = jacobian_problem

    msg = "Received: "
    with testing.raises(TypeError, match=msg):
        _ = jacobian.calculate_trace_along_d(fun, [*x], state)
    with testing.raises(ValueError, match=msg):
        _ = jacobian.calculate_trace_along_d(fun, x.reshape((-1,)), state)

    fx, J_trace, _state = jacobian.calculate_trace_along_d(fun, x, state)
    assert fx.shape == (n_out, d)
    assert J_trace.shape == (n_out, n_in)


def test_calculate_diagonal_along_d(jacobian_problem) -> None:
    """Assert that calculate_diagonal_along_d returns a correctly shaped diagonal and rejects wrong-shaped inputs."""
    jacobian, fun, x, state, n_out, n_in, d = jacobian_problem

    msg = "Received: "
    with testing.raises(TypeError, match=msg):
        _ = jacobian.calculate_diagonal_along_d(fun, [*x], state)
    with testing.raises(ValueError, match=msg):
        _ = jacobian.calculate_diagonal_along_d(fun, x.reshape((-1,)), state)

    fx, J_diag, _state = jacobian.calculate_diagonal_along_d(fun, x, state)
    assert fx.shape == (n_out, d)
    assert J_diag.shape == (d, n_out, n_in)
