"""Tests for Jacobian handlers."""

from probdiffeq import probdiffeq
from probdiffeq.backend import np, testing
from probdiffeq.backend.typing import Array


@testing.parametrize(
    "jacobian",
    [
        probdiffeq.jacobian_materialize(),
        probdiffeq.jacobian_monte_carlo_fwd(),
        probdiffeq.jacobian_monte_carlo_rev(),
    ],
)
@testing.parametrize("n_out", [2])
@testing.parametrize("n_in", [3])
@testing.parametrize("d", [5])
def test_materialize_dense(
    jacobian: probdiffeq.Jacobian, n_out: int, n_in: int, d: int
):

    def fun(s: Array) -> Array:
        s = np.asarray(s)
        mean = np.mean(s, axis=0)
        return np.stack([mean for _ in range(n_out)])

    x = np.ones((n_in, d))
    state = jacobian.init_jacobian_handler()

    # Assert that if vecs of the wrong shape are passed, errors are raised.
    msg = "Received: "
    with testing.raises(ValueError, match=msg):
        _ = jacobian.materialize_dense(fun, [*x], state)

    with testing.raises(ValueError, match=msg):
        _ = jacobian.materialize_dense(fun, x.reshape((-1,)), state)

    # Assert that when the right input is passed, the Jacobian has the expected shape
    fx, J, _ = jacobian.materialize_dense(fun, x, state)

    assert fx.shape == (n_out, d)
    assert J.shape == (n_out, d, n_in, d)


@testing.parametrize(
    "jacobian",
    [
        probdiffeq.jacobian_materialize(),
        probdiffeq.jacobian_monte_carlo_fwd(),
        probdiffeq.jacobian_monte_carlo_rev(),
    ],
)
@testing.parametrize("n_out", [2])
@testing.parametrize("n_in", [3])
@testing.parametrize("d", [5])
def test_calculate_trace_along_d(
    jacobian: probdiffeq.Jacobian, n_out: int, n_in: int, d: int
):

    def fun(s: Array) -> Array:
        s = np.asarray(s)
        mean = np.mean(s, axis=0)
        return np.stack([mean for _ in range(n_out)])

    x = np.ones((n_in, d))
    state = jacobian.init_jacobian_handler()

    # Assert that if vecs of the wrong shape are passed, errors are raised.
    msg = "Received: "
    with testing.raises(ValueError, match=msg):
        _ = jacobian.materialize_dense(fun, [*x], state)

    with testing.raises(ValueError, match=msg):
        _ = jacobian.materialize_dense(fun, x.reshape((-1,)), state)

    # Assert that when the right input is passed, the Jacobian has the expected shape
    fx, J_trace, _state = jacobian.calculate_trace_along_d(fun, x, state)

    assert fx.shape == (n_out, d)
    assert J_trace.shape == (n_out, n_in)


@testing.parametrize(
    "jacobian",
    [
        probdiffeq.jacobian_materialize(),
        probdiffeq.jacobian_monte_carlo_fwd(),
        probdiffeq.jacobian_monte_carlo_rev(),
    ],
)
@testing.parametrize("n_out", [2])
@testing.parametrize("n_in", [3])
@testing.parametrize("d", [5])
def test_jacobian_calculate_diagonal_along_d(
    jacobian: probdiffeq.Jacobian, n_out: int, n_in: int, d: int
):

    def fun(s: Array) -> Array:
        s = np.asarray(s)
        mean = np.mean(s, axis=0)
        return np.stack([mean for _ in range(n_out)])

    x = np.ones((n_in, d))
    state = jacobian.init_jacobian_handler()

    # Assert that if vecs of the wrong shape are passed, errors are raised.
    msg = "Received: "
    with testing.raises(ValueError, match=msg):
        _ = jacobian.materialize_dense(fun, [*x], state)

    with testing.raises(ValueError, match=msg):
        _ = jacobian.materialize_dense(fun, x.reshape((-1,)), state)

    # Assert that when the right input is passed, the Jacobian has the expected shape
    fx, J_diag, _state = jacobian.calculate_diagonal_along_d(fun, x, state)

    assert fx.shape == (n_out, d)
    assert J_diag.shape == (d, n_out, n_in)
