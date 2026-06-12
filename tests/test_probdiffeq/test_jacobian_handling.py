"""Tests for Jacobian handlers."""

from probdiffeq import probdiffeq
from probdiffeq.backend import np, testing


@testing.parametrize(
    "jacobian",
    [
        probdiffeq.jacobian_materialize(),
        probdiffeq.jacobian_monte_carlo_fwd(),
        probdiffeq.jacobian_monte_carlo_rev(),
    ],
)
@testing.parametrize("n", [1, 3])
@testing.parametrize("d", [1, 2])
def test_jacobian_materialize_dense(jacobian: probdiffeq.Jacobian, n: int, d: int):

    def vf_nd(vec_nd):
        vec_nxd = vec_nd.reshape((n, d))
        return np.mean(vec_nxd, axis=0)

    vec = np.arange(0, n * d)
    vec /= np.mean(vec)

    state = jacobian.init_jacobian_handler(num_tcoeffs=n, d=d)

    # Assert that if vecs of the wrong shape are passed, errors are raised.
    msg = "expects a flat"
    with testing.raises(ValueError, match=msg):
        _ = jacobian.materialize_dense(vf_nd, vec.reshape((d, n)), state)

    with testing.raises(ValueError, match=msg):
        _ = jacobian.materialize_dense(vf_nd, vec.reshape((d, n)).T, state)

    # Assert that when the right input is passed, the Jacobian has the expected shape
    _fx, J, _ = jacobian.materialize_dense(vf_nd, vec, state)

    assert J.shape == (d, n * d)


@testing.parametrize(
    "jacobian",
    [
        probdiffeq.jacobian_materialize(),
        probdiffeq.jacobian_monte_carlo_fwd(),
        probdiffeq.jacobian_monte_carlo_rev(),
    ],
)
@testing.parametrize("n", [1, 3])
@testing.parametrize("d", [1, 2])
def test_jacobian_calculate_trace_along_d(
    jacobian: probdiffeq.Jacobian, n: int, d: int
):

    def vf_nxd(vec_nxd):
        return np.mean(vec_nxd, axis=0)

    vec = np.arange(0, n * d).reshape((n, d))
    vec /= np.mean(vec)

    state = jacobian.init_jacobian_handler(num_tcoeffs=n, d=d)

    # Assert that if vecs of the wrong shape are passed, errors are raised.
    msg = "expects an nxd"
    with testing.raises(ValueError, match=msg):
        _ = jacobian.calculate_trace_along_d(vf_nxd, vec.reshape((d * n,)), state)

    if d != n:
        with testing.raises(ValueError, match=msg):
            _ = jacobian.calculate_trace_along_d(vf_nxd, vec.reshape((d, n)), state)

    # Assert that when the right input is passed, the Jacobian has the expected shape
    _fx, J, _ = jacobian.calculate_trace_along_d(vf_nxd, vec, state)

    assert J.shape == (n,)


@testing.parametrize(
    "jacobian",
    [
        probdiffeq.jacobian_materialize(),
        probdiffeq.jacobian_monte_carlo_fwd(),
        probdiffeq.jacobian_monte_carlo_rev(),
    ],
)
@testing.parametrize("n", [1, 3])
@testing.parametrize("d", [1, 2])
def test_jacobian_calculate_diagonal_along_d(
    jacobian: probdiffeq.Jacobian, n: int, d: int
):

    def vf_dxn(vec_dxn):
        return np.mean(vec_dxn, axis=-1)

    vec = np.arange(0, n * d).reshape((d, n))
    vec /= np.mean(vec)

    state = jacobian.init_jacobian_handler(num_tcoeffs=n, d=d)

    # Assert that if vecs of the wrong shape are passed, errors are raised.
    msg = "expects a dxn"
    with testing.raises(ValueError, match=msg):
        _ = jacobian.calculate_diagonal_along_d(vf_dxn, vec.reshape((d * n,)), state)

    if d != n:
        with testing.raises(ValueError, match=msg):
            _ = jacobian.calculate_diagonal_along_d(vf_dxn, vec.reshape((n, d)), state)

    # Assert that when the right input is passed, the Jacobian has the expected shape
    _fx, J, _ = jacobian.calculate_diagonal_along_d(vf_dxn, vec, state)

    assert J.shape == (d, n)
