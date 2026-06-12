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
@testing.parametrize("n", [3])
@testing.parametrize("d", [2])
def test_jacobian_materialize_dense(jacobian: probdiffeq.Jacobian, n: int, d: int):

    def vf_nd(vec_nd):
        vec_nxd = vec_nd.reshape((n, d))
        return np.mean(vec_nxd, axis=0)

    vec = np.arange(0, n * d)
    vec /= np.mean(vec)

    state = jacobian.init_jacobian_handler(num_tcoeffs=n, d=d)
    _fx, J, _ = jacobian.materialize_dense(vf_nd, vec, state)

    assert J.shape == (d, n * d)

    # Assert that if vecs of the wrong shape are passed, errors are raised.
    msg = "expects a flat"
    with testing.raises(ValueError, match=msg):
        _ = jacobian.materialize_dense(vf_nd, vec.reshape((d, n)), state)

    with testing.raises(ValueError, match=msg):
        _ = jacobian.materialize_dense(vf_nd, vec.reshape((d, n)).T, state)
