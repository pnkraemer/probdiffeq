"""Tests for integrated Wiener processes."""

from probdiffeq import probdiffeq
from probdiffeq.backend import np, testing


@testing.parametrize("ssm_fact", ["isotropic", "dense"])
def test_transitions_are_correct_in_1d(ssm_fact) -> None:
    tcoeffs = [2.0, 3.0, 4.0, 5.0]
    ssm = probdiffeq.state_space_model(ssm_fact=ssm_fact)
    _init, iwp = probdiffeq.prior_wiener_integrated(tcoeffs, ssm=ssm)

    cond = iwp(1.0, 1.0)
    A_expected = np.asarray(
        [
            [1.0, 3.0, 3.0, 1.0],
            [0.0, 1.0, 2.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    Q_expected = np.asarray(
        [
            [1.0 / 7.0, 1.0 / 6.0, 1.0 / 5.0, 1.0 / 4.0],
            [1.0 / 6.0, 1.0 / 5.0, 1.0 / 4.0, 1.0 / 3.0],
            [1.0 / 5.0, 1.0 / 4.0, 1.0 / 3.0, 1.0 / 2.0],
            [1.0 / 4.0, 1.0 / 3.0, 1.0 / 2.0, 1.0 / 1.0],
        ]
    )
    assert testing.allclose(cond.A, A_expected)
    assert testing.allclose(
        cond.noise.cholesky_flat @ cond.noise.cholesky_flat.T, Q_expected
    )


# Separate test because conditional shapes differ
def test_transitions_are_correct_in_1d_blockdiag() -> None:
    tcoeffs = [2.0, 3.0, 4.0, 5.0]
    ssm = probdiffeq.state_space_model(ssm_fact="blockdiag")
    _init, iwp = probdiffeq.prior_wiener_integrated(tcoeffs, ssm=ssm)

    cond = iwp(1.0, np.ones((1,)))
    A_expected = np.asarray(
        [
            [1.0, 3.0, 3.0, 1.0],
            [0.0, 1.0, 2.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    Q_expected = np.asarray(
        [
            [1.0 / 7.0, 1.0 / 6.0, 1.0 / 5.0, 1.0 / 4.0],
            [1.0 / 6.0, 1.0 / 5.0, 1.0 / 4.0, 1.0 / 3.0],
            [1.0 / 5.0, 1.0 / 4.0, 1.0 / 3.0, 1.0 / 2.0],
            [1.0 / 4.0, 1.0 / 3.0, 1.0 / 2.0, 1.0 / 1.0],
        ]
    )
    assert testing.allclose(cond.A, A_expected[None, ...])
    cov = np.einsum("ijk,ilk->ijl", cond.noise.cholesky_flat, cond.noise.cholesky_flat)
    assert testing.allclose(cov, Q_expected[None, ...])
