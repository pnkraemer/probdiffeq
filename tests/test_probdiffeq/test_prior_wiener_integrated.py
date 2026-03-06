"""Tests for integrated Wiener processes."""

from probdiffeq import probdiffeq
from probdiffeq.backend import np, testing


@testing.parametrize("ssm_fact", ["isotropic", "dense"])
def test_transitions_are_correct_in_1d(ssm_fact) -> None:
    tcoeffs = [2.0, 3.0, 4.0, 5.0]
    _init, ssm = probdiffeq.ssm_taylor(tcoeffs, ssm_fact=ssm_fact)
    iwp = probdiffeq.prior_wiener_integrated(ssm=ssm)

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
    assert testing.allclose(cond.noise.cholesky @ cond.noise.cholesky.T, Q_expected)


# Separate test because conditional shapes differ
def test_transitions_are_correct_in_1d_blockdiag() -> None:
    tcoeffs = [2.0, 3.0, 4.0, 5.0]
    _init, ssm = probdiffeq.ssm_taylor(tcoeffs, ssm_fact="blockdiag")
    iwp = probdiffeq.prior_wiener_integrated(ssm=ssm)

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
    cov = np.einsum("ijk,ilk->ijl", cond.noise.cholesky, cond.noise.cholesky)
    assert testing.allclose(cov, Q_expected[None, ...])
