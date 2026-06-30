"""Tests for integrated Wiener processes."""

from probdiffeq import probdiffeq
from probdiffeq.backend import np, testing


@testing.parametrize(
    "ssm_factory",
    [probdiffeq.state_space_model_isotropic, probdiffeq.state_space_model_dense],
)
@testing.parametrize("dt", [1.234, -1.234])
def test_transitions_are_correct_in_1d(ssm_factory, dt) -> None:
    """Assert that the IWP transition matrix and noise covariance match the analytical formulas in 1d."""
    tcoeffs = [2.0, 3.0, 4.0, 5.0]
    ssm = ssm_factory()
    iwp = ssm.prior_wiener_integrated(tcoeffs)
    cond = iwp.transition(dt=dt, output_scale=1.0).preconditioner_apply()

    # A(dt)_{ij} = dt^{j-i} / (j-i)!   (depends on signed dt)
    A_expected = np.asarray(
        [
            [1.0, dt, dt**2 / 2.0, dt**3 / 6.0],
            [0.0, 1.0, dt, dt**2 / 2.0],
            [0.0, 0.0, 1.0, dt],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    assert testing.allclose(cond.A, A_expected)

    # Q(dt)_{ij} = |dt|^{2*q+1-i-j} / ((q-i)! (q-j)! (2*q+1-i-j)) (depends on |dt|)
    h = abs(dt)
    Q_expected = np.asarray(
        [
            [h**7 / 252.0, h**6 / 72.0, h**5 / 30.0, h**4 / 24.0],
            [h**6 / 72.0, h**5 / 20.0, h**4 / 8.0, h**3 / 6.0],
            [h**5 / 30.0, h**4 / 8.0, h**3 / 3.0, h**2 / 2.0],
            [h**4 / 24.0, h**3 / 6.0, h**2 / 2.0, h**1 / 1.0],
        ]
    )

    q_received, Q_received = cond.noise.to_multivariate_normal()
    assert testing.allclose(q_received, np.zeros((4,)))
    assert testing.allclose(Q_received, Q_expected)


# Separate test for blockdiag because output-scale shapes differ
@testing.parametrize("dt", [1.234, -1.234])
def test_transitions_are_correct_in_1d_blockdiag(dt) -> None:
    """Assert that the IWP transition matrix and noise covariance match the analytical formulas in 1d."""
    tcoeffs = [2.0, 3.0, 4.0, 5.0]
    ssm = probdiffeq.state_space_model_blockdiag()
    iwp = ssm.prior_wiener_integrated(tcoeffs)
    cond = iwp.transition(dt=dt, output_scale=np.ones((1,))).preconditioner_apply()

    # A(dt)_{ij} = dt^{j-i} / (j-i)!   (signed dt; backward step inverts automatically)
    A_expected = np.asarray(
        [
            [1.0, dt, dt**2 / 2.0, dt**3 / 6.0],
            [0.0, 1.0, dt, dt**2 / 2.0],
            [0.0, 0.0, 1.0, dt],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    # Q(dt)_{ij} = |dt|^{7-i-j} / ((3-i)! (3-j)! (7-i-j))   (magnitude; PSD for both signs)
    h = abs(dt)
    Q_expected = np.asarray(
        [
            [h**7 / 252.0, h**6 / 72.0, h**5 / 30.0, h**4 / 24.0],
            [h**6 / 72.0, h**5 / 20.0, h**4 / 8.0, h**3 / 6.0],
            [h**5 / 30.0, h**4 / 8.0, h**3 / 3.0, h**2 / 2.0],
            [h**4 / 24.0, h**3 / 6.0, h**2 / 2.0, h**1 / 1.0],
        ]
    )

    assert testing.allclose(cond.A, A_expected[None, ...])

    q_received, Q_received = cond.noise.to_multivariate_normal()
    assert testing.allclose(q_received, np.zeros((4,)))
    assert testing.allclose(Q_received, Q_expected)
