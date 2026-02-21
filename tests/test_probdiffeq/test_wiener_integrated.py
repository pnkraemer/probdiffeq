"""Tests for integrated Wiener processes."""

from probdiffeq import probdiffeq
from probdiffeq.backend import np, testing


def test_default_yield_correct_transitions():
    tcoeffs = [2.0, 3.0, 4.0, 5.0]
    init, iwp, ssm = probdiffeq.prior_wiener_integrated(tcoeffs)

    assert testing.allclose(init.std, [0.0, 0.0, 0.0, 0.0])

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


def test_output_scale_inflates_output_scale_correctly():
    tcoeffs = [np.asarray([[[[2.0]]]]), np.asarray([[[[2.0]]]])]

    def scale(s):
        return 123.45 * s

    init, iwp, ssm = probdiffeq.prior_wiener_integrated(tcoeffs, output_scale=scale)

    cond = iwp(1.0, 1.0)
    Q_expected = 123.45**2 * np.asarray(
        [[1.0 / 3.0, 1.0 / 2.0], [1.0 / 2.0, 1.0 / 1.0]]
    )
    assert testing.allclose(cond.noise.cholesky @ cond.noise.cholesky.T, Q_expected)


def test_add_derivatives_adds_derivatives_correctly():
    tcoeffs = [2.0, 3.0]
    init, iwp, ssm = probdiffeq.prior_wiener_integrated(tcoeffs, add_derivatives=3)
    assert testing.allclose(init.mean, [2.0, 3.0, 0.0, 0.0, 0.0])
    assert testing.allclose(init.std, [0.0, 0.0, 1.0, 1.0, 1.0])


def test_diffuse_initialisation_yields_correct_stdevs():
    tcoeffs = [2.0, 3.0]
    tcoeffs_std = [4.0, 5.0]
    init, iwp, ssm = probdiffeq.prior_wiener_integrated_diffuse(tcoeffs, tcoeffs_std)
    assert testing.allclose(init.mean, [2.0, 3.0])
    assert testing.allclose(init.std, [4.0, 5.0])


def test_differential_variable_inflates_std():
    tcoeffs = [np.asarray([1.0, 2.0, 3.0])]
    is_differential_variable = [np.asarray([True, False, True])]
    init, iwp, ssm = probdiffeq.prior_wiener_integrated(
        tcoeffs, is_differential_variable=is_differential_variable
    )

    [m], [s] = init.mean, init.std
    assert testing.allclose(m, np.asarray([1.0, 2.0, 3.0]))
    assert testing.allclose(s == 0, np.asarray([True, False, True]))
    assert testing.allclose(s > 0, np.asarray([False, True, False]))
