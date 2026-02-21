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


def test_diffuse_derivatives():
    tcoeffs = [2.0, 3.0]
    init, iwp, ssm = probdiffeq.prior_wiener_integrated(
        tcoeffs, diffuse_derivatives=3, diffuse_eps=123.0
    )
    assert testing.allclose(init.mean, [2.0, 3.0, 0.0, 0.0, 0.0])
    assert testing.allclose(init.std, [0.0, 0.0, 123.0, 123.0, 123.0])


def test_differential_variables_dense():
    tcoeffs = [np.asarray([1.0, 2.0, 3.0])]
    isdiff = [np.asarray([True, False, True])]
    init, iwp, ssm = probdiffeq.prior_wiener_integrated(
        tcoeffs, is_differential=isdiff, nondifferential_eps=0.123
    )

    [m], [s] = init.mean, init.std
    assert testing.allclose(m, np.asarray([1.0, 2.0, 3.0]))
    assert testing.allclose(s, np.asarray([0.0, 0.123, 0.0]))

    with testing.raises(ValueError, match="wrong PyTree structure"):
        tcoeffs = [np.asarray([1.0, 2.0, 3.0])]
        isdiff = [np.asarray(False)]  # wrong shape
        _ = probdiffeq.prior_wiener_integrated(tcoeffs, is_differential=isdiff)


def test_output_scale_dense():

    # Non-(d,) shape. Values don't matter.
    tcoeffs = [np.ones((1, 1, 1)), np.ones((1, 1, 1))]

    def scale(s):
        return 123.45 * s

    init, iwp, ssm = probdiffeq.prior_wiener_integrated(tcoeffs, output_scale=scale)

    cond = iwp(1.0, 1.0)
    Q_expected = 123.45**2.0 * 1.0 / np.asarray([[3.0, 2.0], [2.0, 1.0]])
    assert testing.allclose(cond.noise.cholesky @ cond.noise.cholesky.T, Q_expected)

    init, iwp, ssm = probdiffeq.prior_wiener_integrated(
        tcoeffs, output_scale=scale, diffuse_derivatives=3, diffuse_eps=1
    )

    zero = np.zeros((1, 1, 1))
    nonzero = 123.45 * np.ones((1, 1, 1))
    assert testing.allclose(
        init.std, [zero, zero, nonzero, nonzero, nonzero], strict_shapes=True
    )
