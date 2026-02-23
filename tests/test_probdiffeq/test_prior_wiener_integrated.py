"""Tests for integrated Wiener processes."""

from probdiffeq import probdiffeq
from probdiffeq.backend import np, testing


def test_default_yield_correct_transitions():
    tcoeffs = [2.0, 3.0, 4.0, 5.0]
    init, iwp, _ssm = probdiffeq.prior_wiener_integrated(tcoeffs)

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
    init, _iwp, _ssm = probdiffeq.prior_wiener_integrated(
        tcoeffs, diffuse_derivatives=3, diffuse_eps=123.0
    )
    assert testing.allclose(init.mean, [2.0, 3.0, 0.0, 0.0, 0.0])
    assert testing.allclose(init.std, [0.0, 0.0, 123.0, 123.0, 123.0])


def test_differential_variables_dense():
    tcoeffs = [np.asarray([1.0, 2.0, 3.0])]
    isdiff = [np.asarray([True, False, True])]
    init, _iwp, _ssm = probdiffeq.prior_wiener_integrated(
        tcoeffs, is_differential=isdiff, nondifferential_eps=0.123, ssm_fact="dense"
    )

    [m], [s] = init.mean, init.std
    assert testing.allclose(m, np.asarray([1.0, 2.0, 3.0]))
    assert testing.allclose(s, np.asarray([0.0, 0.123, 0.0]))

    with testing.raises(ValueError, match="wrong PyTree structure"):
        tcoeffs = [np.asarray([1.0, 2.0, 3.0])]
        isdiff = [np.asarray(False)]  # wrong shape
        _ = probdiffeq.prior_wiener_integrated(tcoeffs, is_differential=isdiff)


def test_differential_variables_blockdiag():
    tcoeffs = [np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, 2.0, 3.0])]
    isdiff = [np.asarray([True, False, True]), np.asarray([False, False, True])]
    init, _iwp, _ssm = probdiffeq.prior_wiener_integrated(
        tcoeffs, is_differential=isdiff, nondifferential_eps=0.123, ssm_fact="blockdiag"
    )

    [m1, m2], [s1, s2] = init.mean, init.std
    assert testing.allclose(m1, np.asarray([1.0, 2.0, 3.0]))
    assert testing.allclose(m2, np.asarray([1.0, 2.0, 3.0]))
    assert testing.allclose(s1, np.asarray([0.0, 0.123, 0.0]))
    assert testing.allclose(s2, np.asarray([0.123, 0.123, 0.0]))

    with testing.raises(ValueError, match="wrong PyTree structure"):
        tcoeffs = [np.asarray([1.0, 2.0, 3.0])]
        isdiff = [np.asarray(False)]  # wrong shape
        _ = probdiffeq.prior_wiener_integrated(
            tcoeffs, is_differential=isdiff, ssm_fact="blockdiag"
        )


def test_differential_variables_isotropic():
    tcoeffs = [np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, 2.0, 3.0])]
    isdiff = [np.asarray(True), np.asarray(False)]
    init, _iwp, _ssm = probdiffeq.prior_wiener_integrated(
        tcoeffs, is_differential=isdiff, nondifferential_eps=0.123, ssm_fact="isotropic"
    )

    [m1, m2], [s1, s2] = init.mean, init.std
    assert testing.allclose(m1, np.asarray([1.0, 2.0, 3.0]))
    assert testing.allclose(m2, np.asarray([1.0, 2.0, 3.0]))
    assert testing.allclose(s1, np.asarray(0.0))
    assert testing.allclose(s2, np.asarray(0.123))

    with testing.raises(ValueError, match="wrong PyTree structure"):
        tcoeffs = [np.asarray([1.0, 2.0, 3.0])]
        isdiff = [np.asarray([True, False, True])]  # wrong shape
        _ = probdiffeq.prior_wiener_integrated(
            tcoeffs, is_differential=isdiff, ssm_fact="isotropic"
        )


@testing.parametrize("ssm_fact", ["dense", "blockdiag"])
def test_output_scale_dense_blockdiag(ssm_fact):

    # 1d problem, but "unusual" shapes. Values don't matter.
    tcoeffs = [np.ones((1, 1, 1)), np.ones((1, 1, 1))]
    scale = 123.45 * np.ones((1, 1, 1))

    # Test that the transition covariances are scaled correctly
    init, iwp, ssm = probdiffeq.prior_wiener_integrated(
        tcoeffs, output_scale=scale, ssm_fact=ssm_fact
    )

    cond = iwp(1.0, 1.0)
    Q_expected = 123.45**2.0 * 1.0 / np.asarray([[3.0, 2.0], [2.0, 1.0]])
    _, cov = cond.noise.to_multivariate_normal()
    assert testing.allclose(cov, Q_expected)

    # Test that the "diffuse_derivatives" are scaled correctly
    init, iwp, ssm = probdiffeq.prior_wiener_integrated(
        tcoeffs,
        output_scale=scale,
        diffuse_derivatives=3,
        diffuse_eps=1,
        ssm_fact=ssm_fact,
    )

    zero = np.zeros((1, 1, 1))
    nonzero = 123.45 * np.ones((1, 1, 1))
    assert testing.allclose(
        init.std, [zero, zero, nonzero, nonzero, nonzero], strict_shapes=True
    )

    # Test that for the wrong shape or type, an error is raised
    tcoeffs = [np.ones((1, 1, 1)), np.ones((1, 1, 1))]
    for shapes in [(), (1,), (1, 1)]:
        scale = 123.45 * np.ones(shapes)
        with testing.raises(ValueError, match="wrong shape"):
            _ = probdiffeq.prior_wiener_integrated(
                tcoeffs, output_scale=scale, ssm_fact=ssm_fact
            )


def test_output_scale_isotropic():

    # 1d problem, but "unusual" shapes. Values don't matter.
    tcoeffs = [np.ones((1, 1, 1)), np.ones((1, 1, 1))]
    scale = 123.45 * np.ones(())

    # Test that the transition covariances are scaled correctly
    init, iwp, ssm = probdiffeq.prior_wiener_integrated(
        tcoeffs, output_scale=scale, ssm_fact="isotropic"
    )

    cond = iwp(1.0, 1.0)
    Q_expected = 123.45**2.0 * 1.0 / np.asarray([[3.0, 2.0], [2.0, 1.0]])
    _, cov = cond.noise.to_multivariate_normal()
    assert testing.allclose(cov, Q_expected)

    # Test that the "diffuse_derivatives" are scaled correctly
    init, iwp, ssm = probdiffeq.prior_wiener_integrated(
        tcoeffs,
        output_scale=scale,
        diffuse_derivatives=3,
        diffuse_eps=1,
        ssm_fact="isotropic",
    )

    zero = np.zeros(())
    nonzero = 123.45 * np.ones(())
    assert testing.allclose(init.std, [zero, zero, nonzero, nonzero, nonzero])

    # Test that for the wrong shape or type, an error is raised
    tcoeffs = [np.ones((1, 1, 1)), np.ones((1, 1, 1))]
    for shapes in [(1,), (1, 1), (1, 1, 1)]:
        scale = 123.45 * np.ones(shapes)
        with testing.raises(ValueError, match="wrong shape"):
            _ = probdiffeq.prior_wiener_integrated(
                tcoeffs, output_scale=scale, ssm_fact="isotropic"
            )
