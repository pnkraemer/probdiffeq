"""Tests for integrated Wiener processes."""

from probdiffeq import probdiffeq
from probdiffeq.backend import np, testing


def test_diffuse_derivatives() -> None:
    tcoeffs = [2.0, 3.0]
    init, _ssm = probdiffeq.ssm_taylor(
        tcoeffs, diffuse_derivatives=3, diffuse_eps=123.0
    )
    assert testing.allclose(init.mean, [2.0, 3.0, 0.0, 0.0, 0.0])
    assert testing.allclose(init.std, [0.0, 0.0, 123.0, 123.0, 123.0])


def test_differential_variables_dense() -> None:
    tcoeffs = [np.asarray([1.0, 2.0, 3.0])]
    isdiff = [np.asarray([True, False, True])]
    init, _ssm = probdiffeq.ssm_taylor(
        tcoeffs, is_differential=isdiff, nondifferential_eps=0.123, ssm_fact="dense"
    )

    [m], [s] = init.mean, init.std
    assert testing.allclose(m, np.asarray([1.0, 2.0, 3.0]))
    assert testing.allclose(s, np.asarray([0.0, 0.123, 0.0]))

    with testing.raises(ValueError, match="wrong PyTree structure"):
        tcoeffs = [np.asarray([1.0, 2.0, 3.0])]
        isdiff = [np.asarray(False)]  # wrong shape
        _ = probdiffeq.ssm_taylor(tcoeffs, is_differential=isdiff)


def test_differential_variables_blockdiag() -> None:
    tcoeffs = [np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, 2.0, 3.0])]
    isdiff = [np.asarray([True, False, True]), np.asarray([False, False, True])]
    init, _ssm = probdiffeq.ssm_taylor(
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
        _ = probdiffeq.ssm_taylor(tcoeffs, is_differential=isdiff, ssm_fact="blockdiag")


def test_differential_variables_isotropic() -> None:
    tcoeffs = [np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, 2.0, 3.0])]
    isdiff = [np.asarray(True), np.asarray(False)]
    init, _ssm = probdiffeq.ssm_taylor(
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
        _ = probdiffeq.ssm_taylor(tcoeffs, is_differential=isdiff, ssm_fact="isotropic")
